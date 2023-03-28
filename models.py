import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
from commons import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2)).to(device)
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)).to(device)
      self.flows.append(modules.Flip()).to(device)

    self.post_pre = nn.Conv1d(1, filter_channels, 1).to(device)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1).to(device)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout).to(device)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2)).to(device)
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)).to(device)
      self.post_flows.append(modules.Flip()).to(device)

    self.pre = nn.Conv1d(in_channels, filter_channels, 1).to(device)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1).to(device)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout).to(device)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1).to(device)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x).to(device)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g).to(device)
      x = x + self.cond(g)
    x = self.convs(x, x_mask).to(device)
    x = self.proj(x) * x_mask
    x = x.to(device)

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout).to(device)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2).to(device)
    self.norm_1 = modules.LayerNorm(filter_channels).to(device)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2).to(device)
    self.norm_2 = modules.LayerNorm(filter_channels).to(device)
    self.proj = nn.Conv1d(filter_channels, 1, 1).to(device)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1).to(device)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x).to(device)
    if g is not None:
      g = torch.detach(g).to(device)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      emotion_embedding):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.emotion_embedding = emotion_embedding
    
    if self.n_vocab!=0:
      self.emb = nn.Embedding(n_vocab, hidden_channels)
      if emotion_embedding:
        self.emo_proj = nn.Linear(1024, hidden_channels)
      nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout).to(device)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1).to(device)

  def forward(self, x, x_lengths, emotion_embedding=None):
    if self.n_vocab!=0:
      x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    if emotion_embedding is not None:
      x = x + self.emo_proj(emotion_embedding.unsqueeze(1))
    x = (torch.transpose(x, 1, -1).to(device)) # [b, h, t]
    x_mask = (torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)).to(device)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True)).to(device)
      self.flows.append(modules.Flip()).to(device)

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x.to(device), x_mask.to(device), g=g.to(device) if g is not None else None, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x.to(device), x_mask.to(device), g=g.to(device) if g is not None else None, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1).to(device)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels).to(device)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1).to(device)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x.to(device)) * x_mask.to(device)
    x = self.enc(x.to(device), x_mask.to(device), g=g.to(device) if g is not None else None)
    stats = self.proj(x.to(device)) * x_mask.to(device)
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m.to(device)) * torch.exp(logs.to(device))) * x_mask.to(device)
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3).to(device)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)).to(device))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d).to(device))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False).to(device)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1).to(device)

    def forward(self, x, g=None):
        x = self.conv_pre(x.to(device))
        if g is not None:
          g = torch.detach(g).to(device)
          x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    emotion_embedding=False,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        emotion_embedding).to(device)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels).to(device)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels).to(device)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels).to(device)

    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels).to(device)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels).to(device)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels).to(device)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, emotion_embedding=None):
    x, m_p, logs_p, x_mask = self.enc_p(x.to(device), x_lengths.to(device), emotion_embedding.to(device) if emotion_embedding is not None else None)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1).to(device)
    else:
      g = None

    if self.use_sdp:
      logw = self.dp(x.to(device), x_mask.to(device), g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x.to(device), x_mask.to(device), g=g)
    w = torch.exp(logw) * x_mask.to(device) * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask.to(device), 2) * torch.unsqueeze(y_mask.to(device), -1)
    attn = commons.generate_path(w_ceil.to(device), attn_mask.to(device))

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p.to(device)) * torch.exp(logs_p.to(device)) * noise_scale
    z = self.flow(z_p.to(device), y_mask.to(device), g=g.to(device) if g is not None else None, reverse=True)
    o = self.dec((z * y_mask.to(device))[:,:,:max_len], g=g.to(device) if g is not None else None)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1).to(device)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1).to(device)
    z, m_q, logs_q, y_mask = self.enc_q(y.to(device), y_lengths.to(device), g=g_src.to(device))
    z_p = self.flow(z.to(device), y_mask.to(device), g=g_src.to(device))
    z_hat = self.flow(z_p.to(device), y_mask.to(device), g=g_tgt.to(device), reverse=True)
    o_hat = self.dec(z_hat.to(device) * y_mask.to(device), g=g_tgt.to(device))
    return o_hat, y_mask, (z, z_p, z_hat)
