from scipy.io.wavfile import write
from text import text_to_sequence
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from pydub import AudioSegment
from torch import no_grad, LongTensor
import logging


######Socket######
import socket
ip_port = ('127.0.0.1', 9000)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
s.bind(ip_port)
s.listen(5)


#### CHATGPT INITIALIZE ####
from pyChatGPT import ChatGPT


### TTS ###
CN_Model = r".\model\CN\model.pth"
CN_Config = r".\model\CN\config.json"

JP_Model =r".\model\H_excluded.pth"
JP_Config = r".\model\config.json"

voiceLanguage = 0
speakerID = 0


logging.getLogger('numba').setLevel(logging.WARNING)

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def generateSound(inputString):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False
    if(voiceLanguage == 0):
        model = CN_Model
        config = CN_Config
    elif(voiceLanguage == 1):
        model = JP_Model
        config = JP_Config

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        if not emotion_embedding:
            # while True:
            if (1 == 1):
                choice = 't'
                if choice == 't':
                    text = inputString
                    if text == '[ADVANCED]':
                        text = "我不会说"

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    speaker_id = speakerID
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                            0, 0].data.cpu().float().numpy()


                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')


if __name__ == "__main__":
    print("链接已生成，等待UI连接")
    client, client_addr = s.accept()
    print("链接已建立,等待接受token")

    total_data = bytes()
    while True:
        data = client.recv(1024)
        total_data += data
        if len(data) < 1024:
            break
    session_token = total_data.decode()

    if(session_token):
        print("收到token:"+ session_token)
        api = ChatGPT(session_token)
        voiceLanguage = int(client.recv(1024).decode())
        if(voiceLanguage == 0):
            print("设置中文为输出语言")
        elif(voiceLanguage == 1):
            print("设置日语为输出语言")

    speakerID = int(client.recv(1024).decode())
    print("语音输出编号：" + str(speakerID))

    while True:
        total_data = bytes()
        while True:
            data = client.recv(1024)
            total_data += data
            if len(data) < 1024:
                break

        question = total_data.decode()
        print("Question Received: "+ question)
        #question = question + " 你的回答不能超过80个字。"
        if(voiceLanguage == 1):
            question = question + " 使用日本语"
        if(len(question) > 0):
            resp = api.send_message(question)
            answer = resp["message"].replace('\n', '')
            print("ChatGPT:")
            print(answer)
            if (voiceLanguage == 0):
                response= "[ZH]" + str(answer) + "[ZH]"
            elif(voiceLanguage == 1):
                response = str(answer)
            generateSound(response)

            # convert wav to mp3
            src = "./output.wav"
            dst = "./Game/game/audio/test.ogg"
            #dst = "./ChatWaifuGameL2D/game/audio/test.ogg"
            sound = AudioSegment.from_wav(src)
            sound.export(dst, format="ogg")

            #send response to UI
            client.send(answer.encode())