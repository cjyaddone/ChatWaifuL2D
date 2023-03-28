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
import argparse
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import threading
import time

chinese_model_path = ".\model\CN\model.pth"
chinese_config_path = ".\model\CN\config.json"
japanese_model_path = ".\model\H_excluded.pth"
japanese_config_path = ".\model\config.json"
inputVoice = -1
status = False

#########################################
#Voice Recognition
q = queue.Queue()
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-f", "--filename", type=str, metavar="FILENAME",
    help="audio file to store recording to")
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)
try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info["default_samplerate"])

    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None
        
except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)

def voice_input(language):
    model = Model(lang=language)
    print("You:")
    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                           dtype="int16", channels=1, callback=callback):

        rec = KaldiRecognizer(model, args.samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                a = json.loads(rec.Result())
                a = str(a['text'])
                a = ''.join(a.split())
                if(len(a) > 0):
                    print(a)
                    user_input = a
                    return user_input
            if dump_fn is not None:
                dump_fn.write(data)

######Socket######
import socket
ip_port = ('127.0.0.1', 9000)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
s.bind(ip_port)
s.listen(5)


#### CHATGPT INITIALIZE ####
from pyChatGPT import ChatGPT


### TTS ###
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


def generateSound():
    global status,model_id,speaker_id
    if model_id == 0:
        model = chinese_model_path
        config = chinese_config_path
    elif model_id == 1:
        model = japanese_model_path
        config = japanese_config_path
        

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
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
            while True:
                if shengcheng!='':
                    text = shengcheng
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

                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

                write(out_path, hps_ms.data.sampling_rate, audio)
                status = False
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
        client.send("已加载".encode())
        inputMethod = int(client.recv(1024).decode()) #inputMethod: Keyboard/Voice
        if(inputMethod == 0): #Keyboard
            print("设置为键盘输入")
        elif(inputMethod == 1): #voice
            print("设置为语音输入")
            inputVoice = int(client.recv(1024).decode())  # voiceInputMethod: CN/JP/EN
            if(inputVoice == 0):
                voiceModel = "cn"
                print("设置中文为识别语言")
            elif(inputVoice == 1):
                voiceModel = "ja"
                print("设置日本语为识别语言")
            elif(inputVoice == 2):
                voiceModel = "en-us"
                print("设置英语为识别语言")

        outputMethod = int(client.recv(1024).decode()) #outputMethod: CN/JP
        if(outputMethod == 0):
            print("设置为中文输出")
        elif(outputMethod == 1):
            print("设置为日语输出")

        speaker = int(client.recv(1024).decode())  # outputMethod: CN/JP
        model_id =outputMethod
        speaker_id=speaker
        generatevoice=threading.Thread(target=generateSound)
        generatevoice.start()
    while True:
        if(inputMethod == 0): #Keyboard
            total_data = bytes()
            while True:
                data = client.recv(1024)
                total_data += data
                if len(data) < 1024:
                    break
            question = total_data.decode()

        elif(inputMethod == 1): #Voice
            question = voice_input(voiceModel)
            client.send(question.encode())

        print("Question Received: " + question)

        if(outputMethod == 1 and (inputVoice == 0 or inputVoice == 2 or inputVoice == -1)):
            question = question + " 使用日本语回答"
        if (outputMethod == 0 and (inputVoice == 1 or inputVoice == 2 or inputVoice == -1)):
            question = question + " 使用中文回答"
        resp = api.send_message(question)
        answer = resp["message"].replace('\n', '')
        answerG = answer
        print("ChatGPT:")
        print(answer)
        if(outputMethod == 0):
            answerG = "[ZH]" + answer + "[ZH]"
        shengcheng=answerG
        #generateSound(answerG,speaker,outputMethod)
        while status!=True:
            time.sleep(0.2)
            continue
        shengcheng=''
        # convert wav to ogg
        src = "./output.wav"
        dst = "./ChatWaifuGameL2D/game/audio/test.ogg"
        sound = AudioSegment.from_wav(src)
        sound.export(dst, format="ogg")
        # send response to UI
        client.send(answer.encode())
        # finish playing audio
        print(client.recv(1024).decode())