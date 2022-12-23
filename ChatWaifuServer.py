from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from pydub import AudioSegment
from torch import no_grad, LongTensor
import logging
from winsound import PlaySound
import argparse
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

chinese_model_path = ".\model\CN\model.pth"
chinese_config_path = ".\model\CN\config.json"
japanese_model_path = ".\model\H_excluded.pth"
japanese_config_path = ".\model\config.json"

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

def voice_input_cnjp():
    model = Model(lang="cn")
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
                    user_input = a + " 使用日本语"
                    return user_input
            if dump_fn is not None:
                dump_fn.write(data)

def voice_input_cncn():
    model = Model(lang="cn")
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

def voice_input_jpjp():
    model = Model(lang="ja")
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
                    user_input = a + " 使用日本语"
                    return user_input
            if dump_fn is not None:
                dump_fn.write(data)

def voice_input_jpcn():
    model = Model(lang="ja")
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

def voice_input_enjp():
    model = Model(lang="en-us")
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
                    user_input = a + " 使用日本语"
                    return user_input
            if dump_fn is not None:
                dump_fn.write(data)

def voice_input_encn():
    model = Model(lang="en-us")
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

def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_model_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_language_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

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


def generateSound(inputString, id, model_id):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

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
            #while True:
            if(1 == 1):
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
                    
                    speaker_id = id 
                    out_path = "output.wav"

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

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

        inputMethod = int(client.recv(1024).decode()) #inputMethod: Keyboard/Voice
        if(inputMethod == 0): #Keyboard
            print("设置为键盘输入")
            outputMethod = int(client.recv(1024).decode()) #outputMethod: CN/JP
            if(outputMethod == 0): #CN
                print("设置为中文输出")
                speakerID = int(client.recv(1024).decode()) #modelChoiceCN: who?(4 total)
                print("语音输出编号：" + str(speakerID))
            elif(outputMethod == 1): #JP
                print("设置为日语输出")
                speakerID = int(client.recv(1024).decode()) #modelChoiceJP: who?(7 total)
                print("语音输出编号：" + str(speakerID))
        elif(inputMethod == 1): #Voice
            print("设置为语音输入")
            inputVoice = int(client.recv(1024).decode()) #voiceInputMethod: CN/JP/EN
            if(inputVoice == 0): #CN
                print("设置为中文输入")
                outputMethod = int(client.recv(1024).decode()) #outputMethod: CN/JP
                if(outputMethod == 0):
                    print("设置为中文输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceCN: who?(4 total)
                    print("语音输出编号：" + str(speakerID))
                elif(outputMethod == 1):
                    print("设置为日语输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceJP: who?(7 total)
                    print("语音输出编号：" + str(speakerID))
            elif(inputVoice == 1): #JP
                print("设置为日语输入")
                outputMethod = int(client.recv(1024).decode()) #outputMethod: CN/JP
                if(outputMethod == 0):
                    print("设置为中文输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceCN: who?(4 total)
                    print("语音输出编号：" + str(speakerID))
                elif(outputMethod == 1):
                    print("设置为日语输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceJP: who?(7 total)
                    print("语音输出编号：" + str(speakerID))
            elif(inputVoice == 2): #EN
                print("设置为英语输入")
                outputMethod = int(client.recv(1024).decode()) #outputMethod: CN/JP
                if(outputMethod == 0):
                    print("设置为中文输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceCN: who?(4 total)
                    print("语音输出编号：" + str(speakerID))
                elif(outputMethod == 1):
                    print("设置为日语输出")
                    speakerID = int(client.recv(1024).decode()) #modelChoiceJP: who?(7 total)
                    print("语音输出编号：" + str(speakerID))     


    while True:
        if(inputMethod == 0): #Keyboard
            total_data = bytes()
            while True:
                data = client.recv(1024)
                total_data += data
                if len(data) < 1024:
                    break
            question = total_data.decode()
            print("Question Received: "+ question)

            if(outputMethod == 1):
                question = question + " 使用日本语"

            if(len(question) > 0):
                resp = api.send_message(question)
                answer = resp["message"].replace('\n', '')
                if(resp == "quit()"):
                    break
                print("ChatGPT:")
                print(answer)
                if (outputMethod == 0):
                    response= "[ZH]" + str(answer) + "[ZH]"
                elif(outputMethod == 1):
                    response = str(answer)
                generateSound(response, int(speakerID), int(outputMethod))
        elif(inputMethod == 1): #Voice
            if(inputVoice == 0 and outputMethod == 0): #CN voice input, CN output
                temp = voice_input_cncn()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound("[ZH]"+answer+"[ZH]", int(speakerID), int(outputMethod))
            elif(inputVoice == 0 and outputMethod == 1): #CN voice input, JP output
                temp = voice_input_cnjp()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound(answer, int(speakerID), int(outputMethod))
            elif(inputVoice == 1 and outputMethod == 0): #JP voice input, CN output
                temp = voice_input_jpcn()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound("[ZH]"+answer+"[ZH]", int(speakerID), int(outputMethod))
            elif(inputVoice == 1 and outputMethod == 1): #JP voice input, JP output
                temp = voice_input_jpjp()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound(answer, int(speakerID), int(outputMethod))
            elif(inputVoice == 2 and outputMethod == 0): #EN voice input, CN output
                temp = voice_input_encn()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound("[ZH]"+answer+"[ZH]", int(speakerID), int(outputMethod))
            elif(inputVoice == 2 and outputMethod == 1): #EN voice input, JP output
                temp = voice_input_enjp()
                client.send(temp.encode())
                resp = api.send_message(temp)
                if(resp == "quit()"):
                    break
                answer = resp["message"].replace('\n','')
                print("ChatGPT:")
                print(answer)
                generateSound(answer, int(speakerID), int(outputMethod))
            
        # convert wav to ogg
        src = "./output.wav"
        dst = "./ChatWaifuGameL2D/game/audio/test.ogg"
        sound = AudioSegment.from_wav(src)
        sound.export(dst, format="ogg")
        # send response to UI
        client.send(answer.encode())
        # finish playing audio
        print(client.recv(1024).decode())