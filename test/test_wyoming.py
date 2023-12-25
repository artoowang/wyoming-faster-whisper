import numpy as np
import pdb
import socket
import whisper

audio_data_fp32 = whisper.load_audio('play-recording.wav')
assert audio_data_fp32.dtype == np.float32
assert len(audio_data_fp32.shape) == 1

audio_data = (audio_data_fp32 * 32768.0).astype(np.int16)

WYOMING_IP = '0.0.0.0'
WYOMING_PORT = 10300

wyoming_channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
wyoming_channel.connect((WYOMING_IP, WYOMING_PORT))
wyoming_channel.send('{"type":"describe"}'.encode('utf-8'))
wyoming_channel.close()