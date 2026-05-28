import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000 # Sample rate in Hz
SECONDS = 5 # Duration of recording in seconds

model = WhisperModel("base.en", device="cuda", compute_type="float16")

print("Speak, you have 5 seconds")

recording = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

sd.wait()
print("Transcribing")

sf.write('recording.wav', recording, SAMPLE_RATE)

segmenents, info = model.transcribe("recording.wav")

text = ""
for segment in segmenents:
    text += segment.text

print("You said: " + text)