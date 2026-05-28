import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000 # Sample rate in Hz
SECONDS = 5 # Duration of recording in seconds

print("Recording in 3")
print("Recording in 2")
print("Recording in 1")
print("Speak")

recording = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

sd.wait() # Wait until recording is finished
print("Recording finished")

sf.write('recording.wav', recording, SAMPLE_RATE)
print("Saved recording to 'recording.wav'")