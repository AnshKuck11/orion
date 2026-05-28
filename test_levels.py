import sounddevice as sd
import numpy as np

print("Testing mic levels for 10 seconds, speak normally...")
print("Watch the RMS values — this tells us what threshold to use\n")

for i in range(20):
    chunk = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    rms = np.sqrt(np.mean(chunk**2))
    bar = "█" * int(rms * 1000)
    print(f"RMS: {rms:.4f} {bar}")