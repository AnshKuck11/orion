from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cuda", compute_type="float16")

segments, info = model.transcribe("recording.wav")

text = ""
for segment in segments:
    text += segment.text

print("You said: " + text)

