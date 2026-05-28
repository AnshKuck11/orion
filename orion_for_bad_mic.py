from peft import PeftModel
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import edge_tts
import os
import re
import time

SAMPLE_RATE = 16000
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ORION_MODEL = "orion_model"
INACTIVITY_TIMEOUT = 30

print("Initializing Orion")

whisper = WhisperModel("small.en", device="cpu", compute_type="int8")

print("Whisper loaded...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cuda")
brain = PeftModel.from_pretrained(base, ORION_MODEL)

print("Brain loaded...")

print("Voice loaded...")

conversation = [
    {"role": "system", "content": "You are Orion, a composed and highly intelligent AI assistant. You are calm, observant, concise, and quietly confident. Your humor is subtle and understated. You speak naturally, with the demeanor of an experienced butler and advisor."}
]

def is_speech(audio_chunk, threshold=0.0010):
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold

def count_speech_frames(chunk):
    frame_size = int(0.1 * SAMPLE_RATE)
    speech_frames = 0
    for i in range(0, len(chunk) - frame_size, frame_size):
        frame = chunk[i:i + frame_size]
        if is_speech(frame):
            speech_frames += 1
    return speech_frames

def wait_for_wake_word():
    wake_words = ["orion", "o'ryan", "oh ryan", "ryan", "good morning", "hello", "good evening", "hey"]
    print("Sleeping... say Orion to wake up")
    while True:
        chunk = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        if count_speech_frames(chunk) >= 3:
            sf.write("wake.wav", chunk, SAMPLE_RATE)
            segments, _ = whisper.transcribe("wake.wav")
            os.remove("wake.wav")
            text = "".join([s.text for s in segments]).lower()
            if any(word in text for word in wake_words):
                print(f"Wake word detected: {text.strip()}")
                return text

def listen():
    print("Listening... speak now! (5 seconds)")
    time.sleep(0.5)  # small delay so you don't get cut off
    recording = sd.rec(
        int(5 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    sf.write("temp.wav", recording, SAMPLE_RATE)
    segments, _ = whisper.transcribe("temp.wav")
    text = "".join([s.text for s in segments])
    os.remove("temp.wav")
    return text.strip()

def think(user_input):
    conversation.append({"role": "user", "content": user_input})
    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = brain.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            min_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full.split("[/INST]")[-1].strip()
    response = re.sub(r'http\S+|www\S+|\[.*?\]\(.*?\)', '', response).strip()
    response = re.sub(r'^Orion:\s*', '', response).strip()
    conversation.append({"role": "assistant", "content": response})
    return response

def speak(text):
    print(f"Orion: {text}")
    async def _speak():
        communicate = edge_tts.Communicate(text, voice="en-US-EricNeural")
        await communicate.save("speech.mp3")
    try:
        asyncio.run(_speak())
        from playsound3 import playsound
        try:
            playsound("speech.mp3")
        except KeyboardInterrupt:
            pass
        finally:
            if os.path.exists("speech.mp3"):
                os.remove("speech.mp3")
    except Exception as e:
        print(f"Voice error: {e}")
        print("Continuing without audio...")

print("Orion is online. Speak after 'Sleeping...'")
print("Press Ctrl+C to quit")

while True:
    wake_text = wait_for_wake_word()
    speak("Yes, Master.")
    last_active = time.time()
    while True:
        user_input = listen()
        if user_input:
            listen.silent_count = 0
            last_active = time.time()
            print(f"You: {user_input}")
            user_input = re.sub(r'http\S+|www\S+|\[.*?\]\(.*?\)', '', user_input).strip()
            response = think(user_input)
            if response:
                speak(response)
                time.sleep(0.5)
            else:
                print("Orion: ...")
        else:
            print("Nothing heard, listening again...")
            silent_count = getattr(listen, 'silent_count', 0) + 1
            listen.silent_count = silent_count
            if silent_count >= 3 or time.time() - last_active > INACTIVITY_TIMEOUT:
                listen.silent_count = 0
                print("Going back to sleep...")
                speak("Going to sleep, Master.")
                break