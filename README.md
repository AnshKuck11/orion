# Orion — Local AI Assistant

Orion is a Jarvis-inspired AI assistant that runs entirely on your own hardware. No cloud, no API keys, no subscription. Just a locally hosted AI you can have a real conversation with.

I built this because every "Jarvis" video I kept seeing online was just a wrapper around OpenAI or Claude with a fancy UI on top. That always felt like cheating to me. I wanted to build something I actually trained myself, that lives on my machine, and that I understand end to end.

## What it does

- Wakes up when you say its name
- Listens to your voice and transcribes it using Whisper
- Responds using a fine-tuned Mistral 7B model running locally on your GPU
- Speaks back using a neural text-to-speech voice
- Goes back to sleep after inactivity
- Runs completely offline once set up

## How the brain works

The core model is Mistral 7B, a 7 billion parameter language model. Training all 7 billion parameters from scratch would take months and cost thousands of dollars, so instead I used a technique called LoRA (Low Rank Adaptation).

LoRA freezes the original model weights entirely and adds two small trainable matrices alongside the attention layers. Instead of updating 7 billion parameters, only about 6.8 million get trained which is roughly 0.09% of the model. This makes fine tuning possible on a personal GPU in under an hour.

I wrote 100+ custom training conversations to shape Orion's personality, then trained on those using a PyTorch training loop. The model went from a loss of 32.5 on step one to an average of 0.28 by the end of training.

## Stack

- **LLM** — Mistral 7B Instruct via Hugging Face Transformers
- **Fine tuning** — LoRA via PEFT
- **Speech to text** — faster-whisper (OpenAI Whisper)
- **Text to speech** — edge-tts (Microsoft Eric Neural voice)
- **Audio** — sounddevice, soundfile
- **Training** — PyTorch with AdamW optimizer

## Hardware

Built and trained on:
- Intel Core Ultra 9 (24 core)
- NVIDIA RTX 5080 (16GB VRAM)
- 64GB DDR5 RAM

## Setup

```bash
git clone https://github.com/AnshKuck11/orion.git
cd orion
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install faster-whisper transformers accelerate peft bitsandbytes
pip install sounddevice soundfile edge-tts playsound3 datasets
```

You'll need to supply your own training data and run `train.py` to generate the model weights. The trained weights are not included in this repo.

```bash
python train.py
python orion_for_bad_mic.py
```

## Roadmap

- [x] Voice pipeline (wake word, STT, TTS)
- [x] Fine tuned personality via LoRA
- [ ] Persistent memory between sessions
- [ ] Tool use (open apps, search web, control PC)
- [ ] GUI
- [ ] Vision

---

Built by Ansh Kuckreja
