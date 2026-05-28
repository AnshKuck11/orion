import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAINING_DATA_DIR = "training_data"
OUTPUT_DIR = "orion_model"
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda"
)

print("Applying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading training data...")

def load_conversations():
    conversations = []
    for filename in os.listdir(TRAINING_DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(TRAINING_DATA_DIR, filename), "r") as f:
                data = json.load(f)
                conversations.append(data["conversations"])
    return conversations

def format_conversation(messages):
    text = ""
    for message in messages:
        if message["role"] == "system":
            text += f"<s>[INST] {message['content']}\n"
        elif message["role"] == "user":
            text += f"{message['content']} [/INST]"
        elif message["role"] == "assistant":
            text += f" {message['content']}</s>"
    return text

class ConversationDataset(Dataset):
    def __init__(self, conversations):
        self.examples = []
        for conv in conversations:
            text = format_conversation(conv)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

conversations = load_conversations()
print(f"Loaded {len(conversations)} conversations")

dataset = ConversationDataset(conversations)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
print("Watch the loss drop — that number is how wrong Orion is right now\n")

model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.4f}\n")

print("Saving Orion...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Orion saved to {OUTPUT_DIR}")