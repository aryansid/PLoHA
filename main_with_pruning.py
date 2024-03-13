import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import json
from torch.nn.utils import prune
import openai

class MedicalQADataset(Dataset):
    def __init__(self, tokenizer, data_file, max_len=512):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            self.inputs.append(f"answer: {data['question']}")
            self.targets.append(data['answer'])
            if data.get('rationale'):
                self.inputs.append(f"rationale: {data['question']}")
                self.targets.append(data['rationale'])

        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_enc = self.tokenizer(input_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        target_enc = self.tokenizer(target_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': input_enc['input_ids'].flatten(),
            'attention_mask': input_enc['attention_mask'].flatten(),
            'labels': target_enc['input_ids'].flatten()
        }

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    return np.mean(losses)

def apply_structured_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=0)
            prune.remove(module, 'weight')
    return model

def generated_feedback_gpt3(instruction_prompt):
    # This function uses OpenAI's Completion API to generate feedback
    # The prompt_text should be formatted to include the instruction, input, and output
    instruction_prompt = (
    "I want you to act as a reward model trained in the field of medical response evaluation. "
    "I will present you with an input, a predicted output, and the target output. "
    "I want you to rate the predicted output on a scale of 1-10 as "
    "as if it were to come from a doctor, and return that scalar value as feedback. "
    "Please make your decision based on the following constitutional principles:\n"
    "1) The reward rating shall be as similar as possible to how a licensed, well-renowned doctor's reward rating would be.\n"
    "2) The answer shall align with state-of-the-art medical knowledge and theory.\n"
    "3) The answer is supposed to be from a real-life licensed doctor answering in a real-life clinical session, not from a website, a forum, "
    "or anything similar. If links, phone numbers or such information "
    "is in the answer, it is most likely not the best answer.\n"
    "4) The answer shall be harmless, helpful, and empathetic.\n"
    "5) The answer shall be similar to what a real-life, "
    "human psychologist would write in a chat message.\n\n"
    "Please output the scalar reward value. Output nothing else. "
    "Here is the input, output, and predicted output, separated by new lines."
)

    response = openai.Completion.create(
        engine="davinci",
        prompt=instruction_prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def run_rlhf_training(model, data_loader, tokenizer, device, epochs=3):
    model.train()

    for epoch in range(epochs):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Convert model outputs to text
            predicted_output = tokenizer.decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            
            # Generate a prompt for feedback using GPT-3
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            feedback_prompt = f"The following is a medical query and a model-generated response:\n\nQuery: {input_text}\nResponse: {predicted_output}\n\nRate the response on a scale of 1-10 for accuracy and relevance."
            
            # Obtain feedback
            feedback = generate_feedback_gpt3(feedback_prompt)
            
            # Parse feedback and convert to a numerical reward
            # Placeholder: Assume feedback is a number; convert to float for reward
            # In a real setup, you'd need to parse and possibly validate the feedback format
            reward = float(feedback)
            
            # Apply a simple reward mechanism to adjust weights
            # This is a placeholder for a PPO or other RL algorithm
            loss = outputs.loss - reward
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch + 1}/{epochs}, Batch loss: {loss.item():.4f} (Feedback reward: {reward})")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    model = apply_structured_pruning(model)

    train_dataset = MedicalQADataset(tokenizer, 'anki_flashcards_train.json')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * epochs)

    run_rlhf_training(model, train_data_loader, tokenizer, device, epochs=3)

    model.save_pretrained("rlhf_trained_model")
    tokenizer.save_pretrained("rlhf_trained_tokenizer")