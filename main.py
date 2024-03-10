import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import json

# Custom Dataset class for handling medical QA data
class MedicalQADataset(Dataset):
    def __init__(self, tokenizer, data_file, max_len=512):
        self.tokenizer = tokenizer  # T5 tokenizer
        self.inputs = []  # Store input texts
        self.targets = []  # Store target texts

        # Load data from JSON file
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            # Prepare inputs for both answering and rationale generation, if available
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

        # Tokenize input and target texts
        input_enc = self.tokenizer(input_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        target_enc = self.tokenizer(target_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': input_enc['input_ids'].flatten(),
            'attention_mask': input_enc['attention_mask'].flatten(),
            'labels': target_enc['input_ids'].flatten()
        }

# Train the model for one epoch
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()  # Set model to training mode
    losses = []  # Store losses for each step

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Extract the loss

        losses.append(loss.item())

        loss.backward()  # Backward pass to compute gradient of loss wrt parameters
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Clear gradients

    return np.mean(losses)  # Return average loss

# Evaluate the model
def eval_model(model, data_loader, device):
    model = model.eval()  # Set model to evaluation mode
    losses = []

    with torch.no_grad():  # Inference mode, gradients not needed
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    return np.mean(losses)  # Return average loss

# Run the training process
def run_training(model, train_data_loader, val_data_loader, optimizer, device, scheduler, epochs=3):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

        val_loss = eval_model(model, val_data_loader, device)
        print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}')

if __name__ == "__main__": 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base')  # Initialize tokenizer
  model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base').to(device)  # Load model and send to device

  # Prepare datasets and data loaders for training and validation
  train_dataset = MedicalQADataset(tokenizer, 'anki_flashcards_train.json')
  train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_dataset = MedicalQADataset(tokenizer, 'anki_flashcards_test.json')
  val_data_loader = DataLoader(val_dataset, batch_size=64)

  # Initialize optimizer and learning rate scheduler
  optimizer = AdamW(model.parameters(), lr=5e-5)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * 3)

  # Start the training and validation process
  run_training(model, train_data_loader, val_data_loader, optimizer, device, scheduler, epochs=3)

  # Save the model
  model.save_pretrained("my_model")
  tokenizer.save_pretrained("my_tokenizer")