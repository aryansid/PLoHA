# *****************
# Evaluate model performance on USMLE 
# ****************

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json

# Load the model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()
model.to('cpu')

# Load the sentence transformer model for embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
with open("USMLE.json", "r") as f: 
  dataset = json.load(f)

is_correct = 0
count = 0
# Iterate through the dataset
for example in tqdm(dataset):
    input_text = example['question']
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cpu')

    # Generate an answer with the model
    output = model.generate(input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Model prediction: {answer} \n")
    print(f"Correct answer: {example['answer']} \n")
    
    if count == 5: 
      break

    # Compute embeddings for model answer and correct answer
    answer_embedding = embed_model.encode(answer, convert_to_tensor=True)
    correct_answer_embedding = embed_model.encode(example['answer'], convert_to_tensor=True)

    # Compute cosine similarity
    cosine_sim = util.pytorch_cos_sim(answer_embedding, correct_answer_embedding)

    # Check if similarity is above the threshold
    if cosine_sim > 0.3:
        is_correct += 1

# Calculate accuracy
accuracy = is_correct / len(dataset)
print(f"Accuracy: {accuracy:.4f}")

def format_example(example):
    question = example['question']
    options = example['options']
    options_text = " ".join([f"{key}: {value}" for key, value in options.items()])
    input_text = f"{question} Options: {options_text}"
    return input_text
  
def format_dataset(): 
  dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
  
  new_dataset = []

  for example in dataset: 
    formatted_question = format_example(example)

    answer = example["answer"]
    
    qa_pair = {"question": formatted_question, "answer": answer}
    
    new_dataset.append(qa_pair)

  with open("USMLE.json", "w") as f: 
    json.dump(new_dataset, f, indent=4)


