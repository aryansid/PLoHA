from datasets import load_dataset, load_from_disk
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import csv

# dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
# dataset.save_to_disk("medical_datasets/medical_meadow_medical_flashcards")

client = OpenAI()

PROMPT = """
You are a medical expert. You are given a medical question and its corresponding answer. Your task is to generate a rationale that explains how the answer stems from the question. YOU MUST BE CONCISE. Do not output anything aside from the rationale.
"""

def gpt(system, user, model="gpt-4-1106-preview"):
    messages = [
        {
            "role": "system", 
            "content": system
        },
        {
            "role": "user", 
            "content": user
        }
    ]
    
    try: 
        response = client.chat.completions.create(
        model=model,
        messages=messages, 
        temperature=0.5,
        )
        
        gpt_response = response.choices[0].message.content.strip()

        return gpt_response
    except Exception as e: 
        print(f"Error in OpenAI API call: {e}")
        return None

if __name__ == "__main__": 
    input_file_path = "mmmf"
    output_file_path = "mmmf_1.txt"
    
    seen_inputs = set()
    
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:
            
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        
        current_iteration = 0
        
        for parts in reader: 
            current_iteration += 1
            print(f"Iteration {current_iteration} completed")
            
            if len(parts) == 5 and parts[3] not in seen_inputs: 
                formatted_line = [parts[3], parts[1], parts[4]]
                writer.writerow(formatted_line)
                seen_inputs.add(parts[3])
                
                
# CHANGE MODEL NOW. CHANGE MODEL !!!!!!!!!!!!!!!!!!!!
            
#   dataset = load_from_disk("medical_datasets/medical_meadow_medical_flashcards")
#   df = pd.DataFrame(dataset['train'])
  
#   rationales = []
#   for index, row in df.iterrows():
#     print(f"Index: {index}")
#     qa = f"Question: {row['input']}\nAnswer: {row['output']}"
#     rationale = gpt(PROMPT, qa)
#     rationales.append(rationale)
    
#   df['rationale'] = rationales
  
#   df.to_csv("mmmf")


  
  
