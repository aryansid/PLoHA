from datasets import load_dataset, load_from_disk
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

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
  dataset = load_from_disk("medical_datasets/medical_meadow_medical_flashcards")
  df = pd.DataFrame(dataset['train'])
  print(len(df))
#   rationales = []
#   for index, row in df.iterrows():
#     print(f"Index: {index}")
#     qa = f"Question: {row['input']}\nAnswer: {row['output']}"
#     rationale = gpt(PROMPT, qa)
#     rationales.append(rationale)
    
#   df['rationale'] = rationales
  
#   df.to_csv("mmmf")

    # import json
    # file_path = "datasets/cqa/llm/train_CoT_0.json"

    # # Reading the content of the file
    # with open(file_path, 'r') as file:
    #     data = json.load(file)

    # print(data[0])
    # print("*** \n")
    # print(data[1])
  
  
  
