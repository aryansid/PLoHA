import json
import pandas as pd
import csv
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from openai import OpenAI
from dotenv import load_dotenv

# dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
# dataset.save_to_disk("medical_datasets/medical_meadow_medical_flashcards")

client = OpenAI()

PROMPT = """
You are a medical expert. You are given a medical question and its corresponding answer. Your task is to generate a rationale that explains how the answer stems from the question. YOU MUST BE CONCISE. Do not output anything aside from the rationale.
"""

def gpt(system, user, model="gpt-3.5-turbo-0125"):
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
    

class PrepareDataset: 
    def __init__(self, data_root, dataset_name): 
        self.data_root = data_root
        self.dataset_name = dataset_name
        
    def create_dataset(self): 
        # save as JSON ! 
        pass 

    
    def post_process_dataset(self):
        dataset_path = f'{self.data_root}/{self.dataset_name}.txt'
        with open(dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        train_lines, test_lines = train_test_split(lines, test_size=0.2) 
        
        train_file_path = f'{self.data_root}/{self.dataset_name}_train.json'
        test_file_path = f'{self.data_root}/{self.dataset_name}_test.json'
        self._save_lines_to_json(train_lines, train_file_path)
        self._save_lines_to_json(test_lines, test_file_path)
    
    # If in case data was stored in format other than JSON    
    def _save_lines_to_json(self, lines, file_path): 
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in csv.reader(lines, delimiter=',', quotechar='"'):
                if len(line) == 3:
                    data = {
                        "question": line[0].strip(),
                        "answer": line[1].strip(),
                        "rationale": line[2].strip()
                    }
                    json.dump(data, file)
                    file.write('\n') 


    

if __name__ == "__main__": 
    data_root = "cot_medical_datasets"
    dataset_name = "anki_flashcards"
    
    preparer = PrepareDataset(data_root=data_root, dataset_name=dataset_name)
    preparer.post_process_dataset()
    
    # input_file_path = "mmmf"
    # output_file_path = "mmmf_1.txt"
    
    # seen_inputs = set()
    
    # with open(input_file_path, 'r', encoding='utf-8') as input_file, \
    #     open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:
            
    #     reader = csv.reader(input_file)
    #     writer = csv.writer(output_file)
        
    #     current_iteration = 0
        
    #     for parts in reader: 
    #         current_iteration += 1
    #         print(f"Iteration {current_iteration} completed")
            
    #         if len(parts) == 5 and parts[3] not in seen_inputs: 
    #             formatted_line = [parts[3], parts[1], parts[4]]
    #             writer.writerow(formatted_line)
    #             seen_inputs.add(parts[3])
                
            
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


  
  
