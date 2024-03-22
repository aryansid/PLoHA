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
    
def combine_datasets(path1, path2, destination_path): 
    with open(destination_path, "w") as dest_file: 
        with open(path1, "r") as file1:
            for line in file1: 
                obj = json.loads(line)
                dest_file.write(json.dumps(obj) + "\n")
                
        with open(path2, 'r') as file2:
            for line in file2:
                obj = json.loads(line)
                dest_file.write(json.dumps(obj) + '\n')

class PrepareMCQDataset(): 
    def __init__(self, data_path, dataset_root, dataset_name):
        self.data_path = data_path
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        
    def format_entry(self, example):
        question = example['question']
        options = example['options']
        options_text = " ".join([f"{key}: {value}" for key, value in options.items()])
        input_text = f"{question} Options: {options_text}"
        return input_text
    
    def create_dataset(self):
        raise NotImplementedError
    
    def split_dataset(self): 
        data = []
        with open(f"{self.dataset_root}/{self.dataset_name}.json", "r") as file: 
            for line in file: 
                data.append(json.loads(line))
            
        train_data, test_data = train_test_split(data, test_size=0.2)
        
        with open(f"{self.dataset_root}/{self.dataset_name}_train.json", "w") as outfile:
            for entry in train_data: 
                json.dump(entry, outfile)
                outfile.write("\n")
        
        with open(f"{self.dataset_root}/{self.dataset_name}_test.json", "w") as outfile:
            for entry in test_data: 
                json.dump(entry, outfile)
                outfile.write("\n")   

class PrepareMCQDatasetUsmle4(PrepareMCQDataset):
    def __init__(self, data_path, dataset_root, dataset_name):
        super().__init__(data_path, dataset_root, dataset_name)
        
    def format_entry(self, example):
        return super().format_entry(example) 
    
    def create_dataset(self):
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
  
        with open("cot_medical_datasets/usmle_4.json", "w") as outfile:
            for entry in dataset: 
                formatted_question = self.format_entry(entry)
                answer = entry["answer"]
                
                qa = f"Question: {formatted_question}\nAnswer: {answer}"
                # rationale = gpt(PROMPT, qa)
                rationale = ""
                
                entry = {
                    "question": formatted_question, 
                    "answer": answer, 
                    "rationale": rationale
                }
                
                json.dump(entry, outfile)
                outfile.write("\n")
    
class PrepareMCQDatasetUsmleSelfAssesment(PrepareMCQDataset): 
    def __init__(self, data_path, dataset_root, dataset_name):
        super().__init__(data_path, dataset_root, dataset_name)
        
    def preprocess_data(self): 
        question_file_paths = ["C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step1.json", "C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step2.json", "C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step3.json"]
        combined_questions = []
        
        for path in question_file_paths: 
            with open(path, 'r') as file: 
                data = json.load(file)
                combined_questions.extend(data)
                
        with open("C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/questions.json", "w") as combined_file: 
            json.dump(combined_questions, combined_file, indent=4)
            
        answer_file_paths = ["C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step1_solutions.json", "C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step2_solutions.json", "C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/step3_solutions.json"]
        combined_answers = []
        
        for path in answer_file_paths: 
            with open(path, 'r') as file: 
                data = json.load(file)
                combined_answers.extend(data.values())
                
        with open("C:/Users/aryaa/OneDrive/Documents/Stanford/CS 224N/PLoHA/medalpaca_datasets/medical_meadow_usmle_self_assessment/answers.json", "w") as combined_file: 
            json.dump(combined_answers, combined_file, indent=4)
    
    
    def format_entry(self, example):
        return super().format_entry(example) 
    
    def create_dataset(self):  
        with open("medalpaca_datasets/medical_meadow_usmle_self_assessment/questions.json", "r") as file1, open("medalpaca_datasets/medical_meadow_usmle_self_assessment/answers.json", "r") as file2:
            questions = json.load(file1)
            answers = json.load(file2)
        
        with open("cot_medical_datasets/usmle_self_assessment.json", "w") as outfile: 
            for question, answer in zip(questions, answers):
                formatted_question = self.format_entry(question)
                if answer in question["options"]: 
                    formatted_answer = question["options"][answer]
                else: 
                    print(f"Q: {formatted_question} \n")
                    print(f"Key: {answer}")
                    break
                
                qa = f"Question: {formatted_question}\nAnswer: {formatted_answer}"
                # rationale = gpt(PROMPT, qa)
                rationale = ""
                
                entry = {
                    "question": formatted_question, 
                    "answer": formatted_answer, 
                    "rationale": rationale
                }
                
                json.dump(entry, outfile)
                outfile.write("\n")
            

if __name__ == "__main__": 
    dataset_root = "cot_medical_datasets"
    
    # data_path = ""
    
    # dataset_name = "usmle_4"
    # preparer = PrepareMCQDatasetUsmle4(data_path=data_path, dataset_root=dataset_root, dataset_name=dataset_name)
    # preparer.split_dataset()
    
    # dataset_name = "usmle_self_assessment"
    # preparer = PrepareMCQDataset(data_path=data_path, dataset_root=dataset_root, dataset_name=dataset_name)
    # preparer.split_dataset()
    
    path1 = "cot_medical_datasets/usmle_4_test.json"
    path2 = "cot_medical_datasets/usmle_self_assessment_test.json"
    path3 = "cot_medical_datasets/usmle_both_test.json"
    combine_datasets(path1, path2, path3)
    
    
    
    
  
  
