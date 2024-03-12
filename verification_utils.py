import json
import pandas as pd
import csv
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI()

PROMPT = """
You are a medical expert. You are given a medical question and its corresponding answer. Your task is to generate a rationale that explains how the answer stems from the question. YOU MUST BE CONCISE. Do not output anything aside from the rationale.
"""

def gpt(prompt, model="gpt-3.5-turbo-0125"):
    try: 
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
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
        
    def generate_verification_questions(self, rationale):
        # Placeholder function: generate verification questions based on the rationale
        # In practice, this might use another prompt with GPT or a heuristic approach
        verification_prompt = f"Generate verification questions for the following rationale: {rationale}"
        questions = gpt(verification_prompt)
        return questions.split('\n')  # Assuming each question is separated by a newline
    
    def verify_questions(self, questions):
        answers = []
        for question in questions:
            answer = gpt(question)
            answers.append(answer)
        return answers
    
    def post_process_dataset(self):
        dataset_path = f'{self.data_root}/{self.dataset_name}.txt'
        with open(dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        train_lines, test_lines = train_test_split(lines, test_size=0.2) 
        self._process_and_save_lines(train_lines, f'{self.data_root}/{self.dataset_name}_train.json')
        self._process_and_save_lines(test_lines, f'{self.data_root}/{self.dataset_name}_test.json')
    
    def _process_and_save_lines(self, lines, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in csv.reader(lines, delimiter=',', quotechar='"'):
                if len(line) == 3:
                    data = {
                        "question": line[0].strip(),
                        "answer": line[1].strip(),
                        "rationale": line[2].strip(),
                    }
                    # Generate verification questions and verify
                    verification_questions = self.generate_verification_questions(data["rationale"])
                    verification_answers = self.verify_questions(verification_questions)
                    # Incorporate verification results back into data
                    data["verification_questions"] = verification_questions
                    data["verification_answers"] = verification_answers
                    json.dump(data, file)
                    file.write('\n')

if __name__ == "__main__": 
    data_root = "cot_medical_datasets"
    dataset_name = "anki_flashcards"
    
    preparer = PrepareDataset(data_root=data_root, dataset_name=dataset_name)
    preparer.post_process_dataset()
