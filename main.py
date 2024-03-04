import argparse
from transformers import AutoTokenizer

import data_utils

def use_standard_distillation(): 
  def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

def run(args): 
  pass 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='medalpaca/medical_meadow_medical_flashcards')
    parser.add_argument('--model_type', type=str, default='standard')
    parser.add_argument('--label_type', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    run(args)
  