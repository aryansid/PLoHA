import argparse
from transformers import AutoTokenizer

import data_utils

def run(args): 
  datasets = dataset_loader.load_from_json()

  if args.llm == 'gpt':
    train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
    test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')

  if args.llm is not None: 
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales) 
    
  if dataset_loader.has_valid:
    if args.llm == 'gpt': 
      valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
  else: 
    train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

    datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
    })
    
  if args.label_type == 'llm' and args.llm is not None: 
    train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
    test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
    
    print(f'LLM Train Acc: {train_label_acc:.4f}')
    print(f'LLM Test Acc: {test_label_acc:.4f}')

    datasets['train'] = datasets['train'].remove_columns('label')
    datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])
    
  if args.llm is not None:
    if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
    datasets = datasets.rename_column('llm_rationale', 'rationale')

  if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs

  elif args.model_type == 'standard':
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

  if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
  else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )


  if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

  else:
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)


  train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='medalpaca/medical_meadow_medical_flashcards')
    parser.add_argument('--model_type', type=str, default='standard')
    parser.add_argument('--label_type', type=str, default='llm')
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    run(args)
  