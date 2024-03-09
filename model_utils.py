import pandas as pd
import torch
from torch import nn
from transformers import Seq2SeqTrainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_metric

class RationaleDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # Adaptation for handling both prediction and explanation features
        label_features = [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask'], 'labels': f['labels']} for f in features]
        rationale_features = [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask'], 'labels': f['rationale_labels']} for f in features if 'rationale_labels' in f]

        label_batch = super().__call__(label_features, return_tensors=return_tensors)
        rationale_batch = super().__call__(rationale_features, return_tensors=return_tensors)

        # Merge the batches
        return {
            'labels': label_batch,
            'rationales': rationale_batch,
        }

class PLoHATrainer(Seq2SeqTrainer):
    def __init__(self, *args, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # weight for balancing label loss and rationale loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # Split inputs for labels and rationales
        label_inputs = inputs['labels']
        rationale_inputs = inputs['rationales']

        # Compute loss for labels
        label_outputs = model(**label_inputs)
        label_loss = label_outputs.loss

        # Compute loss for rationales if available
        if rationale_inputs:
            rationale_outputs = model(**rationale_inputs)
            rationale_loss = rationale_outputs.loss
        else:
            rationale_loss = 0

        # Weighted sum of label loss and rationale loss
        total_loss = self.alpha * label_loss + (1 - self.alpha) * rationale_loss

        return (total_loss, label_outputs) if return_outputs else total_loss

# Example usage
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# data_collator = RationaleDataCollator(tokenizer=tokenizer, model=model)
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )
