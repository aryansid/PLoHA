# import pandas as pd
# import torch
# from torch import nn
# from transformers import Seq2SeqTrainer, TrainingArguments, DataCollatorForSeq2Seq
# from datasets import load_metric

# class RationaleDataCollator(DataCollatorForSeq2Seq):
#     def __call__(self, features, return_tensors=None):
#         # Adaptation for handling both prediction and explanation features
#         label_features = [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask'], 'labels': f['labels']} for f in features]
#         rationale_features = [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask'], 'labels': f['rationale_labels']} for f in features if 'rationale_labels' in f]

#         label_batch = super().__call__(label_features, return_tensors=return_tensors)
#         rationale_batch = super().__call__(rationale_features, return_tensors=return_tensors)

#         # Merge the batches
#         return {
#             'labels': label_batch,
#             'rationales': rationale_batch,
#         }

# class PLoHATrainer(Seq2SeqTrainer):
#     def __init__(self, *args, alpha=0.5, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = alpha  # weight for balancing label loss and rationale loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Split inputs for labels and rationales
#         label_inputs = inputs['labels']
#         rationale_inputs = inputs['rationales']

#         # Compute loss for labels
#         label_outputs = model(**label_inputs)
#         label_loss = label_outputs.loss

#         # Compute loss for rationales if available
#         if rationale_inputs:
#             rationale_outputs = model(**rationale_inputs)
#             rationale_loss = rationale_outputs.loss
#         else:
#             rationale_loss = 0

#         # Weighted sum of label loss and rationale loss
#         total_loss = self.alpha * label_loss + (1 - self.alpha) * rationale_loss

#         return (total_loss, label_outputs) if return_outputs else total_loss

# # Example usage
# # training_args = TrainingArguments(
# #     output_dir="./results",
# #     learning_rate=2e-5,
# #     per_device_train_batch_size=8,
# #     num_train_epochs=3,
# #     weight_decay=0.01,
# # )

# # data_collator = RationaleDataCollator(tokenizer=tokenizer, model=model)
# # trainer = CustomTrainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_dataset,
# #     eval_dataset=eval_dataset,
# #     data_collator=data_collator,
# #     compute_metrics=compute_metrics,
# # )

# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer


"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs # placeholder only

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )