import torch
from transformers import Seq2SeqTrainer
from torch.nn import functional as F

class PLoHATrainerWithSelfDistillation(Seq2SeqTrainer):
    def __init__(self, *args, distillation_temperature=2.0, distillation_alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs for labels and rationale (if exists)
        label_inputs = inputs['labels']
        rationale_inputs = inputs.get('rationales', None)

        # Forward pass to get logits
        outputs = model(**label_inputs)
        logits = outputs.logits

        # Generate soft targets by performing a forward pass without computing gradients
        with torch.no_grad():
            soft_targets = F.log_softmax(logits / self.distillation_temperature, dim=-1)

        # Compute original task loss (e.g., cross-entropy for labels)
        label_loss = F.cross_entropy(logits, label_inputs['labels'])

        # If rationale inputs are provided, compute rationale loss similarly
        if rationale_inputs is not None:
            rationale_loss = F.cross_entropy(logits, rationale_inputs['labels'])
            task_loss = (label_loss + rationale_loss) / 2  # Example way to combine losses
        else:
            task_loss = label_loss

        # Compute distillation loss
        distillation_loss = F.kl_div(F.log_softmax(logits / self.distillation_temperature, dim=-1), soft_targets, reduction='batchmean') * (self.distillation_temperature ** 2)

        # Combine task loss and distillation loss
        total_loss = self.distillation_alpha * task_loss + (1 - self.distillation_alpha) * distillation_loss

        return (total_loss, outputs) if return_outputs else total_loss
