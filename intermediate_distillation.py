import torch
from transformers import Seq2SeqTrainer
from torch.nn import functional as F
from torch.nn import MSELoss

class PLoHATrainerWithIntermediateDistillation(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model, intermediate_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # ensure the teacher model is in evaluation mode
        self.intermediate_loss_weight = intermediate_loss_weight
        self.mse_loss = MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, output_hidden_states=True)
        student_hidden_states = outputs.hidden_states

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)
            teacher_hidden_states = teacher_outputs.hidden_states

        # Assuming the same number of layers in teacher and student, which might need adjustment
        intermediate_losses = []
        for student_layer, teacher_layer in zip(student_hidden_states, teacher_hidden_states):
            loss = self.mse_loss(student_layer, teacher_layer)
            intermediate_losses.append(loss)

        # Average loss across all intermediate layers
        intermediate_distillation_loss = sum(intermediate_losses) / len(intermediate_losses)

        # Compute task-specific loss
        logits = outputs.logits
        labels = inputs['labels']
        task_loss = F.cross_entropy(logits, labels)

        # Combine task loss with intermediate distillation loss
        total_loss = (1 - self.intermediate_loss_weight) * task_loss + self.intermediate_loss_weight * intermediate_distillation_loss

        return (total_loss, outputs) if return_outputs else total_loss
