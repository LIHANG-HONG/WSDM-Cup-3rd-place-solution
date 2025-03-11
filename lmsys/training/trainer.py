
from trl import SFTConfig, SFTTrainer
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

class SFTChoiceTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, return_choice_logit=False):
        # アルファベット以降のim_endのtoken labelを-100に変更
        labels = inputs['labels']
        labels[labels > 57] = -100
        inputs['labels'] = labels
        _, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        logits = outputs.logits
        # シフト操作: ロジットを右にシフトしてラベルと整合
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_target = []
        labels_target = []
        for i in range(len(shift_labels)):
            lbl = shift_labels[i].cpu().numpy()
            target_idx = np.where(lbl != -100)[0][0]
            # A-Yのラベルのみ取得
            logits_target.append(shift_logits[i][target_idx][32:34])
            # labelを0-25に変換
            labels_target.append(shift_labels[i][target_idx]-32)

        # (batch_size, 26)
        logits_target = torch.stack(logits_target, dim=0)
        # (batch_size)
        labels_target = torch.tensor(labels_target).to(outputs.logits.device)
        loss = F.cross_entropy(logits_target, labels_target)
        if return_choice_logit:
            return (loss, outputs, logits_target) if return_outputs else (loss, logits_target)
        else:
            return (loss, outputs) if return_outputs else loss


class SFTDistillTrainer(SFTChoiceTrainer):
    def post_init(self,exp_config, idx_to_id):
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.names = []
        self.teacher_logits_dict = {}
        distil_params = exp_config.llm_config.distil_params
        distil_weights = []
        for name, path, path_tta, temperature, distill_weight in distil_params:
            print('================================================================================')
            print(f"Loading teacher: {name}")
            print(f"Loading teacher logits from {path}")
            print(f"Loading teacher logits tta from {path_tta}")
            print(f"Distilling {name} with temperature {temperature} and weight {distill_weight}")
            print('================================================================================')
            teacher_logits = torch.load(path)
            teacher_logits_tta = torch.load(path_tta)
            self.set_distill_settings(teacher_logits, teacher_logits_tta, name, temperature, distill_weight)
            distil_weights.append(distill_weight)
        self.hard_weight = 1.0 - sum(distil_weights)
        print('================================================================================')
        print(f"Hard loss weight: {self.hard_weight}")
        print('================================================================================')
        self.idx_to_id = idx_to_id

    def set_distill_settings(self, teacher_logits, teacher_logits_tta, name, temperature=5.0, distill_weight = 0.6):
        self.teacher_logits_dict[name] = (teacher_logits, teacher_logits_tta, temperature, distill_weight)
        self.names.append(name)

    def calc_soft_loss(self, inputs, outputs, name, logits_target):
        unique_ids = inputs["unique_id"].cpu()
        shuffles = inputs['shuffle'].cpu()
        ids = [self.idx_to_id[unique_id.item()] for unique_id in unique_ids]
        teacher_logits_all, teacher_logits_tta_all, temperature, distill_weight = self.teacher_logits_dict[name]
        teacher_logits = []
        for id_str, shuffle in zip(ids, shuffles):
            if not shuffle:
                teacher_logit = teacher_logits_all[id_str]
            else:
                teacher_logit = teacher_logits_tta_all[id_str]
            teacher_logits.append(teacher_logit)
        teacher_logits = torch.stack(teacher_logits, dim=0)
        teacher_logits = teacher_logits.to(outputs.logits.device)

        soft_loss = self.distill_loss(F.log_softmax(logits_target / temperature, dim=-1), F.softmax(teacher_logits / temperature, dim=-1)) * (temperature ** 2)
        return soft_loss, distill_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # アルファベット以降のim_endのtoken labelを-100に変更
        labels = inputs['labels']
        labels[labels > 57] = -100
        inputs['labels'] = labels
        hard_loss, outputs, logits_target = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, return_choice_logit=True)
        loss = 0
        for name in self.names:
            soft_loss, distill_weight = self.calc_soft_loss(inputs, outputs, name, logits_target)
            loss = loss + soft_loss * distill_weight
        loss = loss + hard_loss * self.hard_weight


        return (loss, outputs) if return_outputs else loss
