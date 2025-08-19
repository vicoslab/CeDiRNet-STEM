import torch
import torch.nn as nn


def positive_class_weight(batch):
    bg_pixels = torch.count_nonzero(batch == 0)
    fg_pixels = batch.numel() - bg_pixels

    return bg_pixels / fg_pixels


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, calculate_pos_weight=True, loss_name="BCE_loss", **kwargs):
        super().__init__()
        self.calculate_pos_weight = calculate_pos_weight
        self.loss_name = loss_name

    def forward(self, x, sample, **kwargs):
        target = sample["segmentation_mask"].float()
        losses = torch.zeros(x.shape[0]).to(x.device)

        for b in range(x.shape[0]):
            pos_weight = positive_class_weight(target[b]) if self.calculate_pos_weight else 1
            loss = nn.functional.binary_cross_entropy_with_logits(x[b], target[b], pos_weight=pos_weight)
            losses[b] = loss

        return (losses, losses)

    def get_loss_dict(self, loss_tensor):
        loss = loss_tensor[0]
        return {"loss": loss, self.loss_name: loss, "losses_tasks": {self.loss_name: loss.sum()}, "losses_groups": {self.loss_name: loss.sum()} }
