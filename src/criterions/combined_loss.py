import torch
import numpy as np
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, model, center_model, losses_list, get_criterion):
        super().__init__()
        self.criterions = []
        start_channel = 0

        for loss_args in losses_list:
            loss_type = loss_args["type"]
            loss_kwargs = loss_args["kwargs"]
            loss_channels = loss_args["channels"]

            criterion = get_criterion(loss_type, loss_kwargs, model, center_model)

            end_channel = start_channel + loss_channels
            self.criterions.append({"channels": (start_channel, end_channel), "criterion": criterion})
            start_channel = end_channel

        self.losses_per_criterion = np.zeros(len(self.criterions), dtype=int)

    def forward(self, output, sample, **kwargs):
        all_losses = [torch.zeros(output.shape[0]).to(output.device)]

        for i, criterion in enumerate(self.criterions):
            start, end = criterion["channels"]

            losses = criterion["criterion"](output[:, start:end], sample, **kwargs)

            all_losses[0] += losses[0]
            all_losses += losses[1:]
            self.losses_per_criterion[i] = len(losses) - 1

        return all_losses

    def get_loss_dict(self, loss_tensor):
        start = 1

        loss_tensor = [l.sum() for l in loss_tensor]

        final_dict = dict(loss=loss_tensor[0])

        for i, criterion in enumerate(self.criterions):
            criterion = criterion["criterion"]

            end = start + self.losses_per_criterion[i]
            losses = [loss_tensor[0]] + loss_tensor[start:end]
            start = end

            loss_dict = criterion.get_loss_dict(losses)
            for k, v in loss_dict.items():
                if k in final_dict and isinstance(final_dict[k], dict):
                    final_dict[k].update(v)
                else:
                    final_dict.update({k: v})

        return final_dict
