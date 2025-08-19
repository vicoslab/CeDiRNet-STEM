from criterions.center_direction_loss import CenterDirectionLoss
from criterions.shape_loss import ShapeLoss
from criterions.combined_loss import CombinedLoss
from criterions.segmentation_losses import BinaryCrossEntropyLoss

def get_criterion(type, loss_opts, model, center_model):

    if type in ["CombinedLoss"]:
        criterion = CombinedLoss(model, center_model, loss_opts, get_criterion)
    elif type in ['CenterDirectionLoss','PolarCenterLossV2']:
        criterion = CenterDirectionLoss(center_model, **loss_opts)
    elif type in ['CenterDirectionLossShape','ShapeLoss']:
        criterion = ShapeLoss(center_model, **loss_opts)
    elif type in ["BinaryCrossEntropyLoss"]:
        criterion = BinaryCrossEntropyLoss(**loss_opts)
    else:
        raise Exception("Unknown 'loss_type' in config: only allowed 'PolarCenterLoss', 'PolarCenterLossV2' or 'CenterDirectionLoss'")

    return criterion