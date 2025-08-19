import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from models.center_groundtruth import CenterDirGroundtruth

from criterions.weightings.instance_weight import InstanceGroupWeighting
from criterions.center_direction_loss import CenterDirectionLoss

from criterions.per_pixel_losses import get_per_pixel_loss_func

from utils.utils import get_log_and_inverse_fn, ShapeType

class ShapeLoss(nn.Module):
    def __init__(self, model, shape_args=dict(), **center_dir_args):
        super().__init__()

        self.enable_shape_loss = shape_args.get('enable')
        self.shape_type = shape_args.get('shape_type') if 'shape_type' in shape_args else ShapeType.CIRCLE
        self.no_instance_loss = shape_args.get('no_instance_loss')
        self.loss_weighting = shape_args.get('loss_weighting')

        self.individual_uw = shape_args.get('individual_uw')

        self.shape_type = ShapeType[self.shape_type.upper()] if type(self.shape_type) is str else self.shape_type

        self.use_log = shape_args.get('use_log') if 'use_log' in shape_args else True
        self.use_log_base = shape_args.get('use_log_base') if 'use_log_base' in shape_args else 'exp'
        self.use_log_channels = shape_args.get('use_log_channels')
        
        self.w_shape_lambda = shape_args.get('w_shape_lambda')

        self.gt_keyname = shape_args.get('gt_keyname') if 'gt_keyname' in shape_args else 'gt_shape_coef'

        ################################################################
        # Prepare log and inverse log functions
        if self.use_log:
            self.log_fn, self.inverse_log_fn = get_log_and_inverse_fn(self.use_log_base)

        self.centerdir_loss_op = CenterDirectionLoss(model, **center_dir_args)

        # keep log_r for compatabilty reason !!
        self.use_log_r = self.centerdir_loss_op.use_log_r
        self.log_r_fn = self.centerdir_loss_op.log_r_fn
        self.num_vector_fields = self.centerdir_loss_op.num_vector_fields
        
        if self.shape_type == ShapeType.CIRCLE:
            self.NUM_SHAPE_FIELDS = 1
        elif self.shape_type == ShapeType.CIRCLE_WITH_CIRCULARITY:
            self.NUM_SHAPE_FIELDS = 2
        elif self.shape_type == ShapeType.ELIPSE:
            self.NUM_SHAPE_FIELDS = 3

        REQUIRED_VECTOR_FIELDS = 3 + self.NUM_SHAPE_FIELDS
        
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

        self.regression_loss_fn = get_per_pixel_loss_func(shape_args.get('regression_loss'))

        self.shape_weighting = InstanceGroupWeighting(border_weight=center_dir_args.get('border_weight',1.0),
                                                       border_weight_px=center_dir_args.get('border_weight_px',0),
                                                       add_distance_gauss_weight=False)
        self.tmp = nn.Conv2d(8,8,3)

    def forward(self, prediction, sample, centerdir_responses=None, centerdir_gt=None, ignore_mask=None,
                difficult_mask=None, w_shape=1, w_fg_shape=1, w_bg_shape=1, w_radius=1, w_circularity=1, reduction_dims=(1, 2, 3), epoch_percent=None, **kwargs):

        loss_output_shape = [d for i, d in enumerate(prediction.shape) if i not in reduction_dims]
        loss_zero_init = lambda: torch.zeros(size=loss_output_shape, device=prediction.device)
        
        instances = sample["instance"]
        instances = instances.squeeze(1)

        # batch computation ---
        labels = sample["label"]
        bg_mask = labels == 0
        fg_mask = bg_mask == False

        centerdir_vectors = prediction[:, 0:self.num_vector_fields]

        INPUT_OFFSET = 3
        
        prediction_shape_coef = centerdir_vectors[:, INPUT_OFFSET:INPUT_OFFSET+self.NUM_SHAPE_FIELDS].unsqueeze(2)
        
        if instances.dtype != torch.int16:
            instances = instances.type(torch.int16)

        # mark ignore regions as -9999 in instances so that size can be correctly calculated in InstanceGroupWeighting
        if ignore_mask is not None:
            instances = instances.clone()  # do not destroy original
            instances[ignore_mask.squeeze(dim=1) == 1] = InstanceGroupWeighting.IGNORE_FLAG

        # retrieve groundtruth values (either computed or from cache)
        gt_shape_coef = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=[self.gt_keyname])
        
        assert prediction_shape_coef.shape[1] == gt_shape_coef.shape[1]

        # apply log if needed - but only to specific channels
        if self.use_log:
            # apply to copy
            gt_shape_coef = gt_shape_coef.clone()

            use_log_channels = self.use_log_channels if self.use_log_channels else list(range(gt_shape_coef.shape[1]))
            
            for i in range(gt_shape_coef.shape[1]):
                if i in use_log_channels:
                    gt_shape_coef[:,i] = self.log_fn(gt_shape_coef[:,i])

        # prepare all arguments that are needed for calculating weighting mask
        weighting_args = dict(gt_instances=instances, gt_ignore=ignore_mask, gt_difficult=difficult_mask,
                              w_fg=w_fg_shape, w_bg=w_bg_shape)

        ######################################################
        ### shape loss
        loss_shape = []

        if self.w_shape_lambda is not None:
            w_shape = w_shape * self.w_shape_lambda(epoch_percent)

        if self.enable_shape_loss:

            with torch.no_grad():
                mask_weights = self.shape_weighting(**weighting_args)

                # we need to ignore center parts since they are often wrong
                # mask_weights *= gt_center_ignore.float()

            # add regression loss for different types of shape
            if self.shape_type == ShapeType.CIRCLE:
                loss_radius, = map(torch.clone, [loss_zero_init()] * 1)
                
                if self.no_instance_loss:

                    loss_radius += torch.sum(mask_weights * self.regression_loss_fn(prediction_shape_coef, gt_shape_coef), dim=reduction_dims)
                else:
                    for b in range(mask_weights.shape[0]):
                        fg_mask_weights = mask_weights[b][fg_mask[b]]
                        loss_radius[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_shape_coef[b][:,fg_mask[b]],gt_shape_coef[b][:,fg_mask[b]]))

                loss_radius = torch.stack([w_shape * w_radius * l for l in loss_radius])

                loss_shape = [loss_radius]

            elif self.shape_type == ShapeType.CIRCLE_WITH_CIRCULARITY:
                loss_radius, loss_circularity = map(torch.clone, [loss_zero_init()] * 2)

                prediction_radius, prediciton_circularity = prediction_shape_coef[:,0:1], prediction_shape_coef[:,1:2]
                gt_radius, gt_circulairty = gt_shape_coef[:,0:1], gt_shape_coef[:,1:2]

                # add regression loss 
                if self.no_instance_loss:

                    loss_radius += torch.sum(mask_weights * self.regression_loss_fn(prediction_radius, gt_radius), dim=reduction_dims)
                    loss_circularity += torch.sum(mask_weights * self.regression_loss_fn(prediciton_circularity, gt_circulairty), dim=reduction_dims)
                else:
                    for b in range(mask_weights.shape[0]):
                        fg_mask_weights = mask_weights[b][fg_mask[b]]

                        loss_radius[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_radius[b][:,fg_mask[b]],gt_radius[b][:,fg_mask[b]]))
                        loss_circularity[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediciton_circularity[b][:,fg_mask[b]],gt_circulairty[b][:,fg_mask[b]]))

                loss_radius = torch.stack([w_shape * w_radius*l for l in loss_radius])
                loss_circularity = torch.stack([ w_shape * w_circularity*l for l in loss_circularity])

                loss_shape = [loss_radius, loss_circularity]

            else:
                raise Exception("Not implemented")

        loss_shape_total = torch.stack(loss_shape).sum(dim=0)
        loss_shape_total += prediction.sum() * 0

        # call base loss function for center direction
        all_centerdir_losses = self.centerdir_loss_op.forward(prediction, sample, centerdir_responses,
                                                              centerdir_gt, ignore_mask, difficult_mask,
                                                              reduction_dims=reduction_dims, **kwargs)

        losses_main = [all_centerdir_losses[0] + loss_shape_total]

        return tuple(losses_main + list(all_centerdir_losses[1:]) + [loss_shape_total] + list(loss_shape))


    def get_loss_dict(self, loss_tensor):
        
        loss, loss_cls, loss_centerdir_total, loss_centers, loss_sin, \
        loss_cos, loss_r, loss_magnitude_reg, loss_shape_total = [l.sum() for l in loss_tensor[:9]]

        loss_shape = loss_tensor[9:]
        
        if self.individual_uw:
            if self.shape_type == ShapeType.CIRCLE:
                loss_shape_tasks = dict(loss_radius=loss_shape[0].sum())
            elif self.shape_type == ShapeType.CIRCLE_WITH_CIRCULARITY:
                loss_shape_tasks = dict(loss_radius=loss_shape[0].sum(),
                                        loss_circularity=loss_shape[1].sum())
            else:
                raise Exception("Not implemneted")
        else:
            loss_shape_tasks = dict(loss_shape=torch.stack(loss_shape).sum())

        losses_tasks = dict(centerdir=loss_centerdir_total, **loss_shape_tasks)
        
        return dict(  # main loss for backprop:
            loss=loss,
            # losses for visualization:
            losses_groups=dict(cls=loss_cls, centerdir_total=loss_centerdir_total, centers=loss_centers, shape_total=loss_shape_total),
            losses_centerdir_total=dict(sin=loss_sin, cos=loss_cos, r=loss_r, magnitude_reg=loss_magnitude_reg),
            losses_main=dict(cls=loss_cls, sin=loss_sin, cos=loss_cos, r=loss_r, cent=loss_centers, ),
            # losses for task weighting:
            losses_tasks=losses_tasks
        )
