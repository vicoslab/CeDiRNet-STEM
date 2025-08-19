import numpy as np

import torch
import torch.nn as nn

from functools import partial

from models.center_groundtruth import CenterDirGroundtruth

from criterions.weightings.unbalanced_weight import UnbalancedWeighting
from criterions.weightings.instance_weight import InstanceGroupWeighting
from criterions.weightings.hard_instance_weight import HardInstanceGroupWeighting

from criterions.per_pixel_losses import get_per_pixel_loss_func

from criterions.center_localization_loss import CenterLocalizationLoss

class CenterDirectionLoss(nn.Module):

    def __init__(self, model, num_vector_fields=3,
                 no_instance_loss=True, cls_no_loss=False, enable_centerdir_loss=True, enable_cls_loss=True,
                 regression_loss=None, cls_loss=None, magnitude_regularization=None,
                 cls_l1_loss=False, cls_hinge_loss=False, regression_l1_loss=False, regression_hinge_loss=False, regression_hinge_loss_eps=0.0,
                 cls_instance_weighted=True, centerdir_instance_weighted=True, loss_weighted_by_distance_gauss=0,
                 use_instance_mask_iou_weight=False, instance_mask_iou_only_fp=False, instance_mask_iou_ignore_tp=False,
                 instance_mask_iou_include_fn=False,
                 num_hard_negatives=0, hard_negatives_center_mask=30, hard_negatives_as_single_instance=False,
                 extend_instance_mask_weights=False, extend_instance_mask_as_hard_negative=False,
                 use_log_r=True, use_log_r_base='exp',
                 learnable_center_est=False, learnable_center_loss='l1',
                 learnable_center_ignore_negative_gradient=False, learnable_center_fp_threshold=0.1,
                 learnable_center_with_instance_norm=True, learnable_center_positive_area_radius=1,
                 border_weight=1.0, border_weight_px=0, **kargs):
        super().__init__()

        assert num_vector_fields >= 3

        self.num_vector_fields = num_vector_fields

        ################################
        # enable main settings (cls loss, instance loss, localization loss)
        self.no_instance_loss = no_instance_loss                        # False == using only FG for instances, True == use FG and BG
        self.cls_no_loss = cls_no_loss                                  # no loss for classification

        self.enable_centerdir_loss = enable_centerdir_loss
        self.enable_cls_loss = enable_cls_loss

        if self.enable_cls_loss:
            self.cls_no_loss = cls_no_loss
        else:
            self.cls_no_loss = True

        self.localization_criteria = None
        if learnable_center_est:
            self.localization_criteria = CenterLocalizationLoss(loss_type=learnable_center_loss,
                                                                ignore_negative_gradient=learnable_center_ignore_negative_gradient,
                                                                fp_threshold=learnable_center_fp_threshold,
                                                                use_per_instance_normalization=learnable_center_with_instance_norm,
                                                                positive_area_radius=learnable_center_positive_area_radius)

        ################################
        # equalize importance of pixels from instances: pixels in instances should be considered equally
        # important as bg pixels regardless of thier size
        self.cls_instance_weighted = cls_instance_weighted              # normalize losses to per-instance case for classification
        self.centerdir_instance_weighted = centerdir_instance_weighted          # normalize losses to per-instance case for regressopm

        ################################
        # legacy support for loss type
        if regression_hinge_loss:
            # use hinge loss with absolute distance within desired eps
            #self.regression_loss_fn = partial(self.loss_hinge_fn, eps=self.regression_hinge_loss_eps, sign_fn=torch.abs)
            self.regression_loss = dict(type='hinge', args=dict(eps=regression_hinge_loss_eps, sign_fn=torch.abs))
        else:
            self.regression_loss = dict(type='L1' if regression_l1_loss else 'L2')

        if cls_hinge_loss:
            raise Exception('Hinge loss is not supported for classification any more')
        elif cls_l1_loss:
            self.cls_loss = dict(type='L2' if not cls_l1_loss else 'L1')

        ################################
        # main losses (cls and regression)
        # new way to define loss using dict(name='NAME',args=...)
        if regression_loss is not None:
            self.regression_loss = regression_loss

        if cls_loss is not None:
            self.cls_loss = cls_loss

        self.magnitude_regularization = magnitude_regularization        # add regularization by forcing sin^(x)+cos^(x) to zero

        self.use_log_r = use_log_r                                      # use logarithm of radii as groundtruth

        ################################################################
        # Prepare log and inverse log functions for R regression

        if use_log_r_base.lower() in ['exp', 'e']:
            self.log_r_fn = lambda x: torch.log(x + 1)
            self.inverse_log_r_fn = lambda x: torch.exp(x) - 1
        elif use_log_r_base.lower() in ['decimal', '10']:
            self.log_r_fn = lambda x: torch.log10(x + 1)
            self.inverse_log_r_fn = lambda x: torch.pow(10, x) - 1
        elif use_log_r_base.lower() in ['pow10']:
            self.log_r_fn =  lambda x: torch.log10(x + 1)
            self.inverse_log_r_fn = lambda x: torch.pow(x, 10) - 1
        else:
            raise Exception('Only "exp" and "10" are allowed logarithms for R')

        ################################################################
        # Prepare all loss functions
        self.regression_loss_fn = get_per_pixel_loss_func(self.regression_loss)
        self.cls_loss_fn = get_per_pixel_loss_func(self.cls_loss)

        self.magnitude_regularization_fn = None
        if self.magnitude_regularization is not None:
            assert self.magnitude_regularization.lower() in ['l1','l2'], \
                "invalid magnitude_regularization, only allowed [None, 'L1', 'L2']"

            self.magnitude_regularization_fn = get_per_pixel_loss_func(dict(type=self.magnitude_regularization, Y=1))

        ################################################################
        # Prepare classes for calculating weight masks
        def get_weighting_op(is_instance_weighted):
            if is_instance_weighted:
                if extend_instance_mask_weights or use_instance_mask_iou_weight:
                    return HardInstanceGroupWeighting(use_instance_mask_iou=use_instance_mask_iou_weight,
                                                      iou_only_fp=instance_mask_iou_only_fp,
                                                      iou_ignore_tp=instance_mask_iou_ignore_tp,
                                                      iou_include_fn=instance_mask_iou_include_fn,
                                                      num_hard_negatives=num_hard_negatives,
                                                      hard_negatives_center_mask=hard_negatives_center_mask,
                                                      hard_negatives_as_single_instance=hard_negatives_as_single_instance,
                                                      use_extend_instance_mask=extend_instance_mask_weights,
                                                      extend_instance_mask_as_hard_negative=extend_instance_mask_as_hard_negative,
                                                      border_weight=border_weight, border_weight_px=border_weight_px,
                                                      add_distance_gauss_weight=loss_weighted_by_distance_gauss)
                else:
                    return InstanceGroupWeighting(border_weight=border_weight, border_weight_px=border_weight_px,
                                                  add_distance_gauss_weight=loss_weighted_by_distance_gauss)
            else:
                return UnbalancedWeighting(border_weight=border_weight, border_weight_px=border_weight_px,
                                           add_distance_gauss_weight=loss_weighted_by_distance_gauss)

        self.cls_weighting = get_weighting_op(self.cls_instance_weighted)
        self.dir_weighting = get_weighting_op(self.centerdir_instance_weighted)

        if self.centerdir_instance_weighted == self.cls_instance_weighted and self.cls_no_loss == False:
            # Since cls and centerdir weights are the same we can re-use weights already computed.
            # To achieve this, we wrap self.cls_weighting with special function that will store result and return it when
            # self.dir_weighting is called.

            self.stored_weights = None
            def store_weights(fn, *args, **kwargs):
                self.stored_weights = fn(*args, **kwargs)
                return self.stored_weights

            self.cls_weighting = partial(store_weights,fn=self.cls_weighting)
            self.dir_weighting = lambda *args, **kwargs: self.stored_weights.clone() if self.stored_weights else self.cls_weighting(*args, **kwargs)

        self.tmp = nn.Conv2d(8,8,3)

    def forward(self, prediction, sample, centerdir_responses=None, centerdir_gt=None, ignore_mask=None, difficult_mask=None,
                w_r=1, w_cos=1, w_sin=1, w_magnitude=1, w_fg=1, w_bg=1, w_cls=1, w_cent=1, w_fg_cent=1, w_bg_cent=1,
                reduction_dims=(1,2,3), **kwargs):

        loss_output_shape = [d for i,d in enumerate(prediction.shape) if i not in reduction_dims]
        loss_zero_init = lambda: torch.zeros(size=loss_output_shape,device=prediction.device)

        loss_cls, loss_sin, loss_cos, loss_r = map(torch.clone,[loss_zero_init()]*4)  # centerdir_vectors and cls losses

        loss_centers = loss_zero_init()

        loss_magnitude_reg = loss_zero_init()

        instances = sample["instance"]
        instances = instances.squeeze(1)

        # batch computation ---
        labels = sample["label"]
        bg_mask = labels == 0
        fg_mask = bg_mask == False

        centerdir_vectors = prediction[:, 0:self.num_vector_fields]
        cls_mask = prediction[:, self.num_vector_fields:]  # 1 x h x w

        prediction_sin = centerdir_vectors[:, 0].unsqueeze(1)
        prediction_cos = centerdir_vectors[:, 1].unsqueeze(1)
        prediction_R = centerdir_vectors[:, 2].unsqueeze(1)

        predictions_other = centerdir_vectors[:, 3:]

        if instances.dtype != torch.int16:
            instances = instances.type(torch.int16)

        # mark ignore regions as -9999 in instances so that size can be correctly calculated in InstanceGroupWeighting
        if ignore_mask is not None:
            instances = instances.clone() # do not destroy original
            instances[ignore_mask.squeeze(dim=1) == 1] = InstanceGroupWeighting.IGNORE_FLAG

        # retrieve groundtruth values (either computed or from cache)
        gt_R, gt_theta, gt_sin_th, gt_cos_th,\
            gt_centers, gt_center_ignore, gt_center_mask, \
            gt_extended_instances = CenterDirGroundtruth.parse_groundtruth(centerdir_gt)

        if self.use_log_r:
            gt_R = self.log_r_fn(gt_R)

        if centerdir_responses is not None:
            centers_pred, center_heatmap = centerdir_responses
            centers_pred = self._unroll_center_predictions(centers_pred)
        else:
            centers_pred, center_heatmap = None, None

        # function that returns instance mask for each requested center (needed only by HardInstanceGroupWeighting)
        calc_mask_fn = None

        # prepare all arguments that are needed for calculating weighting mask
        weighting_args = dict(gt_instances=instances, gt_centers=gt_centers,
                              predicted_centers=centers_pred, calc_mask_fn=calc_mask_fn, gt_extended_instances=gt_extended_instances,
                              gt_ignore=ignore_mask, gt_difficult=difficult_mask, gt_R=gt_R,
                              w_fg=w_fg, w_bg=w_bg)

        ######################################################
        ### classification loss
        if self.cls_no_loss == False:
            with torch.no_grad():
                # create weighting mask for cls
                mask_weights = self.cls_weighting(**weighting_args)

            # regress class mask
            loss_cls += torch.sum(mask_weights * self.cls_loss_fn(cls_mask, labels.type(torch.float32)),dim=reduction_dims)

        ######################################################
        ### centerdir_vectors losses (R, cos, sin) + any regularization
        if self.enable_centerdir_loss:
            with torch.no_grad():
                mask_weights = self.dir_weighting(**weighting_args)

                # we need to ignore center parts since they are often wrong
                mask_weights *= gt_center_ignore.float()

            # add regression loss for sin(x), cos(x) and R
            if self.no_instance_loss:
                if w_sin != 0:
                    loss_sin += torch.sum(mask_weights * self.regression_loss_fn(prediction_sin, gt_sin_th), dim=reduction_dims)
                if w_cos != 0:
                    loss_cos += torch.sum(mask_weights * self.regression_loss_fn(prediction_cos, gt_cos_th), dim=reduction_dims)
                if w_r != 0:
                    loss_r += torch.sum(mask_weights * self.regression_loss_fn(prediction_R, gt_R), dim=reduction_dims)

            else:
                for b in range(mask_weights.shape[0]):
                    fg_mask_weights = mask_weights[b][fg_mask[b]]

                    if w_sin != 0:
                        loss_sin[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_sin[b][fg_mask[b]], gt_sin_th[b][fg_mask[b]]))
                    if w_cos != 0:
                        loss_cos[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_cos[b][fg_mask[b]], gt_cos_th[b][fg_mask[b]]))
                    if w_r != 0:
                        loss_r[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_R[b][fg_mask[b]], gt_R[b][fg_mask[b]]))

            if self.magnitude_regularization_fn is not None and w_magnitude != 0:
                prediction_M_square = torch.pow(prediction_sin, 2) + torch.pow(prediction_cos, 2)

                loss_magnitude_reg += torch.sum((~ignore_mask).float()*self.magnitude_regularization_fn(prediction_M_square[fg_mask]), dim=reduction_dims) / ((~ignore_mask).sum())

        ######################################################
        ###  localization loss for estimating center from centerdir_vectors outputs
        if self.localization_criteria is not None and w_cent != 0:
            loss_centers = self.localization_criteria(centers_pred, center_heatmap, gt_centers, gt_center_mask, ignore_mask,
                                                      w_fg=w_fg_cent, w_bg=w_bg_cent, reduction_dims=reduction_dims)

        loss_cls = w_cls * loss_cls

        loss_sin = w_sin * loss_sin
        loss_cos = w_cos * loss_cos
        loss_r = w_r * loss_r

        loss_centers = w_cent * loss_centers

        loss_magnitude_reg = w_magnitude * loss_magnitude_reg

        loss_centerdir_total = loss_sin + loss_cos + loss_r + loss_magnitude_reg

        # total/final loss:
        loss = loss_cls + loss_centerdir_total + loss_centers

        # add epsilon as a way to force values to tensor/cuda
        eps = prediction.sum() * 0
        # convert all losses to tensors to ensure proper parallelization with torch.nn.DataParallel
        losses = [t + eps for t in [loss, loss_cls, loss_centerdir_total, loss_centers,
                                    loss_sin, loss_cos, loss_r,
                                    loss_magnitude_reg]]

        return tuple(losses)

    def get_loss_dict(self, loss_tensor):
        # return dict(loss=loss_tensor[0].mean())

        loss, loss_cls, loss_centerdir_total, loss_centers, \
        loss_sin, loss_cos, loss_r, \
        loss_magnitude_reg = [l.sum() for l in loss_tensor]

        return dict(  # main loss for backprop:
            loss=loss,
            # losses for visualization:
            losses_groups=dict(cls=loss_cls, centerdir_total=loss_centerdir_total, centers=loss_centers),
            losses_centerdir_total=dict(sin=loss_sin, cos=loss_cos, r=loss_r, magnitude_reg=loss_magnitude_reg),
            losses_main=dict(cls=loss_cls, sin=loss_sin, cos=loss_cos, r=loss_r, cent=loss_centers))

    def _unroll_center_predictions(self, centers_pred):
        if type(centers_pred) == torch.Tensor:
            centers_pred_res = []
            for b, c in enumerate(centers_pred.cpu().numpy()):
                valid_center_idx = np.where(c[:, 0] != 0)[0].astype(np.int32)
                centers_pred_res.append(np.concatenate((np.ones((len(valid_center_idx), 1)) * b,
                                                        c[valid_center_idx, 1:]),
                                                       axis=1))
            centers_pred = np.concatenate(centers_pred_res, axis=0)
        return centers_pred