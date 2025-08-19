import torch

import numpy as np

from criterions.weightings.instance_weight import InstanceGroupWeighting

from models.center_groundtruth import CenterDirGroundtruth

from utils.evaluation.center_global_min import CenterGlobalMinimizationEval

class HardInstanceGroupWeighting(InstanceGroupWeighting):

    def __init__(self, use_extend_instance_mask=False, extend_instance_mask_as_hard_negative=False,
                 use_instance_mask_iou=False, iou_only_fp=False, iou_ignore_tp=False, iou_include_fn=False,
                 num_hard_negatives=0, hard_negatives_center_mask=30, hard_negatives_as_single_instance=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # at least use_extend_instance_mask or use_instance_mask_iou need to be enabled
        assert use_extend_instance_mask or use_instance_mask_iou

        ################################
        # groundtruth can be extended to give more importance to negative pixel around instance
        self.use_extend_instance_mask = use_extend_instance_mask  # extends instance mask that is used for normalization
        self.extend_instance_mask_as_hard_negative = extend_instance_mask_as_hard_negative

        ################################
        # consider each individual instance and NOT just all instances as a single group
        self.use_instance_mask_iou = use_instance_mask_iou    # give more weights to false prediction mask
        self.iou_only_fp = iou_only_fp                        # consider only false-positive masks when use_instance_mask_iou_weight=True
        self.iou_ignore_tp = iou_ignore_tp                    # ignore true-positive centers (efectively only doing hard-negatives on BG)
        self.iou_include_fn = iou_include_fn

        self.num_hard_negatives = num_hard_negatives                        # also include hard-negative detections on BG when use_instance_mask_iou_weight=True
        self.hard_negatives_center_mask = hard_negatives_center_mask        # for each hard negative center automatically include mask (30 -> 65x65 mask)
        self.hard_negatives_as_single_instance = hard_negatives_as_single_instance # consider hard negative as equal to one single instance

    def __call__(self, gt_instances, gt_centers, predicted_centers, calc_mask_fn,
                 gt_extended_instances=None, gt_ignore=None, gt_difficult=None, gt_R=None, w_fg=1, w_bg=1, *args, **kwargs):
        batch_size, height, width = gt_instances.shape

        pixel_group_mask = None

        if self.use_extend_instance_mask:
            pixel_group_mask = gt_extended_instances

            if self.extend_instance_mask_as_hard_negative:
                pixel_group_mask[(pixel_group_mask > 0) * (gt_instances == 0)] = -1

        # first find hard negatives by applying center estimation
        if self.use_instance_mask_iou:
            pixel_group_mask = gt_instances.clone().type(torch.int16) if pixel_group_mask is None else pixel_group_mask

            if self.num_hard_negatives > 0 or self.iou_include_fn:
                # convert gt_centers list to dictionary of gt_centers
                gt_centers_dict = CenterDirGroundtruth.convert_gt_centers_to_dictionary(gt_centers, instances=gt_instances,
                                                                                        ignore=gt_ignore)

                # find best matches of predictions to groundtruths
                prediction_gt_match = self._get_predictions_match_to_gt(predicted_centers, gt_centers_dict, gt_ignore,
                                                                        gt_difficult)

            for b in range(0, batch_size):
                # NOTE: select centers that are valid since the matrix has space of max 2000 centers
                gt_center_ids = torch.any(gt_centers[b] > 0, dim=1)
                gt_centers_b = gt_centers[b, gt_center_ids]

                if self.iou_ignore_tp:
                    centers = torch.zeros(size=(0, 2), device=gt_instances.device)
                else:
                    centers = gt_centers_b

                if self.num_hard_negatives > 0:
                    res = predicted_centers[predicted_centers[:, 0] == b, 1:]

                    # eliminate ones that are TP based on provided matched_pred
                    if prediction_gt_match is not None:
                        assert len(prediction_gt_match[b][1]) == len(prediction_gt_match[b][0])
                        false_idx = prediction_gt_match[b][1] == 0
                        if false_idx.sum() > 0:
                            # get original indexes that correspond to the res list
                            false_idx = prediction_gt_match[b][0].cpu().numpy()[false_idx]
                            if len(false_idx) > 0:
                                res = np.array(res[false_idx, :]).reshape(-1, 4)
                            else:
                                res = []
                        else:
                            res = []

                    else:
                        # else just eliminate ones that are close to groundtruth (less then 30 px)
                        res = [p for p in res if
                               (((np.array([p[1], p[0]]) - gt_centers_b.cpu().numpy()) ** 2).sum(axis=1) > 30).all()]

                    if len(res) > 0:
                        res = np.array(res, dtype=np.float)

                        # sort by highest score
                        res = res[np.argsort(res[:, 3])[::-1], :] if len(res) > 0 else res
                        # take only top N
                        res = res[:self.num_hard_negatives, :] if len(res) > self.num_hard_negatives else res
                        # retain only X,Y values and convert to pytorch tensor
                        res = torch.from_numpy(res[:, [1, 0]]).to(centers.device).float()

                        centers = torch.cat((centers, res), dim=0)

                        # also mark around centers of false positives even if they have no assigned mask
                        if self.hard_negatives_center_mask:
                            B = self.hard_negatives_center_mask  # 30 corresponds to half of the max kernel size of 65px used in conv2d for center search
                            for cy, cx in res:
                                cx, cy = int(np.round(cx.item())), int(np.round(cy.item()))
                                if gt_instances[b][cy, cx] > 0: continue  # skip detections/centers within actual instances
                                y0 = np.clip(cy - B, 0, pixel_group_mask.shape[-2])
                                y1 = np.clip(cy + B, 0, pixel_group_mask.shape[-2])
                                x0 = np.clip(cx - B, 0, pixel_group_mask.shape[-1])
                                x1 = np.clip(cx + B, 0, pixel_group_mask.shape[-1])

                                is_bg = pixel_group_mask[b][y0:y1, x0:x1] <= 0
                                pixel_group_mask[b][y0:y1, x0:x1][is_bg] = -1

                if len(centers) > 0:
                    centers = torch.cat((centers, torch.ones((len(centers), 1), device=centers.device)), dim=1)

                    predicted_mask = calc_mask_fn(b, centers)

                    # for id in predicted_mask.unique():
                    #     if id == 0: continue # ignore background
                    #     if id > len(gt_centers_b): continue # ignore false positive centers
                    #     hard_negative_mask[b][predicted_mask == id] = id

                    # mark pixels that contributed to wrong prediction as hard negatives
                    fp_mask = (gt_instances[b] > 0) * (predicted_mask > 0) * (
                            gt_instances[b] != predicted_mask.type(gt_instances.dtype))

                    if self.iou_only_fp:
                        fp_mask *= pixel_group_mask[b] <= 0

                    # mark hard negative pixels as -1 which can be considered as additional "instance"
                    pixel_group_mask[b][fp_mask] = -1

                if self.iou_include_fn:
                    for id in gt_instances[b].unique():
                        if id > 0 and id not in prediction_gt_match[b][-2]:
                            # mark as hard negative pixels that belong to missed object based on groundtruth instances
                            pixel_group_mask[b][gt_instances[b] == id] = -1

        assert pixel_group_mask is not None

        # consider hard negative as equal to one single instance
        if self.hard_negatives_as_single_instance:
            pixel_group_mask[pixel_group_mask <= 0] = pixel_group_mask.max() + 1

        # hard negative pixels should not be counted if in center of gt
        # pixel_group_mask *= gt_center_ignore.type(torch.int8)[:, 0]

        # remove pixels in ignore-list from pixel-group statistics by marking them with InstanceGroupWeighting.IGNORE_FLAG
        if gt_ignore is not None:
            pixel_group_mask[gt_ignore.squeeze(dim=1) == 1] = self.IGNORE_FLAG

        # recalculate the number of pixels per instances/bg/hard-neg
        pixel_group_instance_ids, pixel_group_instance_sizes = pixel_group_mask.reshape( pixel_group_mask.shape[0], -1).unique(return_counts=True, dim=-1)

        pixel_group_instance_sizes_rep = pixel_group_instance_sizes.repeat(batch_size, 1)
        num_bg_pixels = pixel_group_instance_sizes_rep[pixel_group_instance_ids == 0].sum().float()
        num_hard_negative_pixels = pixel_group_instance_sizes_rep[pixel_group_instance_ids == -1].sum().float()

        bg_mask = (gt_instances == 0).unsqueeze(1)
        fg_mask = bg_mask == False

        mask_weights = torch.ones_like(bg_mask, dtype=torch.float32, requires_grad=False, device=gt_instances.device)

        mask_weights[fg_mask] = w_fg
        mask_weights[bg_mask] = w_bg

        # apply additional weights around borders
        if self.border_weight_px > 0:
            mask_weights = self._apply_border_weights(mask_weights)

        if gt_ignore is not None:
            mask_weights *= 1 - gt_ignore.type(mask_weights.type())

        mask_weights = self._init_grouped_weights(mask_weights, pixel_group_mask, pixel_group_instance_ids,
                                                  pixel_group_instance_sizes, num_bg_pixels, num_hard_negative_pixels)

        # apply additional weight based on distance to center
        if self.add_distance_gauss_weight > 0:
            mask_weights = self._apply_gauss_distance_weights(mask_weights, gt_R)

        return mask_weights

    def _get_predictions_match_to_gt(self, center_pred, gt_centers_dict, ignore, difficult):
        """
        Matches all prediction points (center_pred) to groundtruths (gt_centers_dict) based on
        CenterGlobalMinimizationEval class.

        Locations marked with ignore or difficult class are ignored.
        """

        matched_pred = []
        for b in range(len(center_pred)):
            # calc FP and FN
            center_eval = CenterGlobalMinimizationEval()

            valid_pred_mask = center_pred[b, :, 0] != 0
            if ignore is not None:
                valid_pred_mask_ = valid_pred_mask.clone()
                valid_pred_mask[valid_pred_mask_] *= ignore[b, 0,
                                                            center_pred[b, valid_pred_mask_, 2].long(),
                                                            center_pred[b, valid_pred_mask_, 1].long()] == 0

            valid_pred = center_pred[b, valid_pred_mask, 1:3].cpu().numpy()

            _, _, pred_gt_match_i, pred_gt_match_idx_i = center_eval.add_image_prediction(None, None, None, valid_pred,
                                                                                          None, None,
                                                                                          gt_centers_dict[b],
                                                                                          difficult[b],
                                                                                          return_matched_gt_idx=True)

            gt_centers_keys = sorted(gt_centers_dict[b].keys())
            pred_gt_match_idx_i = [gt_centers_keys[int(i.item())] for i in pred_gt_match_idx_i if i >= 0]

            matched_pred.append((valid_pred_mask.nonzero(),
                                 pred_gt_match_i,
                                 pred_gt_match_idx_i,
                                 center_eval.metrics.copy()))

        return matched_pred