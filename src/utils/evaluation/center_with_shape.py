import json
import os

import numpy as np
import torch
import cv2

from models.center_groundtruth import CenterDirGroundtruth

from utils.evaluation import NumpyEncoder
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.overlaps import overlap_pixels
from utils.utils import ShapeType

class CenterShapeEval(CenterGlobalMinimizationEval):

    def __init__(self, *args, use_gt_centers=False, append_error_to_display_name=True, shape_type=ShapeType.CIRCLE, **kwargs):
        super(CenterShapeEval, self).__init__(*args, **kwargs)

        self.use_gt_centers = use_gt_centers
        self.metrics.update(dict(size=[],size_per_img=[],iou=[],iou_per_img=[],translation=[],
                                 circularity=[],circularity_rel=[],
                                 diameter=[],diameter_rel=[],diameter_nm=[]))
        self.append_error_to_display_name = append_error_to_display_name

        self.shape_type = ShapeType[shape_type.upper()] if type(shape_type) is str else shape_type
        

    def save_str(self):
        return "tau=%.1f-score_thr=%.1f" % (self.tau_thr, self.score_thr)


    def add_image_prediction(self, im_name, im_index, im_shape, predictions_data, 
                             gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, return_matched_gt_idx=False,
                             **kwargs):

        # use parent class for center prediction matching
        ret = super(CenterShapeEval, self).add_image_prediction(
            im_name, im_index, im_shape, predictions_data,
            gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, return_matched_gt_idx=True)

        pred_centers = predictions_data['center']
        pred_coef = predictions_data.get('shape_coef')
        pred_ids = sorted(pred_centers.keys())

        predictions = np.array([pred_centers[id] for id in pred_ids]) if len(pred_ids) > 0 else []
        predictions_coef = np.array([pred_coef[id] for id in pred_ids]) if len(pred_ids) > 0 else []

        gt_missed, pred_missed, pred_gt_match_by_center, filename_suffix, pred_gt_match_by_center_idx = ret

        if filename_suffix is None:
            filename_suffix = ''

        gt_shape_coef = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_shape_coef'])
        gt_instances_poly = CenterDirGroundtruth.get_groundtruth_instance_polygon(centerdir_gt)
        px_in_nm = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['image_px_in_nm'])

        pred_polygon = {}

        if pred_gt_match_by_center_idx.shape[0] != 0:
            # indexes in pred_gt_match_by_center_idx are in order, but gt_centers_dict may have missing keys 
            # we need to get mapping from indexes in pred_gt_match_by_center_idx to gt_centers_dict keys
            gt_match_idx_to_gt_dict = {i:np.int16(k) for i,k in enumerate(sorted(gt_centers_dict.keys()))}
            
            gt_selected = np.array([gt_centers_dict[gt_match_idx_to_gt_dict[i]][::-1] for i in pred_gt_match_by_center_idx[:,0] if i >= 0])
            gt_selected_ids = np.array([gt_match_idx_to_gt_dict[i] for i in pred_gt_match_by_center_idx[:,0] if i >= 0])

            if len(gt_selected) > 0:
                assert len(predictions) == len(pred_gt_match_by_center)
                assert len(gt_selected) == sum(pred_gt_match_by_center[:,0] != 0)

                trans_err = np.abs(predictions[pred_gt_match_by_center[:,0] != 0, :2] - gt_selected)

                size_err = []
                diameter_err = []
                diameter_nm_err = []
                diameter_rel_err = []
                iou_err = []
                circularity_err = []
                circularity_rel_err = []

                if len(predictions_coef) > 0:
                    assert len(predictions_coef) == len(pred_gt_match_by_center)
                    predictions_coef = predictions_coef[pred_gt_match_by_center[:,0] != 0,:]

                pred_matched_idx = np.where(pred_gt_match_by_center[:,0] != 0)[0]

                for i, (gt_idx, p_idx) in enumerate(zip(gt_selected_ids,pred_matched_idx)):
                    if pred_gt_match_by_center[i] == 0:
                        continue
                    
                    c_gt = gt_centers_dict[gt_idx][::-1]

                    if self.shape_type in [ShapeType.CIRCLE,ShapeType.CIRCLE_WITH_CIRCULARITY]:
                        gt_radius_i = gt_shape_coef[0,0,int(c_gt[1]), int(c_gt[0])].cpu().numpy()
                        predictions_radius_i = predictions_coef[i][0]

                        e = np.abs(gt_radius_i - predictions_radius_i)
                        rel_e = e/gt_radius_i *100

                        size_err.append(e) # this is radius error 
                        diameter_err.append(2*e) # this is diameter error
                        diameter_rel_err.append(rel_e)

                        if px_in_nm is not None:
                            diameter_nm_err.append(2*e * px_in_nm.item()) # this is diameter error in nanometers
                    
                    if self.shape_type in [ShapeType.CIRCLE_WITH_CIRCULARITY]:
                        gt_circularity_i = gt_shape_coef[1,0,int(c_gt[1]), int(c_gt[0])].cpu().numpy()
                        predictions_circularity_i = predictions_coef[i][1]
                        #print("gt_circularity_i(inverse),predictions_circularity_i(inverse):", gt_circularity_i, predictions_circularity_i)

                        # convert to [0-1] range and get error there
                        gt_circularity_i = 1-1/gt_circularity_i
                        predictions_circularity_i = 1-1/predictions_circularity_i
                        #print("gt_circularity_i,predictions_circularity_i:", gt_circularity_i, predictions_circularity_i)
                        e = np.abs(predictions_circularity_i - gt_circularity_i)
                        rel_e = e/gt_circularity_i *100

                        circularity_err.append(e)                        
                        circularity_rel_err.append(rel_e)

                if len(trans_err) > 0:
                    self.metrics['translation'].extend(trans_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'te_{np.mean(trans_err):05.2f}_{filename_suffix}'

                if len(circularity_err) > 0:
                    self.metrics['circularity'].extend(circularity_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'circ_{np.mean(circularity_err):05.2f}_{filename_suffix}'

                if len(circularity_rel_err) > 0:
                    self.metrics['circularity_rel'].extend(circularity_rel_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'circ_{np.mean(circularity_rel_err):05.0f}%_{filename_suffix}'

                if len(size_err) > 0:
                    size_err = np.array(size_err)                    
                    size_per_img = size_err.mean()

                    self.metrics['size'].extend(size_err)
                    self.metrics['size_per_img'].append(size_per_img)

                    if self.append_error_to_display_name:
                        filename_suffix = f'size_{np.mean(size_per_img):05.2f}_{filename_suffix}'

                if len(diameter_err) > 0:
                    diameter_err = np.array(diameter_err)

                    self.metrics['diameter'].extend(diameter_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'diam_{np.mean(diameter_err):05.2f}px_{filename_suffix}'

                if len(diameter_nm_err) > 0:
                    diameter_nm_err = np.array(diameter_nm_err)

                    self.metrics['diameter_nm'].extend(diameter_nm_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'diam_{np.mean(diameter_nm_err):05.2f}nm_{filename_suffix}'

                if len(diameter_rel_err) > 0:
                    diameter_rel_err = np.array(diameter_rel_err)

                    self.metrics['diameter_rel'].extend(diameter_rel_err)

                    if self.append_error_to_display_name:
                        filename_suffix = f'diam_{np.mean(diameter_rel_err):05.2f}%_{filename_suffix}'


                if len(pred_polygon) > 0:
                    iou_err = np.array(iou_err) if len(iou_err) > 0 else np.array([0])
                    iou_per_img = iou_err.mean()

                    self.metrics['iou'].extend(iou_err)
                    self.metrics['iou_per_img'].append(iou_per_img)
                
                    if self.append_error_to_display_name:
                        filename_suffix = f'iou_{np.mean(iou_per_img):05.2f}_{filename_suffix}'

            else:
                print(f"No matching predictions found for {im_name}")
        
        if return_matched_gt_idx:
            return gt_missed, pred_missed, pred_gt_match_by_center, pred_polygon, filename_suffix, pred_gt_match_by_center_idx
        else:
            return gt_missed, pred_missed, pred_gt_match_by_center, pred_polygon, filename_suffix

    def calc_and_display_final_metrics(self, dataset, print_result=True, plot_result=True, save_dir=None, **kwargs):
        Re = np.array(self.metrics['Re']).mean()
        mae = np.array(self.metrics['mae']).mean()
        rmse = np.array(self.metrics['rmse']).mean()
        ratio = np.array(self.metrics['ratio']).mean()
        AP = np.array(self.metrics['precision']).mean()
        AR = np.array(self.metrics['recall']).mean()
        F1 = np.array(self.metrics['F1']).mean()
        TE = np.array(self.metrics['translation']).mean()
        SE = np.array(self.metrics['size']).mean()
        SE_PER_IMG = np.array(self.metrics['size_per_img']).mean()
        IoU = np.array(self.metrics['iou']).mean()
        IoU_PER_IMG = np.array(self.metrics['iou_per_img']).mean()
        CIRC = np.array(self.metrics['circularity']).mean()
        CIRC_REL = np.array(self.metrics['circularity_rel']).mean()
        DIAM = np.array(self.metrics['diameter']).mean()
        DIAM_REL = np.array(self.metrics['diameter_rel']).mean()
        DIAM_nm = np.array(self.metrics['diameter_nm']).mean()

        if print_result:
            RES = 'Re=%.4f, mae=%.4f, rmse=%.4f, ratio=%.4f, AP=%.4f, AR=%.4f, F1=%.4f, translation=%.4f, '% (Re, mae, rmse, ratio, AP, AR, F1, TE)
            RES += 'size=%.4f, size_per_img=%.4f, iou=%.4f, iou_per_img=%.4f, circularity=%.4f, circularity_rel=%.4f, diameter=%.4f, diameter_rel=%.4f, diameter_nm=%.4f, ' % (SE, SE_PER_IMG, IoU, IoU_PER_IMG, CIRC, CIRC_REL, DIAM, DIAM_REL, DIAM_nm)
            print(RES)

        if len(self.all_detections) > 0:
            all_detections = np.concatenate(self.all_detections,axis=0)

        if self.center_ap_eval is not None:
            metrics_mAP = self.center_ap_eval.calc_and_display_final_metrics(print_result, plot_result)
        else:
            metrics_mAP = None, None

        metrics = dict(AP=AP, AR=AR, F1=F1, ratio=ratio, Re=Re, mae=mae, rmse=rmse, all_images=self.metrics,
                       metrics_mAP=metrics_mAP, translation=TE, size=SE, size_per_img=SE_PER_IMG, iou=IoU, iou_per_img=IoU_PER_IMG, 
                       circularity=CIRC, circularity_rel=CIRC_REL, diameter=DIAM, diameter_rel=DIAM_REL, diameter_nm=DIAM_nm)

        ########################################################################################################
        # SAVE EVAL RESULTS TO JSON FILE
        if metrics is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, 'results.json'), 'w') as file:
                file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics