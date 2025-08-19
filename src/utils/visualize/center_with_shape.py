import os
from functools import partial

import numpy as np
import torch

import cv2

from utils.visualize.vis import Visualizer

from utils.utils import get_log_and_inverse_fn, ShapeType

class CentersShapeVisualizeTest(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'centers-est', 'pred',  'class', 'gt-diff', 'label']

    def __init__(self, keys=(), plot_only=None, shape_type=ShapeType.CIRCLE, polygon=False, image_channels=None, image_grayscale=None, **kwargs):
        super(CentersShapeVisualizeTest, self).__init__(keys=self.KEYS + list(keys), **kwargs)

        self.image_channels = image_channels
        self.image_grayscale = image_grayscale
        self.shape_type = ShapeType[shape_type.upper()] if type(shape_type) is str else shape_type

        self.plot_only = plot_only if plot_only is not None else self.KEYS + list(keys)
        self.polygon = polygon

        self.use_log = kwargs.get('use_log') if 'use_log' in kwargs else True
        self.use_log_base = kwargs.get('use_log_base') if 'use_log_base' in kwargs else 'exp'

        ################################################################
        # Prepare log and inverse log functions
        if self.use_log:
            self.log_fn, self.inverse_log_fn =  get_log_and_inverse_fn(self.use_log_base)
        else:
            self.log_fn = lambda x: x
            self.inverse_log_fn = lambda x: x

    def parse_results(self, sample, result, difficult):
        im = sample['image'][0]
        if self.image_channels is not None:
            im = im[self.image_channels]
        if self.image_grayscale:
            im = (im * 255).clamp(0, 255).to(torch.uint8)
            im = im.repeat(3, 1, 1)

        output = result['output']
        center_est_imshow = result['pred_heatmap']
        centerdir_gt = sample.get('centerdir_groundtruth')
        gt_centers_dict = sample['center_dict']

        gt_list = np.array([gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())])

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        return im, output,  center_est_imshow, centerdir_gt, gt_list, is_difficult_gt

    def plot_radius_predictions(self, ax_, pred_list, radius_prediction_map, **kwargs):
        import matplotlib.patches as patches

        for pt in pred_list:
            radius = radius_prediction_map[int(pt[0]),int(pt[1])]
            radius = self.inverse_log_fn(radius)

            circle = patches.Circle([pt[1],pt[0]], radius, **kwargs)
            
            # Add the circle to the axes
            ax_.add_patch(circle)

    def visualize_pylab(self, sample, result, pred_data, pred_gt_match,
                  difficult, base, save_dir=None, plot_bbox_all=False):

        pred_center = pred_data['center']
        pred_list = np.array([pred_center[id] for id in sorted(pred_center.keys())]) if len(pred_center) > 0 else []

        im, output, center_est_imshow, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        from models.center_groundtruth import CenterDirGroundtruth
        gt_R, gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_R','gt_sin_th','gt_cos_th'])

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        # function pointers
        plot_predictions = partial(self.plot_predictions, pred_list=pred_list, pred_match=pred_gt_match)
        plot_radius_predictions = partial(self.plot_radius_predictions, pred_list=pred_list, radius_prediction_map=output[0, 3])

        if 'image' in self.plot_only:
            fig_img, ax = self.display(im.cpu(), 'image', force_draw=False)
            plot_predictions(ax, markersize=10, markeredgewidth=2)

            if len(gt_list[is_difficult_gt == 0]) > 0:
                ax.plot(gt_list[is_difficult_gt == 0, 1], gt_list[is_difficult_gt == 0, 0], 'g.',
                        markersize=5, markeredgewidth=0.2, markerfacecolor=(0, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
            if len(gt_list[is_difficult_gt != 0]) > 0:
                ax.plot(gt_list[is_difficult_gt != 0, 1], gt_list[is_difficult_gt != 0, 0], 'y.',
                        markersize=5, markeredgewidth=0.2, markerfacecolor=(1, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
            
            plot_radius_predictions(ax, linewidth=2, edgecolor='blue', facecolor='none')
               
        if output is not None and 'centers' in self.plot_only:
            fig_centers, ax = self.display([((output[0, 2])).detach().cpu(),
                                            (output[0, 1]).detach().cpu(),
                                            (output[0, 0]).detach().cpu()], 'centers', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)

        if center_est_imshow is not None and 'centers-est' in self.plot_only:
            fig_centers_conv, ax = self.display(
                [f.detach().cpu() if type(f) == torch.Tensor else f for f in center_est_imshow],
                'centers-est', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)
        
        if output is not None and 'class' in self.plot_only:
            cls = output[0][3].cpu()
            fig_class, ax = self.display(cls, 'class', force_draw=False)
        if centerdir_gt is not None and output is not None and 'gt-diff' in self.plot_only:
            fig_centerdir, ax = self.display([torch.abs(output[0, 2].detach().cpu() - gt_R.cpu()),
                                              torch.abs(output[0, 1].detach().cpu() - gt_sin_th.cpu()),
                                              torch.abs(output[0, 0].detach().cpu() - gt_cos_th.cpu())],
                                             'gt-diff', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)

        if save_dir is not None:
            fig_img.savefig(os.path.join(save_dir, '%s_0.img.png' % base))
            if 'fig_centers' in locals():
                fig_centers.savefig(os.path.join(save_dir, '%s_1.centers.png' % base))
            if 'fig_centers_conv' in locals():
                fig_centers_conv.savefig(os.path.join(save_dir, '%s_2.centers_conv.png' % base))
            if 'fig_class' in locals():
                fig_class.savefig(os.path.join(save_dir, '%s_3.class.png' % base))
            if 'fig_centerdir' in locals():
                fig_centerdir.savefig(os.path.join(save_dir, '%s_1.gt-diff.png' % base))

    def plot_radius_predictions_cv(self, img, pred_list, radius_prediction_map, pred_match, gt_list, gt_radius_map, is_difficult_gt,
                                   gt=False, predictions_args=dict(), radius_args=dict(), radius_gt_args=dict(), **kwargs):
        predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        predictions_args_.update(predictions_args)

        radius_args_ = dict(thickness=2)
        radius_args_.update(radius_args)
        radius_gt_args_ = dict(thickness=1)
        radius_gt_args_.update(radius_gt_args)
        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]

            for p in pred_list_true:                 
                cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 255, 0), **predictions_args_)
                
                radius = radius_prediction_map[(int(p[1]), int(p[0]))]
                radius = self.inverse_log_fn(radius)

                cv2.circle(img, (int(p[0]), int(p[1])), int(radius), color=(0, 255, 0), **radius_args_)

            for p in pred_list_false: 
                cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 255), **predictions_args_)

                radius = radius_prediction_map[(int(p[1]), int(p[0]))]
                radius = self.inverse_log_fn(radius)

                cv2.circle(img, (int(p[0]), int(p[1])), int(radius), color=(0, 0, 255), **radius_args_)

        if gt:
            for i, p in enumerate(gt_list):
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4,
                           color=(255, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4, color=(0, 0, 0), thickness=1)

                radius = gt_radius_map[(int(p[0]), int(p[1]))]

                cv2.circle(img, (int(p[1]), int(p[0])), int(radius), color=(255, 0, 0), **radius_gt_args_)


        return img



    def visualize_opencv(self, sample, result, pred_data, pred_gt_match,
                         difficult, base, save_dir, plot_bbox_all=False):

        pred_center = pred_data['center']
        pred_poly_mask = pred_data.get('polygon')
        pred_list = np.array([pred_center[id] for id in sorted(pred_center.keys())]) if len(pred_center) > 0 else []

        im, output, center_est_imshow, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        from models.center_groundtruth import CenterDirGroundtruth
        gt_R, gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_R','gt_sin_th','gt_cos_th'])
        gt_shape_coef = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_shape_coef'])[:,0]
    
        # function pointer
        plot_predictions = partial(self.plot_predictions_cv,
                                   pred_list=pred_list, pred_poly_mask=pred_poly_mask, pred_match=pred_gt_match,
                                   gt_list=gt_list, is_difficult_gt=is_difficult_gt)
        if self.shape_type in [ShapeType.CIRCLE, ShapeType.CIRCLE_WITH_CIRCULARITY]:
            plot_predictions_with_radius = partial(self.plot_radius_predictions_cv,
                                                    pred_list=pred_list, pred_match=pred_gt_match, radius_prediction_map=output[0, 3],
                                                    gt_list=gt_list, gt_radius_map=gt_shape_coef[0], is_difficult_gt=is_difficult_gt)
        else:
            plot_predictions_with_radius = plot_predictions

        if 'image' in self.plot_only:
            fig_img = self.display_opencv(im.cpu(), 'image', plot_fn=partial(plot_predictions_with_radius, gt=True, bbox=True))

        if 'label' in self.plot_only:
            fig_instances = self.display_opencv(sample['label'].cpu().unsqueeze(0).int(), 'label', plot_fn=partial(plot_predictions_with_radius, gt=True, bbox=True))


        if output is not None and 'centers' in self.plot_only:
            fig_centers = self.display_opencv([((output[0, 2])).detach().cpu(),
                                            (output[0, 1]).detach().cpu(),
                                            (output[0, 0]).detach().cpu(),
                                            torch.atan2(output[0, 0], output[0, 1]).detach().cpu()], 'centers',
                                            plot_fn=partial(plot_predictions, bbox=plot_bbox_all),
                                            image_colormap=[cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA,
                                                            cv2.COLORMAP_HSV])
        if center_est_imshow is not None and 'centers-est' in self.plot_only:
            fig_centers_conv = self.display_opencv(
                [f.detach().cpu() if type(f) == torch.Tensor else f for f in center_est_imshow],
                'centers-est',
                plot_fn=partial(plot_predictions, bbox=plot_bbox_all, predictions_args=dict(thickness=1)))

        if output is not None and 'class' in self.plot_only:
            cls = output[0][4].cpu()
            fig_cls = self.display_opencv(cls, 'class', plot_fn=partial(plot_predictions, bbox=plot_bbox_all))

        if centerdir_gt is not None and output is not None and 'gt-diff' in self.plot_only:
            fig_centerdir = self.display_opencv([torch.abs(output[0, 2].detach().cpu() - gt_R.cpu()),
                                                 torch.abs(output[0, 1].detach().cpu() - gt_sin_th.cpu()),
                                                 torch.abs(output[0, 0].detach().cpu() - gt_cos_th.cpu())],
                                                'gt-diff', plot_fn=partial(plot_predictions, bbox=plot_bbox_all))
        if save_dir is not None:
            if 'fig_img' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_0.img.png' % base), fig_img)
            if 'fig_instances' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_0.instnaces.png' % base), fig_instances)
            if 'fig_centers' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.centers.png' % base), fig_centers)
            if 'fig_centers_conv' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_2.centers_conv.png' % base), fig_centers_conv)
            if 'fig_cls' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_3.class.png' % base), fig_cls)
            if 'fig_centerdir' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.gt-diff.png' % base), fig_centerdir)

class CentersShapeVisualizeTrain(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'pred', 'class', 'centerdir_gt', 'conv_centers']

    def __init__(self, keys=(), shape_type=ShapeType.CIRCLE, **kwargs):
        super(CentersShapeVisualizeTrain, self).__init__(keys=self.KEYS + list(keys), **kwargs)

        self.shape_type = ShapeType[shape_type.upper()] if type(shape_type) is str else shape_type

        self.use_log = kwargs.get('use_log') if 'use_log' in kwargs else True
        self.use_log_base = kwargs.get('use_log_base') if 'use_log_base' in kwargs else 'exp'
        
        ################################################################
        # Prepare log and inverse log functions

        if self.use_log:
            self.log_fn, self.inverse_log_fn =  get_log_and_inverse_fn(self.use_log_base)
        else:
            self.log_fn = lambda x: x
            self.inverse_log_fn = lambda x: x

    
    def plot_radius_prediction_at_gt(self, ax_, gt_list, radius_prediction_map, circle_args=dict()):
        import matplotlib.patches as patches

        if type(ax_) not in [list, tuple, np.ndarray]:
            ax_ = [ax_]

        for a in ax_:                
            for pt in gt_list:
                radius = radius_prediction_map[int(pt[0]),int(pt[1])]
                circle = patches.Circle([pt[1],pt[0]], radius, **circle_args)
                
                # Add the circle to the axes
                a.add_patch(circle)


    def visualize_pylab(self, im, output, pred_mask=None, center_conv_resp=None, centerdir_gt=None,
                        gt_centers_dict=None, gt_difficult=None, log_r_fn=None, plot_batch_i=0, device=None, denormalize_args=None):
        with torch.no_grad():
            gt_list = np.array(
                [c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]), 0, gt_difficult.shape[0] - 1),
                                                         np.clip(int(c[1]), 0, gt_difficult.shape[1] - 1)].item() != 0
                                            for c in gt_list])

            from models.center_groundtruth import CenterDirGroundtruth
            gt_R, gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_R','gt_sin_th', 'gt_cos_th'])
            gt_center_mask = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_center_mask'])
            gt_shape_coef = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_shape_coef'])
            
            SHAPE_DIMS = gt_shape_coef.shape[1]

            if log_r_fn is not None:
                gt_R = log_r_fn(gt_R)
            
            _, ax = self.display(im[plot_batch_i].cpu(), 'image', denormalize_args=denormalize_args)
            self.plot_gt(ax, gt_list, is_difficult_gt)

            if self.shape_type in [ShapeType.CIRCLE,ShapeType.CIRCLE_WITH_CIRCULARITY]:
                self.plot_radius_prediction_at_gt(ax, gt_list, self.inverse_log_fn(output[plot_batch_i, 3]), 
                                                  circle_args=dict(linewidth=2, edgecolor='blue', facecolor='none'))
            
                self.plot_radius_prediction_at_gt(ax, gt_list, gt_shape_coef[plot_batch_i, 0, 0], 
                                                  circle_args=dict(linewidth=1, edgecolor='red', facecolor='none'))

            centers_viz = [(output[plot_batch_i, 2]).detach().cpu(), # center-dir R
                           (output[plot_batch_i, 1]).detach().cpu(), # center-dir cos/sin
                           (output[plot_batch_i, 0]).detach().cpu()] # center-dir cos/sin
            
            centers_viz += [(output[plot_batch_i, 3+i]).detach().cpu() for i in range(SHAPE_DIMS)]

            centers_gt_viz = []
            
            if centerdir_gt is not None and len(centerdir_gt) > 0:

                # diff between predicted and gt
                centers_viz += [torch.abs(output[plot_batch_i, 2] - gt_R[plot_batch_i, 0].to(device)).detach().cpu(),       # center-dir R
                                torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),  # center-dir cos/sin
                                torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu()]  # center-dir cos/sin
                
                centers_viz += [torch.abs(output[plot_batch_i, 3+i] - self.log_fn(gt_shape_coef[plot_batch_i, i]).to(device)).detach().cpu() for i in range(SHAPE_DIMS)]

                # just gt values
                centers_gt_viz += [gt_R[plot_batch_i, 0].cpu(),
                                   gt_sin_th[plot_batch_i, 0].cpu(),
                                   gt_cos_th[plot_batch_i, 0].cpu()]
                centers_gt_viz += [gt_shape_coef[plot_batch_i, i].cpu() for i in range(SHAPE_DIMS)]                                   

            _, ax = self.display(centers_viz, 'centers')
            self.plot_gt(ax, gt_list, is_difficult_gt)

            if center_conv_resp is not None:
                conv_centers = center_conv_resp[plot_batch_i].detach().cpu()
                if gt_center_mask is not None:
                    conv_centers = [conv_centers, torch.abs(center_conv_resp[plot_batch_i] - gt_center_mask[plot_batch_i,0].to(device)).detach().cpu()]
                
                _, ax = self.display(conv_centers, 'conv_centers')
                self.plot_gt(ax, gt_list, is_difficult_gt)

            cls = output[plot_batch_i][-1].cpu()
            self.display(cls, 'class', vmin=0, vmax=1)

            if len(centers_gt_viz) > 0:
                _, ax = self.display(centers_gt_viz, 'centerdir_gt')
                self.plot_gt(ax, gt_list, is_difficult_gt)
