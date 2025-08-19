import os
from functools import partial
import numpy as np
import sys
import torch
from torch import nn as nn

from utils.morphology import Dilation2d
from utils.utils import GaussianLayer

from utils.utils import variable_len_collate

def make_generic_regression_map(gt_map, out_key, id_x, id_y, out_id, dev):
    out = torch.zeros(size=gt_map.shape, dtype=torch.float, device=dev, requires_grad=False)
    out[:, out_id[:,0], out_id[:,1]] = gt_map[:, id_x, id_y]

    # this is for compatability with the older implementation where output must be [batch,N,1,W,H] (batch is added later)
    if len(out.shape) < 4:
        out = out.unsqueeze(1)

    return {out_key:out}

def make_orientation_map(orientation, key, id_x, id_y, out_id, dev, orientation_type):

    ROT_DIMS, height, width = orientation.shape

    # orientation may hold confidence score at the end as well
    has_confidence_score = ROT_DIMS == 2 or ROT_DIMS == 4
    if has_confidence_score:
        ROT_DIMS -= 1

    assert ROT_DIMS > 0 and ROT_DIMS < 4

    angle = orientation[0:ROT_DIMS, id_x, id_y]

    maps = dict()
    if orientation_type.lower() in ['euler','axis']:
        # reparametrize 3D using sin/cos for Euler or axis-aligned angles
        maps['gt_orientation_sin'] = torch.zeros(size=[ROT_DIMS, 1, height, width], dtype=torch.float, device=dev, requires_grad=False)
        maps['gt_orientation_cos'] = torch.zeros(size=[ROT_DIMS, 1, height, width], dtype=torch.float, device=dev, requires_grad=False)

        maps['gt_orientation_sin'][:, 0, out_id[:,0], out_id[:,1]] = torch.sin(angle)
        maps['gt_orientation_cos'][:, 0, out_id[:,0], out_id[:,1]] = torch.cos(angle)

    elif orientation_type.lower() in ['quaternion']:
        assert ROT_DIMS == 3, "Input error: quaternion can only be computed from 3D angle"

        # convert orientation from euler into quaternion
        from utils.utils_3d import euler_to_quaternion_matrix
        
        maps['gt_orientation_quaternion'] = torch.zeros(size=[4, 1, height, width], dtype=torch.float, device=dev, requires_grad=False)
        maps['gt_orientation_quaternion'][:, 0, out_id[:,0], out_id[:,1]] = euler_to_quaternion_matrix(angle)
        
    elif orientation_type.lower() in ['3d-so(3)']:
        raise Exception("Not implemneted yet")
    else:
        raise Exception("Unknown orientation representation type")
    
    if has_confidence_score:
        maps['gt_confidence_score'] = torch.zeros(size=[1, 1, height, width], dtype=torch.float, device=dev, requires_grad=False)
        maps['gt_confidence_score'][0, 0, out_id[:,0], out_id[:,1]] = orientation[-1, id_x, id_y]

    return maps


class CenterDirGroundtruth(nn.Module):
    def __init__(self, centerdir_gt_cache=None, extend_instance_mask_weights=0, MAX_NUM_CENTERS=100,
                 backbone_output_cache=None, load_cached_backbone_output_probability=0,
                 use_cached_backbone_output=False, save_cached_backbone_output_only=False,
                 add_synthetic_output=False, ignore_instance_mask_and_use_closest_center=False,
                 center_ignore_px=3, center_gt_blur=2, skip_gt_center_mask_generate=False, orientation_type='euler',
                 generic_regression_maps=[]):
        super().__init__()

        self.centerdir_gt_cache = centerdir_gt_cache

        self.backbone_output_cache = backbone_output_cache
        self.load_cached_backbone_output_probability = load_cached_backbone_output_probability
        self.use_cached_backbone_output = use_cached_backbone_output
        self.save_cached_backbone_output_only = save_cached_backbone_output_only

        self.add_synthetic_output = add_synthetic_output

        self.ignore_instance_mask_and_use_closest_center = ignore_instance_mask_and_use_closest_center
        self.extend_instance_mask_weights = extend_instance_mask_weights

        self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

        self.center_ignore_px = center_ignore_px
        self.skip_gt_center_mask_generate = skip_gt_center_mask_generate

        self.custom_regression_maps = {tuple(key): make_generic_regression_map for key in generic_regression_maps}
        self.custom_regression_maps['orientation'] = partial(make_orientation_map, orientation_type=orientation_type)

        ################################################################
        # Prepare location/coordinate map
        xym = self._create_xym(0)

        self.register_buffer("xym", xym, persistent=False)

        ################################################################
        # Prepare dilation op for extending GT

        if self.extend_instance_mask_weights:
            with torch.no_grad():
                self.instance_dilation_op = Dilation2d(1, 1, kernel_size=self.extend_instance_mask_weights, soft_max=False)

        with torch.no_grad():
            self.gaussian_blur = GaussianLayer(num_channels=1, sigma=center_gt_blur)

        if self.centerdir_gt_cache is not None and os.path.exists(self.centerdir_gt_cache) is False:
            os.makedirs(self.centerdir_gt_cache)

    def _create_xym(self, size):
        # coordinate map
        # CAUTION: original code may not have correctly aligned offsets
        #          since xm[-1] will not be size-1, but size
        #          -> this has been fixed by adding +1 element to xm and ym array
        align_fix = 1

        xm = torch.linspace(0, 1, size + align_fix).view(1, 1, -1).expand(1, size + align_fix, size + align_fix) * size
        ym = torch.linspace(0, 1, size + align_fix).view(1, -1, 1).expand(1, size + align_fix, size + align_fix) * size
        return torch.cat((xm, ym), 0)

    def _get_xym(self, height, width):
        max_size = max(height, width)
        if max_size > min(self.xym.shape[1], self.xym.shape[2]):
            self.xym = self._create_xym(max_size).to(self.xym.device)

        return self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

    def forward(self, sample, batch_index):
        # generate groundtruth maps for each sample in batch 
        with torch.no_grad():
            batch_size, _, height, width = sample['instance'].shape

            xym_s = self._get_xym(height, width)

            centerdir_gt = []
            for b in range(batch_size):                
                centerdir_gt_b = dict(height=height, width=width)

                # first generate gt_centers list, instance mask, and an associated center pos for every pixel
                gt_centers, (gt_center_x, gt_center_y), valid_instance_mask = self._get_centers_and_instance_masks(sample, b, batch_index, xym_s)

                # add list of centers
                centerdir_gt_b['gt_centers'] = gt_centers
                
                # append maps for center direction
                centerdir_gt_b.update(self._generate_center_directions(sample, b, gt_centers, gt_center_x, gt_center_y, valid_instance_mask, xym_s))

                # append maps for orientation
                centerdir_gt_b.update(self._generate_custom_regression_map(sample, b, gt_centers, gt_center_x, gt_center_y, valid_instance_mask, xym_s))

                # append maps for extented instances
                centerdir_gt_b.update(self._generate_extented_instance_mask(sample, b, gt_centers, gt_center_x, gt_center_y, valid_instance_mask, xym_s))

                centerdir_gt.append(centerdir_gt_b)

            # convert from a list of dict to a dict of tensors with first dimension as batch index
            centerdir_gt = variable_len_collate(centerdir_gt)

        if 'instance_polygon' in sample:
            centerdir_gt['instance_polygon'] = sample['instance_polygon']
        
        if 'image_px_in_nm' in sample:
            centerdir_gt['image_px_in_nm'] = sample['image_px_in_nm']

        # also create synthetic output if does not exist yet
        if self.add_synthetic_output:
            output = sample.get('output')
            label = sample.get('label')
            if output is None:
                output = self._create_synthetic_output(centerdir_gt, label)

            sample['output'] = output

        # attach centerdir groundtruh and output values to returned sample
        sample['centerdir_groundtruth'] = centerdir_gt

        return sample

    def _get_centers_and_instance_masks(self, sample, b, batch_index, xym_s):
        instances = sample['instance'][b,0]
        sample_name = sample['im_name'][batch_index[b].item()]
        centers = sample['center'][b]

        instance_ids_b = instances.unique()

        gt_centers = torch.zeros(size=(self.MAX_NUM_CENTERS, 2), dtype=torch.float, device=xym_s.device)

        gt_center_x, gt_center_y = None, None

        # version that assigns closest distance to each pixel and computes results for the whole image at once
        if self.ignore_instance_mask_and_use_closest_center:
            # requires list of centers first
            assert centers is not None

            # list of all centers valid for this batch
            valid_centers = (centers[:,0] > 0) | (centers[:,1] > 0)
            gt_centers[valid_centers,:] = (centers[valid_centers][:,[1,0]]).float()

            # skip if no centers
            if valid_centers.sum() > 0:
                assigned_center_ids = CenterDirGroundtruth.find_closest_center(centers, instances, xym_s)

                # per-pixel center locations
                gt_center_x = centers[assigned_center_ids[:], 1].unsqueeze(0)
                gt_center_y = centers[assigned_center_ids[:], 0].unsqueeze(0)

            # all pixels are considered as instance mask since we may not have valid instance mask at all
            instance_mask = torch.ones_like(instances.unsqueeze(0), dtype=torch.bool)
        else:
            # assign centers based on instance map (if there are any centers)

            if len(instance_ids_b) > 1:
                gt_center_x = torch.ones_like(instances.unsqueeze(0), dtype=torch.float) * -10000
                gt_center_y = torch.ones_like(instances.unsqueeze(0), dtype=torch.float) * -10000

                for id in instance_ids_b:
                    if id <= 0: continue
                    if id > len(centers):
                        print("ERROR: GOT OUT OF BOUNDS INDEX:  %d for img: " % id, sample_name)
                        sys.stdout.flush()
                        continue

                    in_mask = instances.eq(id).unsqueeze(0)

                    if centers is None:
                        # calculate center of attraction
                        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                        center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        center = centers[id.item()-1,:]

                    gt_centers[id.item()] = center.squeeze()[[1, 0]]

                    gt_center_x[in_mask] = center[1].float()
                    gt_center_y[in_mask] = center[0].float()

                instance_mask = instances.unsqueeze(0) > 0

        return gt_centers, (gt_center_x, gt_center_y), instance_mask

    def _generate_center_directions(self, sample, b, gt_centers, gt_center_x, gt_center_y, instance_mask, xym_s):
        instances = sample['instance'][:,0]

        centerdir_gt_b = {}

        gt_center_ignore = torch.ones_like(instances[b:b+1], dtype=torch.uint8, device=xym_s.device)
        # we need separate buffer for gt_center_mask if center_ignore_px is zero since we will not
        # have any GT values otherwise (just reuse gt_center_ignore if self.center_ignore_px > 0 )
        gt_center_mask = torch.ones_like(instances[b:b + 1], dtype=torch.uint8, device=xym_s.device) if self.center_ignore_px <= 0 else gt_center_ignore

        # do nothing if no centers
        if gt_center_x is not None and gt_center_y is not None:

            gt_X = gt_center_x - xym_s[1].unsqueeze(0)
            gt_Y = gt_center_y - xym_s[0].unsqueeze(0)

            if self.center_ignore_px <= 0:
                # just set gt_center_mask with default distance vals but not gt_center_ignore
                if not self.skip_gt_center_mask_generate:
                    gt_center_mask *= ~((gt_X.abs() < 3) * (gt_Y.abs() < 3))
            else:
                gt_center_ignore *= ~((gt_X.abs() < self.center_ignore_px) * (gt_Y.abs() < self.center_ignore_px))
                gt_center_mask = gt_center_ignore

            gt_R = torch.sqrt(torch.pow(gt_X, 2) + torch.pow(gt_Y, 2)) * instance_mask.float()
            gt_theta = torch.atan2(gt_Y, gt_X)
            gt_sin_th = torch.sin(gt_theta) * instance_mask.float()
            gt_cos_th = torch.cos(gt_theta) * instance_mask.float()

            # normalize groundtruth vector to 1 for all instance pixels
            gt_M = torch.sqrt(torch.pow(gt_sin_th, 2) + torch.pow(gt_cos_th, 2))
            gt_sin_th[instance_mask] = gt_sin_th[instance_mask] / gt_M[instance_mask]
            gt_cos_th[instance_mask] = gt_cos_th[instance_mask] / gt_M[instance_mask]

            centerdir_gt_b['gt_R'] = gt_R
            centerdir_gt_b['gt_theta'] = gt_theta
            centerdir_gt_b['gt_sin_th'] = gt_sin_th
            centerdir_gt_b['gt_cos_th'] = gt_cos_th

        if gt_center_mask.all() or self.skip_gt_center_mask_generate:
            gt_center_mask = torch.zeros_like(gt_center_mask)
        else:
            gt_center_mask = self.gaussian_blur(1 - gt_center_mask.unsqueeze(0).float())[0]
            gt_center_mask /= gt_center_mask.max()

        centerdir_gt_b['gt_center_ignore'] = gt_center_ignore
        centerdir_gt_b['gt_center_mask'] = gt_center_mask

        return centerdir_gt_b

    def _generate_extented_instance_mask(self, sample, b, gt_centers, gt_center_x, gt_center_y, instance_mask, xym_s):
        instances = sample['instance'][b,0]

        centerdir_gt_b = {}

        if self.extend_instance_mask_weights:
            gt_extended_instances = instances.clone().type(torch.int16)
            bg_mask = instances == 0

            # ditlate samples only when using instance mask and not closest center 
            if not self.ignore_instance_mask_and_use_closest_center:

                for id in instances.unique():
                    if id <= 0: continue

                    in_mask = instances.eq(id).unsqueeze(0)

                    mask_dilated = self.instance_dilation_op((in_mask.unsqueeze(0) > 0).float()).squeeze() > 0

                    gt_extended_instances[mask_dilated * bg_mask] = id.type(gt_extended_instances.dtype)
            
            centerdir_gt_b['gt_extended_instances'] = gt_extended_instances
            centerdir_gt_b['gt_extend_instance_mask_weights'] = self.extend_instance_mask_weights

        return centerdir_gt_b

    def _generate_custom_regression_map(self, sample, b, gt_centers, gt_center_x, gt_center_y, instance_mask, xym_s):

        centerdir_gt_b = {}        
        
        if gt_center_x is not None and gt_center_y is not None:
            # locations where original values will be read from
            id_x = gt_center_x[instance_mask].long()
            id_y = gt_center_y[instance_mask].long()
            
            # locations where values are stored to
            out_id = torch.nonzero(instance_mask[0])

            for keys, fn in self.custom_regression_maps.items():
                key,out_key = keys if type(keys) in [tuple,list] else (keys,keys)

                if key in sample:
                    centerdir_gt_b.update(fn(sample[key][b], out_key, id_x, id_y, out_id, xym_s.device))

        return centerdir_gt_b
    
    @staticmethod
    def parse_groundtruth(centerdir_gt, ignore_mask=None, return_orientation=False, return_radius=False, return_flux=False):
        if return_orientation:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_orientation_sin', 'gt_orientation_cos',
                    'gt_centers', 'gt_center_ignore', 'gt_center_mask', 'gt_extended_instances']
        elif return_radius:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_radius',
                    'gt_centers', 'gt_center_ignore', 'gt_center_mask', 'gt_extended_instances']
        elif return_flux:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_flux',
                    'gt_centers', 'gt_center_ignore', 'gt_center_mask', 'gt_extended_instances']
        else:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th',  'gt_centers', 'gt_center_ignore',
                    'gt_center_mask', 'gt_extended_instances']

        return CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys)

    @staticmethod
    def parse_groundtruth_map(centerdir_gt, keys=None):
        if centerdir_gt is None:
            centerdir_gt = {}
            out_size = [0,0,0]
        else:
            if len(centerdir_gt['height'].shape) > 0:
                out_size = [len(centerdir_gt['height']), int(centerdir_gt['height'][0]), int(centerdir_gt['width'][0])]
            else:
                out_size = [int(centerdir_gt['height']), int(centerdir_gt['width'])]
        
        ret = [centerdir_gt[k] if k in centerdir_gt else torch.zeros(out_size) for k in keys]
        return ret if len(keys) > 1 else ret[0] 
        
    @staticmethod
    def get_groundtruth_instance_polygon(centerdir_gt):
        return centerdir_gt.get('instance_polygon')

    @staticmethod
    def convert_gt_centers_to_dictionary(gt_centers, instances, ignore=None):
        gt_centers_dict = []
        for b in range(len(gt_centers)):
            valid_idx = torch.nonzero(torch.logical_and(gt_centers[b, :, 0] > 0, gt_centers[b, :, 1] > 0)).squeeze()
            present_idx = torch.unique(instances[b])

            valid_idx = valid_idx.cpu().numpy().reshape((-1,))
            present_idx = present_idx.cpu().numpy()
            # extract centers from valid idx that are also present in instances
            center_dict = {id: gt_centers[b, id, :2].cpu().numpy()
                                for id in set(valid_idx).intersection(present_idx)}

            # ignore centers that fall within ignore region
            if ignore is not None:
                center_dict = {k: c for k, c in center_dict.items() if ignore[b, 0][instances[b] == k].min() == 0}

            gt_centers_dict.append(center_dict)

        return gt_centers_dict

    def get_load_cache_fn(self):
        return lambda f, h, w: self._load_from_cache(f,h,w)

    def _load_from_cache(self, filename, h, w): 
        raise Exception("Caching of centerdir_groundtruth is not supported any more !!")

    def _create_synthetic_output(self, centerdir_gt, label):
        gt_R, gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_R', 'gt_sin_th', 'gt_cos_th'])
        output = torch.cat((gt_sin_th,
                            gt_cos_th,
                            torch.log10(gt_R + 1),
                            label.float()),dim=1)
        return output

    def insert_cached_model(self, model, sample):

        if self.use_cached_backbone_output == False and self.save_cached_backbone_output_only == False:
            return model
        else:
            # save actual model (model.module) to prevent recursive construction (and consequential GPU memory leaks)
            return CachedOutputModel(model.module, sample['output'], sample['im_name'],
                                     self.use_cached_backbone_output,
                                     self.save_cached_backbone_output_only,
                                     self.backbone_output_cache)
    @staticmethod
    def find_closest_center(centers, instances, xym_s):
        assert len(instances.shape) == 2

        height, width = instances.shape

        X, Y = xym_s[1], xym_s[0]

        # function used to calc closest distance
        def _calc_closest_center_patch_i(_center_x,_center_y, _X, _Y):
            # distance in cartesian space to center
            distances_to_center = torch.sqrt((_X[...,None] - _center_x) ** 2 + (_Y[..., None] - _center_y) ** 2)

            closest_center_index = torch.argmin(distances_to_center, dim=-1)

            return closest_center_index.long()

        # select patch size that is dividable but still as large as possible (from ranges of 16 to 128 - from 2**4 to 2**7)
        patch_size_options = [(2**i)*(2**j) for j in range(4,8) for i in range(4,8)]
        patch_size_options = [s for s in patch_size_options if height*width % s == 0]
        patch_size = max(patch_size_options)
        patch_count = (height*width) // patch_size

        # reshape all needed matrices into new shape
        (_X, _Y) = [x.reshape(patch_count,-1)  for x in [X, Y]]

        # main section to calc closest dist by splitting it
        valid_centers = (centers[:, 0] > 0) | (centers[:, 1] > 0) # use only valid centers and then remap indexes

        center_y = centers[valid_centers, 0]
        center_x = centers[valid_centers, 1]

        closest_center_index = torch.zeros((patch_count, patch_size), dtype=torch.long, device=instances.device)
        for i in range(patch_count):
            closest_center_index[i, :] = _calc_closest_center_patch_i(center_x, center_y, _X[i], _Y[i])

        # re-map indexes from list of selected/valid center to list of all centers
        valid_centers_idx = torch.nonzero(valid_centers)
        closest_center_index = valid_centers_idx[closest_center_index]

        return closest_center_index.reshape(instances.shape)



class CachedOutputModel:
    def __init__(self, module, output, im_names, use_cached_backbone_output, save_cached_backbone_output_only,
                 backbone_output_cache):
        self.module = module
        self.output = output
        self.im_names = im_names
        self.use_cached_backbone_output = use_cached_backbone_output
        self.save_cached_backbone_output_only = save_cached_backbone_output_only
        self.backbone_output_cache = backbone_output_cache

    def __call__(self, input):
        if self.use_cached_backbone_output:
            output = self.output.to(input.device)
        else:
            output = self.module(input)

        if self.save_cached_backbone_output_only:
            try: os.makedirs(self.backbone_output_cache)
            except: pass

            for i in range(len(input)):
                filename = os.path.basename(self.im_names[i])
                cached_output_filename = os.path.join(self.backbone_output_cache,
                                                      '%s_output_cache.npy' % filename)
                np.save(cached_output_filename, np.array(output[i, :].detach().cpu(), dtype=np.float32))
            output = None

        return output