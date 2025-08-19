import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

import scipy
import torch.nn as nn

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        from utils.visualize.vis import Visualizer
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

class GaussianLayer(nn.Module):
    def __init__(self, num_channels=1, sigma=3):
        super(GaussianLayer, self).__init__()

        self.sigma = sigma
        self.kernel_size = int(2 * np.ceil(3*self.sigma - 0.5) + 1)

        self.conv = nn.Conv2d(num_channels, num_channels, self.kernel_size, stride=1,
                              padding=self.kernel_size//2, bias=None, groups=num_channels)


        self.weights_init()
    def forward(self, x):
        return self.conv(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size,self.kernel_size))
        n[self.kernel_size//2,self.kernel_size//2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))



####################################################################################################################
# General class that can merge split image results - can merge tensor data (image or heatmap) and points
class ImageGridCombiner:
    class Data:
        def __init__(self, w,h):
            self.full_size = [h,w]
            self.full_data = dict()
            self.image_names = []
        def add_image_name(self, name):
            self.image_names.append(name)

        def get_image_name(self):
            org_names = ["_patch".join(n.split("_patch")[:-1]) + ".png" for n in self.image_names]
            # check that all images had the same original name
            org_names = np.unique(org_names)

            if len(org_names) != 1:
                raise Exception("Invalid original names found: %s" % ",".join(org_names))

            return org_names[0]

        def set_tensor2d(self, name, partial_data, roi_x, roi_y, merge_op=None):
            if name not in self.full_data:
                # store data on CPU for large images to avoid excesive memory usage
                dev = partial_data.device if np.prod(self.full_size) < 2500*2500 else 'cpu'
                self.full_data[name] = torch.zeros(list(partial_data.shape[:-2]) + self.full_size, dtype=partial_data.dtype,
                                                   device=dev)

            full_data_roi = self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
            partial_data_roi = partial_data[..., 0:(roi_y[1] - roi_y[0]),0:(roi_x[1] - roi_x[0])].type(full_data_roi.dtype)
            try:
                # default merge operator is to use max unless specified otherwise
                if merge_op is None:
                    merge_op = lambda Y,X: torch.where(Y.abs() < X.abs(), X, Y)

                self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] = merge_op(full_data_roi, partial_data_roi.to(self.full_data[name].device))
            except:
                print('error')

        def set_instance_description(self, name, partial_instance_list, partial_instance_mask, reverse_desc, x, y):
            if name not in self.full_data:
                self.full_data[name] = ([],[])

            if len(partial_instance_list) > 0:
                if np.all([i != 0 for i,desc in partial_instance_list.items()]):
                    all_ids = partial_instance_mask.nonzero().cpu().numpy()
                    all_values = partial_instance_mask[all_ids[:,0],all_ids[:,1]].cpu().numpy()
                else:
                    all_ids, all_values = None, None
                for i,desc in partial_instance_list.items():
                    if all_ids is None or all_values is None:
                        # revert to slower but more correct version
                        i_mask = (partial_instance_mask == i).nonzero().cpu().numpy()
                    else:
                        i_mask = all_ids[all_values == i,:] if len(all_ids) > 0 else np.zeros((0,2))

                    self.full_data[name][0].append(desc + np.array(([y,x] if reverse_desc else [x,y]) + [0]*(len(desc)-2)))
                    self.full_data[name][1].append(i_mask + np.array([y, x]))

        def get(self, data_names):
            return [self.full_data.get(n) for n in data_names]

        def get_instance_description(self, name, out_shape, out_mask_tensor, overlap_thr=0, merge_dist_thr=0):
            instance_list, instance_mask_ids = self.full_data[name]

            # clip mask ids to out_mask_tensor.shape
            instance_mask_ids = [np.array([c for c in i_mask if c[0] >= 0 and c[1] >= 0 and c[0] < out_shape[0] and c[1] < out_shape[1]])
                                        for i_mask in instance_mask_ids]

            instance_indexes = [set(np.ravel_multi_index((np.array(i_mask)[:, 0], np.array(i_mask)[:, 1]), dims=out_shape)) if len(i_mask) > 0 else set()
                                        for i_mask in instance_mask_ids]

            retained = self._find_retained_instances(instance_indexes, overlap_thr)

            if merge_dist_thr > 0:
                for i, x_i in enumerate(instance_list):
                    if retained[i]:
                        for j, x_j in enumerate(instance_list):
                            if j > i and retained[j]:
                                dist = np.sqrt(np.sum(np.abs(x_i[:2]-x_j[:2])**2))
                                if dist < merge_dist_thr:
                                    retained[j] = False

            instance_list = [x for i, x in enumerate(instance_list) if retained[i]]
            instance_mask_ids = [x for i, x in enumerate(instance_mask_ids) if retained[i]]
            instance_indexes = [x for i, x in enumerate(instance_indexes) if retained[i]]

            instance_dict = {}
            instance_indexes_dict = {}

            for id, (i_center, i_mask, i_mask_indexes) in enumerate(zip(instance_list, instance_mask_ids, instance_indexes)):
                i_mask_ids = torch.from_numpy(i_mask)
                if len(i_mask_ids) > 0:
                    if out_mask_tensor is not None:
                        out_mask_tensor[(i_mask_ids[:, 0], i_mask_ids[:, 1])] = id + 1
                instance_dict[id + 1] = i_center
                instance_indexes_dict[id+1] = list(i_mask_indexes)

            return instance_dict, out_mask_tensor, instance_indexes_dict

        def _find_retained_instances(self, instance_indexes, overlap_thr):
            retained = np.ones(shape=len(instance_indexes), dtype=np.bool)
            for i in range(len(instance_indexes)):
                if retained[i]:
                    for j in range(i + 1, len(instance_indexes)):
                        inter_ij = len(instance_indexes[i].intersection(instance_indexes[j]))
                        iou_ratio = inter_ij / (len(instance_indexes[i]) + len(instance_indexes[j]) - inter_ij + 1e-5)
                        if iou_ratio > overlap_thr:
                            if len(instance_indexes[i]) > len(instance_indexes[j]):
                                retained[j] = False
                            else:
                                retained[i] = False
                                break
            return retained
    def __init__(self):
        self.current_index = None
        self.current_data = None

    def add_image(self, im_name, grid_index, data_map_tensor, data_map_instance_desc, custom_merge_ops={}):
        n, x, y, w, h, org_w, org_h = grid_index

        # set roi and clamp to max size
        roi_x = x, min(x + w, org_w)
        roi_y = y, min(y + h, org_h)

        finished_data = None

        if self.current_index is None or n != self.current_index:
            finished_data = self.current_data
            self.current_data = self.Data(org_w, org_h)
            self.current_index = n

        self.current_data.add_image_name(im_name)

        if n == self.current_index:
            for name,partial_data in data_map_tensor.items():
                if partial_data is not None:
                    self.current_data.set_tensor2d(name, partial_data, roi_x, roi_y, merge_op=custom_merge_ops.get(name))

            for name,partial_data in data_map_instance_desc.items():
                if partial_data is not None:
                    self.current_data.set_instance_description(name, partial_data[0], partial_data[1], partial_data[2], x,y)

        return finished_data

def tensor_mask_to_ids(mask):
    ids = {i.item(): (mask == i).nonzero().cpu().numpy()
                for i in mask.unique() if i > 0}
    ids = {i: set(np.ravel_multi_index((np.array(i_mask)[:, 0], np.array(i_mask)[:, 1]), dims=mask.shape[-2:]))
                for i, i_mask in ids.items()}

    return ids

def ids_to_tensor_maks(ids, out_shape):
    out = np.zeros(out_shape)
    for i, indices in ids.items():
        indices = np.unravel_index(indices,out_shape)
        out[(indices[0], indices[1])] = i

    return out

def instance_poly_to_variable_array(polygon_list):
    import numpy as np

    if polygon_list is not None and len(polygon_list) > 0:
        if type(polygon_list) in [list,tuple]:
            # convert from list of [Nx2] to [Nx3] where first value in second axis defines ID of instance
            polygon_list = [np.concatenate(((i + 1) * np.ones((len(p), 1)), p), axis=1) for i, p in enumerate(polygon_list)]
            polygon_list = np.concatenate(polygon_list, axis=0)
        elif len(polygon_list.shape) == 3:
            idx = np.expand_dims(np.repeat(np.expand_dims(np.arange(len(polygon_list)),1),(polygon_list.shape[1]),axis=1),2) + 1
            polygon_list = np.concatenate((idx,polygon_list),axis=2)
        elif len(polygon_list.shape) != 2 or polygon_list.shape[1] != 3:
            raise Exception("Invalid input polygon_list: should be list of Nx2, array of Nx2 or array of Nx3")
    else:
        polygon_list = np.zeros((0,3))

    return polygon_list

import torch
import re
import collections
#from torch._six import string_classes # torch._six does not exist in latest pytorch !!
string_classes = (str,)

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

from torch.nn.utils.rnn import pad_sequence

def variable_len_collate(batch, batch_first=True, padding_value=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        numel = [x.numel() for x in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            storage = elem.storage()._new_shared(sum(numel))
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out) if np.all(numel[0] == numel) else pad_sequence(batch,
                                                                                             batch_first=batch_first,
                                                                                             padding_value=padding_value)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return variable_len_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: variable_len_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(variable_len_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [variable_len_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_log_and_inverse_fn(log_base = None):
    log_base = 'exp' if log_base is None else log_base

    if log_base.lower() in ['exp', 'e']:
        log_fn = lambda x: (torch.log(x + 1) if type(x) is torch.Tensor else np.log(x + 1))
        inverse_log_fn = lambda x: ((torch.exp(x) - 1) if type(x) is torch.Tensor else (np.exp(x) - 1))
    elif log_base.lower() in ['decimal', '10']:
        log_fn = lambda x: (torch.log10(x + 1) if type(x) is torch.Tensor else np.log10(x + 1))
        inverse_log_fn = lambda x: ((torch.pow(10, x) - 1) if type(x) is torch.Tensor else (np.pow(10, x) - 1))
    elif log_base.lower() in ['pow10']:
        log_fn =  lambda x: (torch.log10(x + 1) if type(x) is torch.Tensor else np.log10(x + 1))
        inverse_log_fn = lambda x: ((torch.pow(x, 10) - 1) if type(x) is torch.Tensor else (np.pow(x, 10) - 1))
    else:
        raise Exception('Only "exp" and "10" are allowed logarithms')
    return log_fn, inverse_log_fn

from enum import Enum

class ShapeType(Enum):
    CIRCLE = 1
    CIRCLE_WITH_CIRCULARITY = 2
    ELIPSE = 3
