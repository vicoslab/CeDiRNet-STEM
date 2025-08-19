import numpy as np
import torch
from torch.utils.data import Sampler
import pickle

from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess

class IHardExamplesBatchSampler:
    def has_hard_samples(self):
        raise Exception("Not implemented")
    def update_difficulty_score(self, gt_sample, difficulty_scores, index_key='index', storage_keys=[], selected_samples_only=None):
        raise Exception("Not implemented")
    def retrieve_hard_sample_storage_batch(self, ids, key=None):
        raise Exception("Not implemented")

    def get_difficulty_scores(self):
        raise Exception("Not implemented")
    def get_hard_example_indices(self):
        raise Exception("Not implemented")
    def get_avg_difficulty_score(self):
        raise Exception("Not implemented")
    def get_sample_frequency_use(self):
        raise Exception("Not implemented")

class DistributedBatchSampler(Sampler):

    def __init__(self, default_sampler, batch_size, drop_last,
                 device=None, world_size=None, rank=None, is_distributed=False):
        if not isinstance(default_sampler, Sampler):
            raise ValueError("default_sampler should be an instance of "
                             "torch.utils.data.Sampler, but got default_sampler={}"
                             .format(default_sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.is_distributed = is_distributed and world_size > 1
        self.world_size = world_size if self.is_distributed else 1
        self.rank = rank if self.is_distributed else 0
        self.device = device

        self.default_sampler = default_sampler

        self.batch_size = batch_size
        self.drop_last = drop_last

    def _synchronize_dict(self, array):
        return distributed_sync_dict(array, self.world_size, self.rank, self.device)

    def __iter__(self):
        from itertools import islice
        
        max_index = len(self.default_sampler)
        if self.drop_last:
            total_batch_size = self.batch_size * self.world_size
            max_index = (max_index // total_batch_size) * total_batch_size

        batch = []
        self.usage_freq = {i: 0 for i in range(len(self.default_sampler))}
        for idx in islice(self.default_sampler,self.rank,max_index,self.world_size):
            batch.append(idx)
            # stop when spaces for normal samples filled
            if len(batch) == self.batch_size:
                for b in batch: self.usage_freq[b] += 1
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            for b in batch: self.usage_freq[b] += 1
            yield batch

    def __len__(self):
        size_default = len(self.default_sampler)

        if self.is_distributed:
            size_default = size_default // self.world_size

        if self.drop_last:
            return size_default // self.batch_size
        else:
            return (size_default + self.batch_size - 1) // self.batch_size

import torch.distributed as dist

class DistributedRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, device=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.device = device

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            iter_order = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).to(self.device)
        else:
            iter_order = torch.randperm(n).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return iter(iter_order.tolist())

    def __len__(self):
        return self.num_samples


class DistributedSubsetRandomSampler(Sampler):
    def __init__(self, indices, device=None):
        self.indices = indices
        self.device = device

    def __iter__(self):
        iter_order = torch.randperm(len(self.indices)).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return (self.indices[i.item()] for i in iter_order)

    def __len__(self):
        return len(self.indices)

def distributed_sync_dict(array, world_size, rank, device, MAX_LENGTH=10*2**20): # default MAX_LENGTH = 10MB
    def _pack_data(_array):
        data = pickle.dumps(_array)
        data_length = int(len(data))
        data = data_length.to_bytes(4, "big") + data
        assert len(data) < MAX_LENGTH
        data += bytes(MAX_LENGTH - len(data))
        data = np.frombuffer(data, dtype=np.uint8)
        assert len(data) == MAX_LENGTH
        return torch.from_numpy(data)
    def _unpack_data(_array):
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        return pickle.loads(data[4:data_length+4])
    def _unpack_size(_array):
        print(_array.shape, _array[:4])
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        print(data_length,data[:4])
        return data_length

    # prepare output buffer
    output_tensors = [torch.zeros(MAX_LENGTH, dtype=torch.uint8, device=device) for _ in range(world_size)]
    # pack data using pickle into input/output
    output_tensors[rank][:] = _pack_data(array)

    # sync data
    dist.all_gather(output_tensors, output_tensors[rank])

    # unpack data and merge into single dict
    return {id:val for array_tensor in output_tensors for id,val in _unpack_data(array_tensor).items()}

