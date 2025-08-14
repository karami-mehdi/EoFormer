import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import torch.distributed as dist
import warnings
from medpy.metric import binary

    
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
 
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(
            len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank *
                          self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples

def calculate_metric(y_pred=None, y=None, eps=1e-6):
    
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
    
    batch_size, n_class = y_pred.shape[:2]
    
    dsc = np.empty((batch_size, n_class))
    iou = np.empty((batch_size, n_class))
    cnt = np.zeros((n_class))
    yp_lab = np.argmax(y_pred, axis=1)
    yt_lab = np.argmax(y, axis=1)
    acc = (yp_lab == yt_lab).sum() / (yp_lab.size + eps)
    for b, c in np.ndindex(batch_size, n_class):
        edges_pred, edges_gt = y_pred[b, c], y[b, c]
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan distance.")
        p = edges_pred > 0.5
        g = edges_gt > 0.5
        sp = p.sum()
        sg = g.sum()
        if (sp > 0 and sg > 0):
            inter = np.logical_and(p, g).sum()
            dsc[b, c] = (2.0 * inter + eps) / (sp + sg + eps)
            uni = sp + sg - inter
            iou[b, c] = (inter + eps) / (uni + eps)
            cnt[c] += 1
        elif (sp == 0 and sg == 0):
            dsc[b, c] = 1
            iou[b, c] = 1
            cnt[c] += 1
        else:
            dsc[b, c] = 0
            iou[b, c] = 0
            cnt[c] += eps
    dsc = np.sum(dsc, axis=0) / cnt
    iou = np.sum(iou, axis=0) / cnt
    dice_avg = np.mean(dsc)
    miou = np.mean(iou)
    
    return torch.from_numpy(dsc), torch.tensor(dice_avg), torch.tensor(miou), torch.tensor(acc)