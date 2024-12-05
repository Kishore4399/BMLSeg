import os
import sys
import random
import builtins
import warnings
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import argparse
from omegaconf import OmegaConf
from collections import deque

from sklearn.metrics import roc_auc_score

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True

def dist_setup(args):
    torch.multiprocessing.set_start_method('fork', force=True)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

def get_conf():

    conf_file = sys.argv[1]
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)

    parser = argparse.ArgumentParser(description='Self Supervised-learning pre-training')
    parser.add_argument('conf_file', type=str, help='path to config file')

    for key in conf:
        if conf[key] is None:
            parser.add_argument(f'--{key}', default=None)
        else:
            if key == "gpu":
                parser.add_argument('--gpu', type=int, default=conf[key])
            elif key == 'multiprocessing_distributed':
                parser.add_argument('--multiprocessing_distributed', type=bool, default=conf[key])
            else:
                parser.add_argument(f'--{key}', type=type(conf[key]), default=conf[key])
            
    args = parser.parse_args()

    if args.gpu:
        args.gpu = int(args.gpu)
    
    args.output_dir = '/'.join([*(args.output_dir.split('/')[:-1]), args.run_name])
    if not hasattr(args, 'num_samples'):
        args.num_samples = 4
    print(args)
    return args

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        print(f'count: {self.count} | total: {self.total}')

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
    
@torch.no_grad()
def concat_all_gather(tensor, distributed=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if distributed:
        dist.barrier()
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        # print(f"World size: {dist.get_world_size()}")
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor

def compute_aucs(pred, gt):
    auc_list = []
    if pred.ndim == 2:
        n_classes = pred.shape[1]
    elif pred.ndim == 1:
        n_classes = 1
    else:
        raise ValueError("Prediction shape wrong")
    for i in range(n_classes):
        try:
            auc = roc_auc_score(gt[:, i], pred[:, i])
        except (IndexError, ValueError) as error:
            if isinstance(error, IndexError):
                auc = roc_auc_score(gt, pred)
            elif isinstance(error, ValueError):
                auc = 0
            else:
                raise Exception("Unexpected Error")
        auc_list.append(auc)
    mAUC = np.mean(auc_list)
    return auc_list, mAUC