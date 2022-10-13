# ------------------------------------------------------------
# some of useful functions are defined here.
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------

import os
import shutil

import torch

# import numpy as np
#
# def pad_rep(image, ori_size):
#     h, w = image.shape
#     oh, ow = ori_size
#     pl = (ow - w) // 2
#     pr = ow - w - pl
#     pt = oh - h
#     image_pad = np.pad(image, pad_width=((pt, 0), (pl, pr)), mode='edge')
#     return image_pad

import math
import torch.distributed

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Params:{total_params}")
    return total_params

# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
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
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", '*wandb*', '*test*', '*val*')

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if not args.pretrain == '':
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)
            new_args.defrost()
            new_args.start_epoch = checkpoint['epoch']
            new_args.freeze()
            pre_args = checkpoint['args']
            # check if the important parametes setting is same as the pre setting
            # * dataset
            assert new_args.data_name == pre_args.data_name
            assert new_args.patch_height == pre_args.patch_height
            assert new_args.patch_width == pre_args.patch_width
            assert new_args.top_crop == pre_args.top_crop
            assert new_args.max_depth == pre_args.max_depth
            assert new_args.augment == pre_args.augment
            assert new_args.num_sample == pre_args.num_sample
            assert new_args.test_crop == pre_args.test_crop
            # * loss
            assert new_args.loss == pre_args.loss
            # * apex level
            assert new_args.opt_level == pre_args.opt_level
            # * training
            assert new_args.epochs == pre_args.epochs
            # assert new_args.batch_size == pre_args.batch_size
            # * optimizer
            # assert new_args.lr == pre_args.lr
            assert new_args.optimizer == pre_args.optimizer
            assert new_args.momentum == pre_args.momentum
            assert new_args.betas == pre_args.betas
            assert new_args.epsilon == pre_args.epsilon
            assert new_args.weight_decay == pre_args.weight_decay
            assert new_args.scheduler == pre_args.scheduler
            assert new_args.decay_step == pre_args.decay_step
            assert new_args.decay_factor == pre_args.decay_factor

    return new_args

def count_validpoint(x: torch.Tensor):
    mask = x > 0.001
    num_valid = mask.sum()
    return int(num_valid.data.cpu())
