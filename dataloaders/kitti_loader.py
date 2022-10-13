# ------------------------------------------------------------
# dataloader for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------


import torch.utils.data as data

from dataloaders.utils import *
from dataloaders.paths_and_transform import *
from dataloaders import CoordConv

input_options = ['dep', 'rgb', 'rgbd', 'g', 'gd']


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform_ = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform_
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        sparse = depth_read(self.paths['dep'][index]) if \
            (self.paths['dep'][index] is not None) else None
        gt = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None) else None
        penetgt = depth_read(self.paths['penetgt'][index]) if \
            self.paths['penetgt'][index] is not None else None
        s2dgt = depth_read(self.paths['s2dgt'][index]) if \
            self.paths['s2dgt'][index] is not None else None
        structure = s2r_read(self.paths['structure'][index]) if \
            (self.paths['structure'][index] is not None) else None
        return sparse, gt, penetgt, s2dgt, rgb, structure, self.paths['dep'][index]

    def __getitem__(self, index):

        sparse_dirty, gt, penetgt, s2dgt, rgb_raw, structure, d_path = self.__getraw__(index)
        position = CoordConv.AddCoordsNp(self.args.val_h, self.args.val_w)
        position = position.call()
        rgb = rgb_raw

        ipbasicgt = None
        if self.args.fill_type == 'fast':
            ipbasicgt = fill_in_fast(sparse_dirty, max_depth=100.0,
                                     extrapolate=self.args.extrapolate, blur_type=self.args.blur_type)
        elif self.args.fill_type == 'multiscale':
            ipbasicgt = fill_in_multiscale(sparse_dirty, max_depth=100.0,
                                           extrapolate=self.args.extrapolate, blur_type=self.args.blur_type)

        sparse_dirty, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position = self.transform(sparse_dirty, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, self.args)
        rgb, gray = handle_gray(rgb)

        candidates = {"rgb": rgb,  "dep": sparse_dirty, "gt": gt,
                      "gray": gray, 'position': position, 'K': self.K}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        if self.args.debug_dp or self.args.test:
            items['d_path'] = d_path

        return items

    def __len__(self):
        if self.args.toy_test:
            return self.args.toy_test_number
        else:
            return len(self.paths['gt'])

