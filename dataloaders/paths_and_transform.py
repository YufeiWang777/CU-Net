# ------------------------------------------------------------
# dataloader for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------

import os
import glob
import torch
from dataloaders import transforms


GEOMETRIC = ['BottomCrop', 'HorizontalFlip']
TRAIN_D_AND_GT = ['GEOMETRIC', 'Random_crop']
TRIAN_TRANSFORM_RGB = ['ColorJitter', 'GEOMETRIC', 'Random_crop']
TRIAN_TRANSFORM_STRUCTURE = ['ColorJitter', 'GEOMETRIC', 'Random_crop']
TRAIN_POSITION = ['BottomCrop', 'Random_crop']

VAL_TRANSFORME = ['BottomCrop']

NO_TRANSFORM = []

DEBUG_DATALOADER = False
glob_d, glob_gt, glob_rgb, glob_s2r, transform = None, None, None, None, None
glob_penettruth, glob_s2dtruth = None, None
get_rgb_paths = None
pnew = None
get_penettruth_paths, get_s2dtruth_paths = None, None


def get_paths_and_transform(split, args):
    global glob_d, glob_gt, glob_rgb, glob_s2r, transform, get_rgb_paths, \
        get_penettruth_paths, get_s2dtruth_paths, glob_penettruth, glob_s2dtruth

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )
        glob_s2r = os.path.join(
            args.data_folder,
            'train/*_sync/image_0[2,3]/structure_data/*.npy'
        )
        def get_rgb_paths(p):
            global pnew
            if 'image_02' in p:
                pnew = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
            elif 'image_03' in p:
                pnew = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
            return pnew
        def get_penettruth_paths(p):
            pnew_ = p.replace('groundtruth', 'penettruth')
            return pnew_
        def get_s2dtruth_paths(p):
            pnew_ = p.replace('groundtruth', 's2dtruth')
            return pnew_
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            glob_s2r = os.path.join(
                args.data_folder,
                'val/*_sync/image_0[2,3]/structure_data/*.npy'
            )
            def get_rgb_paths(p):
                global pnew
                if 'image_02' in p:
                    pnew = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
                elif 'image_03' in p:
                    pnew = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
                return pnew
            def get_penettruth_paths(p):
                pnew_ = p.replace('groundtruth', 'penettruth')
                return pnew_
            def get_s2dtruth_paths(p):
                pnew_ = p.replace('groundtruth', 's2dtruth')
                return pnew_
        elif args.val == "select":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/velodyne_raw/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/groundtruth_depth/*.png'
            )
            glob_rgb = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/image/*.png'
            )
            glob_s2r = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/image_structure/*.npy'
            )
            glob_penettruth = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/penettruth_depth/*.png'
            )
            glob_s2dtruth = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/s2dtruth_depth/*.png'
            )
    elif split == "test_completion":
        transform = test_transform
        glob_d = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png'
        )
        glob_gt = None
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/image/*.png'
        )
        glob_s2r = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/image_structure/*.npy'
        )
    elif split == "test_prediction":
        transform = no_transform
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_prediction_anonymous/image/*.png'
        )
        glob_d = None
        glob_gt = None
        glob_s2r = None
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        if split == 'train' or (split == 'val' and args.val == 'full'):
            paths_rgb = [get_rgb_paths(p) for p in paths_d]
        else:
            paths_rgb = sorted(glob.glob(glob_rgb))
        if split == 'train' or (split == 'val' and args.val == 'full'):
            paths_penettruth = [get_penettruth_paths(p) for p in paths_gt]
            paths_s2dtruth = [get_s2dtruth_paths(p) for p in paths_gt]   
        else:
            paths_penettruth = sorted(glob.glob(glob_penettruth))
            paths_s2dtruth = sorted(glob.glob(glob_s2dtruth))      
        paths_s2r = sorted(glob.glob(glob_s2r))
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_s2r = sorted(glob.glob(glob_s2r))
        paths_gt = [None] * len(paths_rgb)
        paths_penettruth = [None] * len(paths_rgb)
        paths_s2dtruth = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))
            
    # DEBUG
    if DEBUG_DATALOADER:
        for i in range(999):
            print("#####")
            print(paths_rgb[i])
            print(paths_d[i])
            print(paths_gt[i])
            print(paths_s2r[i])
        raise OSError('debug end!')

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_s2r) == 0\
            and len(paths_penettruth) == 0 and len(paths_s2dtruth) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_s2r) == 0:
        raise (RuntimeError("Requested structure images but no structure was found"))
    if len(paths_penettruth) == 0:
        raise (RuntimeError("Requested penettruth images but no penettruth was found"))
    if len(paths_s2dtruth) == 0:
        raise (RuntimeError("Requested s2dtruth images but no s2dtruth was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt) or len(paths_gt) != len(paths_s2r)\
            or len(paths_gt) != len(paths_penettruth) or len(paths_gt) != len(paths_s2dtruth):
        print(len(paths_d), len(paths_gt), len(paths_penettruth), len(paths_s2dtruth), len(paths_rgb), len(paths_s2r))

    paths = {"rgb": paths_rgb, "dep": paths_d, "gt": paths_gt, 'penetgt': paths_penettruth, 's2dgt': paths_s2dtruth,
             "structure": paths_s2r}

    items = {}
    for key, val in paths.items():
        if key not in args.dataset:
            items[key] = [None] * len(paths_rgb)
        else:
            items[key] = val

    return items, transform

def train_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = args.val_h
    owidth = args.val_w

    # do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    flip = torch.FloatTensor(1).uniform_(0, 1)
    do_flip = flip.item() < 0.5

    # flip = np.random.uniform(0.0, 1.0)
    # do_flip = flip < 0.5
    # ic(flip)

    transforms_list = [
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]

    # if small_training == True:
    # transforms_list.append(transforms.RandomCrop((rheight, rwidth)))

    transform_geometric = transforms.Compose(transforms_list)

    if sparse is not None:
        sparse = transform_geometric(sparse)
    if gt is not None:
        gt = transform_geometric(gt)
    if ipbasicgt is not None:
        ipbasicgt = transform_geometric(ipbasicgt)
    if penetgt is not None:
        penetgt = transform_geometric(penetgt)
    if s2dgt is not None:
        s2dgt = transform_geometric(s2dgt)
    if rgb is not None:
        # brightness = np.random.uniform(max(0, 1 - args.jitter),
        #                                1 + args.jitter)
        # contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        # saturation = np.random.uniform(max(0, 1 - args.jitter),
        #                                1 + args.jitter)

        brightness = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter),
                                                   1 + args.jitter).item()
        contrast = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter), 1 + args.jitter).item()
        saturation = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter),
                                                   1 + args.jitter).item()

        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
    if structure is not None:
        # brightness = np.random.uniform(max(0, 1 - args.jitter),
        #                                1 + args.jitter)
        # contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        # saturation = np.random.uniform(max(0, 1 - args.jitter),
        #                                1 + args.jitter)

        brightness = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter),
                                                   1 + args.jitter).item()
        contrast = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter), 1 + args.jitter).item()
        saturation = torch.FloatTensor(1).uniform_(max(0, 1 - args.jitter),
                                                   1 + args.jitter).item()

        transform_structure = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        structure = transform_structure(structure)
    # sparse = drop_depth_measurements(sparse, 0.9)

    if position is not None:
        bottom_crop_only = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
        position = bottom_crop_only(position)

    # random crop
    # if small_training == True:
    if not args.not_random_crop:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        # i = np.random.randint(0, h - rheight + 1)
        # j = np.random.randint(0, w - rwidth + 1)

        # i = int(torch.FloatTensor(1).uniform_(0, h - rheight + 1).item())
        # FIXME j=0
        j = int(torch.FloatTensor(1).uniform_(0, w - rwidth + 1).item())

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[h-rheight:h, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[h-rheight:h, j:j + rwidth]
                
        if structure is not None:
            if structure.ndim == 3:
                structure = structure[h-rheight:h, j:j + rwidth, :]
            elif structure.ndim == 2:
                structure = structure[h-rheight:h, j:j + rwidth]

        if sparse is not None:
            if sparse.ndim == 3:
                sparse = sparse[h-rheight:h, j:j + rwidth, :]
            elif sparse.ndim == 2:
                sparse = sparse[h-rheight:h, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[h-rheight:h, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[h-rheight:h, j:j + rwidth]
                
        if ipbasicgt is not None:
            if ipbasicgt.ndim == 3:
                ipbasicgt = ipbasicgt[h-rheight:h, j:j + rwidth, :]
            elif ipbasicgt.ndim == 2:
                ipbasicgt = ipbasicgt[h-rheight:h, j:j + rwidth]
        
        if penetgt is not None:
            if penetgt.ndim == 3:
                penetgt = penetgt[h-rheight:h, j:j + rwidth, :]
            elif penetgt.ndim == 2:
                penetgt = penetgt[h-rheight:h, j:j + rwidth]
        
        if s2dgt is not None:
            if s2dgt.ndim == 3:
                s2dgt = s2dgt[h-rheight:h, j:j + rwidth, :]
            elif s2dgt.ndim == 2:
                s2dgt = s2dgt[h-rheight:h, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[h-rheight:h, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[h-rheight:h, j:j + rwidth]

    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

def val_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform_ = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform_(rgb)
    if structure is not None:
        structure = transform_(structure)
    if sparse is not None:
        sparse = transform_(sparse)
    if gt is not None:
        gt = transform_(gt)
    if ipbasicgt is not None:
        ipbasicgt = transform_(ipbasicgt)
    if penetgt is not None:
        penetgt = transform_(penetgt)
    if s2dgt is not None:
        s2dgt = transform_(s2dgt)
    if position is not None:
        position = transform_(position)

    if not args.val_not_random_crop:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        # i = np.random.randint(0, h - rheight + 1)
        # j = np.random.randint(0, w - rwidth + 1)

        # i = int(torch.FloatTensor(1).uniform_(0, h - rheight + 1).item())
        j = int(torch.FloatTensor(1).uniform_(0, w - rwidth + 1).item())

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[h - rheight:h, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[h - rheight:h, j:j + rwidth]

        if structure is not None:
            if structure.ndim == 3:
                structure = structure[h - rheight:h, j:j + rwidth, :]
            elif structure.ndim == 2:
                structure = structure[h - rheight:h, j:j + rwidth]

        if sparse is not None:
            if sparse.ndim == 3:
                sparse = sparse[h - rheight:h, j:j + rwidth, :]
            elif sparse.ndim == 2:
                sparse = sparse[h - rheight:h, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[h - rheight:h, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[h - rheight:h, j:j + rwidth]
        
        if ipbasicgt is not None:
            if ipbasicgt.ndim == 3:
                ipbasicgt = ipbasicgt[h - rheight:h, j:j + rwidth, :]
            elif ipbasicgt.ndim == 2:
                ipbasicgt = ipbasicgt[h - rheight:h, j:j + rwidth]

        if penetgt is not None:
            if penetgt.ndim == 3:
                penetgt = penetgt[h - rheight:h, j:j + rwidth, :]
            elif penetgt.ndim == 2:
                penetgt = penetgt[h - rheight:h, j:j + rwidth]

        if s2dgt is not None:
            if s2dgt.ndim == 3:
                s2dgt = s2dgt[h - rheight:h, j:j + rwidth, :]
            elif s2dgt.ndim == 2:
                s2dgt = s2dgt[h - rheight:h, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[h - rheight:h, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[h - rheight:h, j:j + rwidth]

    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

def test_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform_ = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform_(rgb)
    if structure is not None:
        structure = transform_(structure)
    if sparse is not None:
        sparse = transform_(sparse)
    if gt is not None:
        gt = transform_(gt)
    if ipbasicgt is not None:
        ipbasicgt = transform_(ipbasicgt)
    if penetgt is not None:
        penetgt = transform_(penetgt)
    if s2dgt is not None:
        s2dgt = transform_(s2dgt)
    if position is not None:
        position = transform_(position)

    if not args.test_not_random_crop:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        # i = np.random.randint(0, h - rheight + 1)
        # j = np.random.randint(0, w - rwidth + 1)

        # i = int(torch.FloatTensor(1).uniform_(0, h - rheight + 1).item())
        j = int(torch.FloatTensor(1).uniform_(0, w - rwidth + 1).item())

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[h - rheight:h, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[h - rheight:h, j:j + rwidth]

        if structure is not None:
            if structure.ndim == 3:
                structure = structure[h - rheight:h, j:j + rwidth, :]
            elif structure.ndim == 2:
                structure = structure[h - rheight:h, j:j + rwidth]

        if sparse is not None:
            if sparse.ndim == 3:
                sparse = sparse[h - rheight:h, j:j + rwidth, :]
            elif sparse.ndim == 2:
                sparse = sparse[h - rheight:h, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[h - rheight:h, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[h - rheight:h, j:j + rwidth]

        if ipbasicgt is not None:
            if ipbasicgt.ndim == 3:
                ipbasicgt = ipbasicgt[h - rheight:h, j:j + rwidth, :]
            elif ipbasicgt.ndim == 2:
                ipbasicgt = ipbasicgt[h - rheight:h, j:j + rwidth]

        if penetgt is not None:
            if penetgt.ndim == 3:
                penetgt = penetgt[h - rheight:h, j:j + rwidth, :]
            elif penetgt.ndim == 2:
                penetgt = penetgt[h - rheight:h, j:j + rwidth]

        if s2dgt is not None:
            if s2dgt.ndim == 3:
                s2dgt = s2dgt[h - rheight:h, j:j + rwidth, :]
            elif s2dgt.ndim == 2:
                s2dgt = s2dgt[h - rheight:h, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[h - rheight:h, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[h - rheight:h, j:j + rwidth]

    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

def no_transform(sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position, args):
    return sparse, gt, ipbasicgt, penetgt, s2dgt, rgb, structure, position

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()
