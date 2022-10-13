# ------------------------------------------------------------
# dataloader for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image
# from random import choice
import matplotlib
import matplotlib.cm
import collections

import torch

def get_sparse_depth_prop(dep, prop):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    num_sample = int(num_idx * prop)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp

def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13
    K[1, 2] = K[1, 2] - 11.5
    return K

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.

    # FIXME
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def npytorgb(arr, cmap="jet"):
    """
    :param cmap:
    :param arr: arr with two dim
    :return:
    """
    assert len(arr.shape) == 2, 'the dim of arr should be 2'

    arr = (arr - np.min(arr)) / np.ptp(arr)
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    img = Image.fromarray(np.uint8(cm(arr) * 255))
    rgb = img.convert('RGB')

    return rgb

def s2r_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    npy_file = np.load(filename)
    rgb_structure = npytorgb(npy_file, 'viridis')
    rgb_structure_npy = np.array(rgb_structure, dtype='uint8')  # in the range [0,255]
    return rgb_structure_npy

# def drop_depth_measurements(depth, prob_keep):
#     mask = np.random.binomial(1, prob_keep, depth.shape)
#     depth *= mask
#     return depth

def handle_gray(rgb):
    if rgb is None:
        return None, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        return rgb, img

# def get_rgb_near(path, args):
#     assert path is not None, "path is None"
#
#     def extract_frame_id(filename):
#         head, tail = os.path.split(filename)
#         number_string = tail[0:tail.find('.')]
#         number = int(number_string)
#         return head, number
#
#     def get_nearby_filename(filename, new_id):
#         head, _ = os.path.split(filename)
#         new_filename = os.path.join(head, '%010d.png' % new_id)
#         return new_filename
#
#     head, number = extract_frame_id(path)
#     count = 0
#     max_frame_diff = 3
#     candidates = [
#         i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
#         if i - max_frame_diff != 0
#     ]
#     while True:
#         random_offset = choice(candidates)
#         path_near = get_nearby_filename(path, number + random_offset)
#         if os.path.exists(path_near):
#             break
#         assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_near)
#
#     return rgb_read(path_near)


# outlier removal for 64 lines lidar
# if you want to more general outlier removal algorithm
# please refer to https://github.com/placeforyiming/RAL_Non-Learning_DepthCompletion
# def outlier_removal(lidar):
#     # output 2 dimension image
#     # DIAMOND_KERNEL_7 = np.asarray(
#     #     [
#     #         [0, 0, 0, 1, 0, 0, 0],
#     #         [0, 0, 1, 1, 1, 0, 0],
#     #         [0, 1, 1, 1, 1, 1, 0],
#     #         [1, 1, 1, 1, 1, 1, 1],
#     #         [0, 1, 1, 1, 1, 1, 0],
#     #         [0, 0, 1, 1, 1, 0, 0],
#     #         [0, 0, 0, 1, 0, 0, 0],
#     #     ], dtype=np.uint8)
#     sparse_lidar = np.squeeze(lidar)
#     valid_pixels = (sparse_lidar > 0.1).astype(np.float)
# 
#     lidar_sum = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
#     lidar_count = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)
# 
#     lidar_aveg = lidar_sum / (lidar_count + 0.00001)
#     potential_outliers = ((sparse_lidar - lidar_aveg) > 1.0).astype(np.float)
# 
#     lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)
#     lidar_cleared = np.expand_dims(lidar_cleared, -1)
# 
#     return lidar_cleared

def outlier_removal(lidar):
    # output 2 dimension image
    # DIAMOND_KERNEL_7 = np.asarray(
    #     [
    #         [0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1, 1, 1],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0],
    #     ], dtype=np.uint8)
    sparse_lidar = np.squeeze(lidar)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)

    lidar_sum_7 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
    lidar_count_7 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)

    lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
    potential_outliers_7 = ((sparse_lidar - lidar_aveg_7) > 1.0).astype(np.float)
    
    lidar_sum_9 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_9)
    lidar_count_9 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_9)

    lidar_aveg_9 = lidar_sum_9 / (lidar_count_9 + 0.00001)
    potential_outliers_9 = ((sparse_lidar - lidar_aveg_9) > 1.0).astype(np.float)

    lidar_sum_13 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_13)
    lidar_count_13 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_13)

    lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
    potential_outliers_13 = ((sparse_lidar - lidar_aveg_13) > 1.0).astype(np.float)

    potential_outliers = potential_outliers_7 + potential_outliers_9 + potential_outliers_13
    lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)
    lidar_cleared = np.expand_dims(lidar_cleared, -1)

    return lidar_cleared

def mixed_gts(dense_gt, sparse_gt):
    assert dense_gt.shape == sparse_gt.shape
    valid_mask = sparse_gt > 0
    dense_gt[valid_mask] = sparse_gt[valid_mask]
    return dense_gt


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    depth_map = np.squeeze(depth_map, axis=-1)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map = np.expand_dims(depth_map, -1)

    return depth_map

def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False
                       ):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """
    depth_map = np.squeeze(depth_map, axis=-1)

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    # TODO
    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    depths_out = np.expand_dims(depths_out, -1)

    return depths_out

