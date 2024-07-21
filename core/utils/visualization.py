import os.path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import wandb
from pathlib import Path


def disp_map(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def pseudoColorMap(arr, vmin=None, vmax=None, cmap=None, kitti_style=False):
    """
    :param arr: one channel array
    :param vmin: Lower limit of truncation
    :param vmax: Upper limit of truncation
    :param cmap:colormap type
    :param kitti_style: whether use kitti colormap
    :return: rgb perceptual colormap
    """
    if kitti_style:
        h, w = arr.shape
        arr = np.clip(arr, vmin, vmax)
        arr = arr / vmax
        rgb = disp_map(arr.reshape(-1, 1)).reshape(h, w, 3)
        rgb = np.uint8(255 * rgb)
    else:
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_clim(vmin, vmax)
        rgba = sm.to_rgba(arr, bytes=True)
        rgb = rgba[:, :, :3]
    return rgb


def logFeatureMap(inputs, logname, wandb, vmin=None, vmax=None, cmap=None, kitti_style=False, local_save=False, name=''):
    """
    :param inputs: tensor like [N,C,H,W]
    :param wandb: wandb handle
    :return:
    """
    if len(inputs.shape) == 4:
        N, C, H, W = inputs.shape
        inputs = inputs.detach().cpu().numpy()
        log_dict = dict()
        # 为了不占用太多的空间，只使用第一个instance的各个channel进行可视化
        for j in range(C):
            slices = inputs[0, j, :, :]
            slicesMap = pseudoColorMap(slices, vmin, vmax, cmap)
            log_dict[logname + "_" + str(j)] = wandb.Image(slicesMap)
        wandb.log(log_dict, commit=False)
    elif len(inputs.shape) == 3:
        N, H, W = inputs.shape
        inputs = inputs.detach().cpu().numpy()
        log_dict = dict()
        # only use the first instance
        slices = inputs[0, :, :]
        slicesMap = pseudoColorMap(slices, vmin, vmax, cmap, kitti_style=kitti_style)
        if local_save:
            pt = "./localSave/" + logname + '/' + name + '.png'
            Path(os.path.split(pt)[0]).mkdir(exist_ok=True, parents=True)
            img = Image.fromarray(slicesMap)
            img.save(pt, format='png')
        else:
            log_dict[logname] = wandb.Image(slicesMap)
            wandb.log(log_dict, commit=False)


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def logErrorMap(disp_pr, disp_gt, log_name, wandb, abs_thres=3., rel_thres=0.05, dilate_radius=1, local_save=False, name='',commit=False):
    D_gt_np = disp_gt.detach().cpu().numpy()[:1, :, :]
    D_est_np = disp_pr.detach().cpu().numpy()[:1, :, :]
    B, H, W = D_gt_np.shape
    # valid mask
    mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]

    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    # print(error_image[0])
    if local_save:
        pt = "./localSave/" + log_name + '/' + name + '.png'
        Path(os.path.split(pt)[0]).mkdir(exist_ok=True, parents=True)
        img = Image.fromarray(np.uint8(255 * error_image[0]))
        img.save(pt, format='png')
    else:
        log_dict = dict()
        log_dict[log_name] = wandb.Image(error_image)
        wandb.log(log_dict, commit=commit)

def save_ply(points, colors, filename):
    assert points.shape == colors.shape, "Shape mismatch between points and colors"
    num_points = points.shape[0]

    # Write PLY header
    ply_header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).format(num_points)

    # Combine points and colors into a single array
    points_colors = np.hstack((points, colors))

    # Save points and colors to PLY file
    with open(filename, 'w') as ply_file:
        ply_file.write(ply_header)
        np.savetxt(ply_file, points_colors, fmt="%f %f %f %d %d %d")



if __name__ == "__main__":
    # Example usage:
    # Assuming you have your points and colors tensors
    # points_tensor = np.random.rand(100, 3)  # Replace with your actual points tensor
    # colors_tensor = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)  # Replace with your actual colors tensor
    from frame_utils import readDepthTartanAir
    from geo_utils import pixel2point
    import skimage
    import torch
    K = torch.from_numpy(np.array([[320.0, 0, 320.0],
                                   [0, 320.0, 240.0],
                                   [0, 0, 1]])).float()
    K_inv = torch.linalg.inv(K)[None].cuda()  # N,3,3

    # K_inv = K_inv.t()
    color = skimage.io.imread('/data/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/image_left/000003_left.png').reshape(-1,3)
    depth = readDepthTartanAir('/data/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/depth_left/000003_left_depth.npy')
    depth = torch.from_numpy(depth)[None, None].cuda()
    point = pixel2point(depth,K_inv)[0].permute(1,2,0)  # 3,h,w
    point = np.array(point.cpu()).reshape(-1,3)  # h,w,3

    # Save PLY file
    save_ply(point, color, "output_cloud.ply")
