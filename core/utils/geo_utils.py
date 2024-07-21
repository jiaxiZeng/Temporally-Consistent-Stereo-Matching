import torch
import torch.nn.functional as F
from core.utils.utils import coords_grid
from core.utils.splatting.softsplat import softsplat


def disp2depth(disp, baseline, fx):
    '''
    :param disp: N,1,H,W
    :param baseline: N,1
    :param fx: N
    :return: depth: N,1,H,W
    '''
    assert not torch.isnan(disp).any() and not torch.isinf(disp).any()
    assert (disp >= 0).all()
    return baseline.view(-1, 1, 1, 1) * fx.view(-1, 1, 1, 1) / torch.clip(disp, min=0.001)


def depth2disp(depth, baseline, fx):
    '''
    :param depth: N,1,H,W
    :param baseline:
    :param fx: N
    :return: disp: N,1,H,W
    '''
    assert not torch.isnan(depth).any() and not torch.isinf(depth).any()
    disp = baseline.view(-1, 1, 1, 1) * fx.view(-1, 1, 1, 1) / depth
    disp = torch.where(torch.isnan(disp) | torch.isinf(disp), -torch.ones_like(disp), disp)
    return disp


def pixel2point(depth, K_inv):
    '''
    :param depth: [N,1,H,W]
    :param K_inv: [N,3,3]
    :return: point: [N,3,H,W]
    '''
    N, _, H, W = depth.shape
    img_coord = coords_grid(N, H, W).to(depth.device)  # N,2,H,W
    img_coord = torch.cat((img_coord, torch.ones_like(depth)), dim=1)  # N,3,H,W
    point = depth.view(N, 1, -1) * torch.matmul(K_inv, img_coord.view(N, 3, -1))  # N,3,H*W
    return point.reshape(N, 3, H, W)  # N,3,H,W


def point2pixel(point, depth, K):
    '''
    :param point: [N,3,H,W]
    :param depth: [N,1,H,W]
    :param K: [N,3,3]
    :return: pixel_coord: [N,2,H,W]
    '''
    assert not torch.isnan(point).any() and not torch.isinf(point).any()
    assert not torch.isnan(depth).any() and not torch.isinf(depth).any()
    N, _, H, W = point.shape
    pixel_coord = torch.matmul(K, point.view(N, 3, -1)) / depth.view(N, 1, -1)
    pixel_coord = torch.where(torch.isnan(pixel_coord) | torch.isinf(pixel_coord), -torch.ones_like(pixel_coord), pixel_coord)
    return pixel_coord[:, :2].reshape(N, 2, H, W)


def disp2point(disp, baseline, K, K_inv):
    '''
    project the disparity map to the point coordinate
    :param disp: [N,1,H,W]
    :param baseline:
    :param K: intrinsic matrix [N,3,3]
    :return: point_coord: [N,3,H,W]
    '''
    depth = disp2depth(disp, baseline, fx=K[:, 0, 0])
    point_coord = pixel2point(depth, K_inv)
    return point_coord


def disp2disp_grad_candidates(disp, level=1):
    '''
    :param disp: N,1,H,W
    :param level: int
    :return: disp_grad_candidates: N, 2, 8*level, H, W
    '''
    N, _, H, W = disp.shape
    # grad kernel
    kernel = torch.zeros((8, 1, 3, 3))
    kernel[:, :, 1, 1] = -1
    vus = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    for i, vu in enumerate(vus):
        v, u = vu
        kernel[i, :, v, u] = 1
    kernel = kernel.to(disp.device)

    grad_candidates = []
    for i in range(level):  # multi-level gradient
        disp_pad = F.pad(disp, pad=(1 + i, 1 + i, 1 + i, 1 + i)).contiguous()
        img_coord = coords_grid(N, H + 2 + 2 * i, W + 2 + 2 * i).to(disp.device)  # x,y
        coord_disp = torch.cat((img_coord, disp_pad), dim=1).reshape(-1, 1, H + 2 + 2 * i, W + 2 + 2 * i).repeat(1, 8, 1, 1)  # n*3,1,8,h,w
        grads = F.conv2d(coord_disp, kernel, padding=0, groups=8, dilation=(i + 1))  # n*3,1,8,h,w
        grad_candidates.append(grads.reshape(N, 3, 8, H, W))  # n,3,8,h,w
    grads = torch.cat(grad_candidates, dim=2)
    grads_roll = torch.roll(grads, shifts=-2, dims=2)

    disp_grad_candidates = torch.cross(grads, grads_roll, dim=1)
    disp_grad_candidates = -disp_grad_candidates[:, :2] / disp_grad_candidates[:, 2:]
    return disp_grad_candidates


def disp2disp_normal_xy(disp):
    '''
    :param disp: N,1,H,W
    :return: disp_normal: N,3,H,W
    '''
    disp_grad, edge_mask = disp2disp_gradient_xy(disp)
    disp_normal = torch.cat([disp_grad, -torch.ones_like(disp_grad[:, :1])], dim=1)
    disp_normal = F.normalize(disp_normal, dim=1)
    return disp_normal, edge_mask


def disp2disp_gradient_xy(disp):
    N, _, H, W = disp.shape
    disp_pad = F.pad(disp, pad=(1, 1, 1, 1), mode='replicate')

    # grad kernel
    kernel = torch.zeros((2, 1, 3, 3))
    kernel[:, :, 1, 1] = -1
    vus = [(1, 2), (2, 1)]
    for i, vu in enumerate(vus):
        v, u = vu
        kernel[i, :, v, u] = 1

    grads = F.conv2d(disp_pad.repeat(1, 2, 1, 1), kernel.to(disp.device), padding=0, groups=2)  # n,2,h,w
    grads_x = grads[:, :1]
    grads_y = grads[:, 1:]
    edge_mask = (grads_x.abs() < 5) & (grads_y.abs() < 5)  # mask out big gradient

    return grads, edge_mask


def relative_transform(x, relative_T):
    '''
    Pose transform from x to x' given relative transformation.
    :param x: [N,3,H,W]
    :param relative_T: [N,4,4]
    :return: x' [N,3,H,W]
    '''
    N, _, H, W = x.shape
    x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)  # N,4,H,W
    x_transform = torch.matmul(relative_T, x.view(N, 4, -1))  # N,4,H*W
    return (x_transform[:, :3]).reshape(N, 3, H, W)  # N,3,H,W


def cal_relative_transformation(T1, T2):
    '''
    relative transformation from T1 to T2
    :param T1: world2cam pose1 (homogeneous 4x4 matrix)
    :param T2: world2cam pose2 (homogeneous 4x4 matrix)
    :return: relative transformation form previous pose to current pose
    '''
    return torch.matmul(T2, torch.linalg.inv(T1))


def warp(disp, fmap, relative_T, K, K_inv, baseline):
    '''
    warp the disparity map and feature map to the current frame
    :param disp: [N,1,H,W]
    :param fmap: [N,C,H,W]
    :param relative_T: [N,4,4]
    :param K: [N,3,3]
    :param K_inv: [N,3,3]
    :param baseline: [N,1]
    :return: current_disp: [N,1,H,W], current_fmap: [N,C,H,W], warped_mask: [N,1,H,W]
    '''
    N, _, H, W = disp.shape
    # fx
    fx = K[:, 0, 0]
    # disparity to depth
    depth = disp2depth(disp, baseline, fx)

    # 2D pixel project to 3D point
    previous_P = pixel2point(depth, K_inv)  # N,2,H,W

    # relative pose transformation
    current_P = relative_transform(previous_P, relative_T)

    # current depth
    current_depth = current_P[:, -1:, :, :]  # B,1,H,W

    # current disp
    current_disp = depth2disp(current_depth, baseline, fx)
    valid_mask = ((current_disp > 0) & (current_disp < W))

    # coord
    coords0 = coords_grid(N, H, W).to(disp.device)  # [B,2,H,W]
    current_coords = point2pixel(current_P, current_depth, K)  # [B,2,H,W]

    forward_flow = current_coords - coords0
    metric = (current_disp - current_disp.mean()).clamp(-50, 50)
    # disp&fmap concat and warp
    feats = torch.cat((current_disp, fmap), dim=1)
    feats, warped_mask = softsplat(feats, forward_flow, metric, 'soft-clipeps', valid_mask.float())
    current_disp, current_fmap = feats[:, :1], feats[:, 1:]
    return current_disp.detach(), current_fmap.detach(), warped_mask.detach()


def get_backward_grid(disp, relative_T, K, K_inv, baseline):
    '''
        map the current frame to the previous frame
        :param disp: [N,1,H,W]
        :param relative_T: [N,4,4]
        :param K: [N,3,3]
        :param K_inv: [N,3,3]
        :param baseline: [N,1]
        :return: previous_coords: [N,2,H,W]
    '''
    assert not torch.isnan(disp).any()

    assert not torch.isnan(relative_T).any()
    N, _, H, W = disp.shape
    # fx
    fx = K[:, 0, 0]
    disp = torch.clip(disp, 0.01)
    # disparity to depth
    depth = disp2depth(disp, baseline, fx)
    # assert (disp > 0).all(), [disp,depth]
    assert not torch.isnan(depth).any() and not torch.isinf(depth).any(), [depth]
    # 2D pixel project to 3D point
    P = pixel2point(depth, K_inv)  # N,2,H,W
    assert not torch.isnan(P).any(), [P]
    # relative pose transformation
    previous_P = relative_transform(P, relative_T)
    assert not torch.isnan(previous_P).any(), [previous_P, relative_T]
    # previous depth
    previous_depth = previous_P[:, -1:, :, :]  # B,1,H,W
    valid_mask = (previous_depth > 0)

    # coord
    previous_coords = point2pixel(previous_P, previous_depth, K)  # [B,2,H,W]
    previous_coords = torch.where(valid_mask, previous_coords, -torch.ones_like(previous_coords))
    assert not torch.isnan(previous_coords).any() and not torch.isinf(previous_coords).any(), [previous_coords]
    return previous_coords
