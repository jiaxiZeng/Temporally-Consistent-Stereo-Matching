from __future__ import print_function, division
import sys

sys.path.append('core')
import cv2
import pykitti
import os
import wandb
import argparse
import time
import skimage
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from core.tc_stereo import TCStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder,bilinear_sampler
from core.utils.frame_utils import readDispTartanAir, read_gen, readFlowTartanAir
from core.utils.geo_utils import disp2point,depth2disp,cal_relative_transformation,relative_transform,coords_grid
from core.utils.visualization import pseudoColorMap


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def submit_kitti(args, model, iters=32, mixed_prec=False):
    """ Peform submission using the KITTI-2015 (seq test) split """
    model.eval()
    aug_params = {}
    submission = True
    imageset = 'kitti_seq/kitti2015_testings'
    P = 'P_rect_02'

    val_dataset = datasets.KITTI(aug_params,
                                 is_test=True,
                                 mode='temporal',
                                 image_set=imageset,
                                 index_by_scene=True,
                                 num_frames=11 if submission else 21)
    torch.backends.cudnn.benchmark = True
    params = dict()
    flow_q = None
    fmap1 = None
    previous_T = None
    net_list = None
    baseline = torch.tensor(0.54).float().cuda(args.device)[None]
    out_list, epe_list, elapsed_list = [], [], []

    def load(args, image1, image2, T):
        # load image & disparity
        image1 = read_gen(image1)
        image2 = read_gen(image2)
        image1 = np.array(image1)
        image2 = np.array(image2)
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        T = torch.from_numpy(T).float()
        T = T[None].cuda(args.device)
        image1 = image1[None].cuda(args.device)
        image2 = image2[None].cuda(args.device)
        return image1, image2, T

    for val_id in tqdm(range(len(val_dataset))):
        image1_list, image2_list, scene_path, pose_list = val_dataset[val_id]
        Pr2 = pykitti.utils.read_calib_file(os.path.join(scene_path, scene_path.split('/')[-1] + '.txt'))[P]
        K = np.array([[Pr2[0], 0, Pr2[2]],
                      [0, Pr2[5], Pr2[6]],
                      [0, 0, 1]])
        K_raw = torch.from_numpy(K).float().cuda(args.device)[None]
        for frame_ind, (image1, image2, T) in tqdm(enumerate(zip(image1_list, image2_list, pose_list))):
            image1, image2, T = load(args, image1, image2, T)
            padder = InputPadder(image1.shape, divis_by=32)
            imgs, K = padder.pad(image1, image2, K=K_raw)
            image1, image2 = imgs
            params.update({'K': K,
                           'T': T,
                           'previous_T': previous_T,
                           'last_disp': flow_q,
                           'last_net_list': net_list,
                           'fmap1': fmap1,
                           'baseline': baseline})
            with autocast(enabled=mixed_prec):
                start = time.time()
                testing_output = model(image1, image2, iters=iters, test_mode=True, params=params if (flow_q is not None) and args.temporal else None)
                end = time.time()
            if val_id > 50 and frame_ind > 6:
                elapsed_list.append(end - start)
            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = T
            disp_pr, K = padder.unpad(disp_pr, K)  # 1,1,h,w
            # save
            if submission:
                if frame_ind == 10:
                    disp_pr = disp_pr.squeeze(0).detach().cpu().numpy()  # 1,h,w
                    submit_dir = os.path.join('./kitti_15_seq_out', 'disp_0')
                    os.makedirs(submit_dir, exist_ok=True)
                    skimage.io.imsave(os.path.join(submit_dir, scene_path.split('/')[-1] + '_10.png'), (disp_pr * 256).astype('uint16'))
            else:  # output as rgb video visualization
                disp_pr = disp_pr[0, 0].detach().cpu().numpy()  # 1,h,w
                disp_pr = pseudoColorMap(disp_pr, vmin=0, vmax=96, kitti_style=True)
                if frame_ind == 0:
                    video_dir = os.path.join('./kitti_15_seq_out', 'video')
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, scene_path.split('/')[-1] + '.avi')
                    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 2, (disp_pr.shape[1], disp_pr.shape[0]))  # 2fps
                video.write(disp_pr)
        if not submission:
            video.release()
    avg_runtime = np.mean(elapsed_list)
    print(f"Submission KITTI: {format(1 / (avg_runtime + 1e-5), '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-fps': 1 / (avg_runtime + 1e-5)}


def track_disp_error(disp_gt_tar, disp_pred_src, flow_tar2src, baseline, K, T_tar, T_src, disp_mask):
    """ Track the disparity error from the source frame to the target frame """
    K_inv = torch.linalg.inv(K)
    # Disparity to 3d Points
    points_pred = disp2point(disp_pred_src, baseline, K, K_inv)
    # source to target transformation
    T_src2tar = cal_relative_transformation(T_src, T_tar)
    points_pred_tar = relative_transform(points_pred, T_src2tar)
    # source disparity to target disparity (numeral)
    disp_pred_tar = depth2disp(points_pred_tar[:, -1:], baseline, K[:, 0, 0])
    neg_mask = (disp_pred_tar < 0)  # negative disparity
    # source disparity to target disparity (alignment)
    warp_grid = coords_grid(disp_gt_tar.shape[0],disp_gt_tar.shape[2],disp_gt_tar.shape[3]).to(flow_tar2src.device) + flow_tar2src
    disp_mask = (disp_mask.bool() & ~neg_mask).float()
    aligned_disp = bilinear_sampler(torch.cat((disp_mask, disp_pred_tar), dim=1), warp_grid.permute(0,2,3,1), align_corners=True)
    disp_mask, disp_pred_tar = torch.split(aligned_disp, 1, dim=1)
    disp_mask = (disp_mask == 1)
    # Disparity Error
    disp_error = torch.abs(disp_pred_tar - disp_gt_tar)

    return disp_error, disp_pred_tar, disp_mask

def evaluate_temporal_consistency(disp_gt_tar, disp_gt_mask, disp_pred_tar, disp_preds_src, disp_src_masks, flows, baseline, K, T_tar, T_srcs=[], flow_masks=[]):
    """ Evaluate the temporal consistency of the disparity predictions """
    assert len(disp_preds_src) == len(flows) == len(T_srcs) == len(flow_masks) == len(disp_src_masks)
    flow = torch.zeros_like(flows[0])
    mask = disp_gt_mask
    epe_tar = torch.abs(disp_gt_tar-disp_pred_tar) # t
    step = torch.zeros_like(mask)
    abs_delta_disp_sum = torch.zeros_like(mask)
    Relu_delta_e_sum = torch.zeros_like(mask)
    d3_list = []
    Relu_delta_e_1_list = []  # epe t < epe t+1
    Relu_delta_e_3_list = []  # epe t < epe t+1
    Relu_delta_e_5_list = []  # epe t < epe t+1

    for iteri, (disp_pred, flow_i, T_src, flow_mask, disp_src_mask) in enumerate(zip(disp_preds_src, flows, T_srcs, flow_masks, disp_src_masks)):
        flow = flow + flow_i  # flow from target to source
        disp_error, disp_pred_src_2_tar, disp_scr2tar_mask = track_disp_error(disp_gt_tar, disp_pred, flow, baseline, K, T_tar, T_src, disp_src_mask.float())
        mask = mask.bool() & flow_mask.bool() & disp_scr2tar_mask.bool()
        if torch.count_nonzero(mask.float()) == 0:
            break
        step = step + mask.float()
        abs_delta_disp = torch.abs(disp_pred_src_2_tar-disp_pred_tar) * mask.float()  # t+1
        epe_diff = (disp_error - epe_tar * mask.float())
        abs_delta_disp_sum = abs_delta_disp_sum + abs_delta_disp
        Relu_delta_e_sum = Relu_delta_e_sum + F.relu(epe_diff)
        d3 = (abs_delta_disp > 3).float()
        d3_list.append(d3[mask])
        Relu_delta_e_1_list.append((epe_diff > 1).float()[mask])
        Relu_delta_e_3_list.append((epe_diff > 3).float()[mask])
        Relu_delta_e_5_list.append((epe_diff > 5).float()[mask])
        assert not torch.isnan(abs_delta_disp_sum).any() and not torch.isinf(abs_delta_disp_sum).any()
    abs_delta_disp_mean = abs_delta_disp_sum / torch.clip(step.float(), min=0.01)
    assert not torch.isnan(abs_delta_disp_mean).any() and not torch.isinf(abs_delta_disp_mean).any()
    abs_delta_disp_mean = abs_delta_disp_mean[step > 0].mean().item()
    Relu_delta_e_mean = Relu_delta_e_sum / torch.clip(step.float(), min=0.01)
    assert not torch.isnan(Relu_delta_e_mean).any() and not torch.isinf(Relu_delta_e_mean).any()
    Relu_delta_e_mean = Relu_delta_e_mean[step > 0].mean().item()
    d3 = torch.concatenate(d3_list)
    mask_rate = torch.numel(d3) / (disp_gt_tar.shape[2] * disp_gt_tar.shape[3] * len(disp_preds_src))
    d3 = d3.mean().cpu().item()
    Relu_delta_e_3 = torch.cat(Relu_delta_e_3_list).mean().cpu().item()
    Relu_delta_e_1 = torch.cat(Relu_delta_e_1_list).mean().cpu().item()
    Relu_delta_e_5 = torch.cat(Relu_delta_e_5_list).mean().cpu().item()
    Relu_delta_e_metrics = {
        'Relu_delta_e': Relu_delta_e_mean,
                    'Relu_delta_e_1': Relu_delta_e_1,
                    'Relu_delta_e_3': Relu_delta_e_3,
                    'Relu_delta_e_5': Relu_delta_e_5,
                    }
    return abs_delta_disp_mean, d3, mask_rate, Relu_delta_e_metrics





@torch.no_grad()
def validate_tartanair(args, model, iters=32, mixed_prec=False):
    """ Peform validation using the Tartanair split """
    model.eval()
    aug_params = {}
    # test set
    keyword_list = []
    scene_list = ['abandonedfactory', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'office', 'office2',
                  'oldtown', 'soulcity']   # ablation study
    part_list = ['P002', 'P007', 'P003', 'P006', 'P001', 'P042', 'P006', 'P004', 'P006', 'P002', 'P015', 'P008']

    for i, (s, p) in enumerate(zip(scene_list, part_list)):
        keyword_list.append(os.path.join(s, 'Easy', p))
        keyword_list.append(os.path.join(s, 'Hard', p))

    val_dataset = datasets.TartanAir(aug_params, root='datasets', scene_list=scene_list, test_keywords=keyword_list, is_test=True, mode='temporal', load_flow=True)

    # camera parameters
    K = np.array([[320.0, 0, 320.0],
                  [0, 320.0, 240.0],
                  [0, 0, 1]])
    K_raw = torch.from_numpy(K).float().cuda(args.device)[None]
    baseline = torch.tensor(0.25).float().cuda(args.device)[None]

    # Evaluate Metrics list
    out_list, epe_list, abs_delta_d_list = [], [], []
    out3_list, abs_delta_d3_list = [], []
    Relu_delta_e_5_list, Relu_delta_e_3_list, Relu_delta_e_1_list = [], [], []
    Relu_delta_e_list = []
    # load function
    def load(args, image1, image2, disp_gt, T, flow_gt):
        # load image & disparity
        image1 = read_gen(image1)
        image2 = read_gen(image2)
        image1 = np.array(image1)
        image2 = np.array(image2)
        disp_gt = readDispTartanAir(disp_gt)
        disp_gt = torch.from_numpy(np.array(disp_gt).astype(np.float32))[:1]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        T = torch.from_numpy(T).float()
        flow, flow_mask = readFlowTartanAir(flow_gt)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow_mask = torch.from_numpy(flow_mask)[None].float()

        T = T[None].cuda(args.device)
        image1 = image1[None].cuda(args.device)
        image2 = image2[None].cuda(args.device)
        flow = flow[None].cuda(args.device)  # 1,2,h,w
        flow_mask = flow_mask[None].cuda(args.device)  # 1,1,h,w
        disp_gt = disp_gt[None].cuda(args.device)  # 1,1,h,w
        return image1, image2, disp_gt, T, flow, flow_mask

    class queue:
        def __init__(self,k):
            self.k = k
            self.queue = [None]*k

        def push(self,x):
            self.queue.pop(0)
            self.queue.append(x)

        def get(self):
            return self.queue

    # Testing
    for val_id in tqdm(range(len(val_dataset))):
        image1_list, image2_list, flow_gt_list, pose_list, flow_list = val_dataset[val_id]
        # temporal parameters
        params = dict()
        flow_q = None
        fmap1 = None
        previous_T = None
        disp_grad_q = None
        grad_mask = None
        net_list = None
        # k frame queue
        k = args.temporal_window_size
        disp_gts_queue = queue(k+1)
        disp_preds_queue = queue(k+1)
        disp_src_mask_queue = queue(k+1)
        flows_queue = queue(k+1)
        flow_masks_queue = queue(k+1)
        T_srcs_queue = queue(k+1)

        for (image1, image2, disp_gt, T, flow_gt) in tqdm(zip(image1_list, image2_list, flow_gt_list, pose_list, flow_list)):
            # load
            image1, image2, disp_gt, T, flow_gt, flow_mask = load(args, image1, image2, disp_gt, T, flow_gt)
            padder = InputPadder(image1.shape, divis_by=32)
            imgs, K = padder.pad(image1, image2, K=K_raw)
            image1, image2 = imgs
            params.update({'K': K,
                           'T': T,
                           'previous_T': previous_T,
                           'last_disp': flow_q,
                           'last_net_list': net_list,
                           'fmap1': fmap1,
                           'disp_grad_q': disp_grad_q,
                           'grad_mask': grad_mask,
                           'baseline': baseline})

            with autocast(enabled=mixed_prec):
                testing_output = model(image1, image2, iters=iters, test_mode=True, params=params if (flow_q is not None) and args.temporal else None)

            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = T
            disp_pr, K = padder.unpad(disp_pr, K=K)

            # keep the temporal queue
            disp_gts_queue.push(disp_gt)
            disp_preds_queue.push(disp_pr)
            disp_src_mask_queue.push(disp_gt.abs() < 192)
            flows_queue.push(flow_gt)
            flow_masks_queue.push(flow_mask)
            T_srcs_queue.push(T)

            # temporal epe evaluation
            if disp_gts_queue.queue[0] is not None:
                abs_delta_d, abs_delta_d3, Tmask_rate, Relu_delta_e_metics  = evaluate_temporal_consistency(disp_gt_tar=disp_gts_queue.queue[0],
                                                                      disp_gt_mask=disp_src_mask_queue.queue[0],
                                                                      disp_pred_tar=disp_preds_queue.queue[0],
                                                                      disp_preds_src=disp_preds_queue.get()[1:],
                                                                      disp_src_masks=disp_src_mask_queue.get()[1:],
                                                                      flows=flows_queue.get()[:-1],
                                                                      baseline=baseline,
                                                                      K=K,
                                                                      T_tar=T_srcs_queue.queue[0],
                                                                      T_srcs=T_srcs_queue.get()[1:],
                                                                      flow_masks=flow_masks_queue.get()[:-1])
                if not np.isnan(abs_delta_d):
                    abs_delta_d_list.append(np.array([abs_delta_d*Tmask_rate, Tmask_rate]))
                    Relu_delta_e_list.append(np.array([Relu_delta_e_metics['Relu_delta_e']*Tmask_rate, Tmask_rate]))
                    abs_delta_d3_list.append(np.array([abs_delta_d3*Tmask_rate, Tmask_rate]))
                    Relu_delta_e_5_list.append(np.array([Relu_delta_e_metics['Relu_delta_e_5']*Tmask_rate, Tmask_rate]))
                    Relu_delta_e_3_list.append(np.array([Relu_delta_e_metics['Relu_delta_e_3']*Tmask_rate, Tmask_rate]))
                    Relu_delta_e_1_list.append(np.array([Relu_delta_e_metics['Relu_delta_e_1']*Tmask_rate, Tmask_rate]))

            # epe evaluation
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.sum((disp_pr.squeeze(0) - disp_gt.squeeze(0)) ** 2, dim=0).sqrt()

            epe = epe.flatten()
            val = (disp_gt.squeeze(0).abs().flatten() < 192)

            out = (epe > 1.0).float()[val].mean().cpu().item()
            out3 = (epe > 3.0).float()[val].mean().cpu().item()
            mask_rate = val.float().mean().cpu().item()
            epe_list.append(epe[val].mean().cpu().item())
            out_list.append(np.array([out*mask_rate, mask_rate]))
            out3_list.append(np.array([out3*mask_rate, mask_rate]))
    abs_delta_d_list = np.stack(abs_delta_d_list,axis=0)
    Relu_delta_e_list = np.stack(Relu_delta_e_list,axis=0)
    epe_list = np.array(epe_list)
    out_list = np.stack(out_list, axis=0)
    out3_list = np.stack(out3_list, axis=0)
    abs_delta_d3_list = np.stack(abs_delta_d3_list, axis=0)
    Relu_delta_e_5_list = np.stack(Relu_delta_e_5_list, axis=0)
    Relu_delta_e_3_list = np.stack(Relu_delta_e_3_list, axis=0)
    Relu_delta_e_1_list = np.stack(Relu_delta_e_1_list, axis=0)


    abs_delta_d = np.sum(abs_delta_d_list[:, 0]) / np.sum(abs_delta_d_list[:, 1])
    Relu_delta_e = np.sum(Relu_delta_e_list[:, 0]) / np.sum(Relu_delta_e_list[:, 1])
    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list[:, 0]) / np.mean(out_list[:, 1])
    d3 = 100 * np.mean(out3_list[:, 0]) / np.mean(out3_list[:, 1])
    abs_delta_d3 = 100 * np.sum(abs_delta_d3_list[:, 0]) / np.sum(abs_delta_d3_list[:, 1])
    Relu_delta_e_3 = 100 * np.sum(Relu_delta_e_3_list[:, 0]) / np.sum(Relu_delta_e_3_list[:, 1])
    Relu_delta_e_1 = 100 * np.sum(Relu_delta_e_1_list[:, 0]) / np.sum(Relu_delta_e_1_list[:, 1])
    Relu_delta_e_5 = 100 * np.sum(Relu_delta_e_5_list[:, 0]) / np.sum(Relu_delta_e_5_list[:, 1])

    print("Validation TartanAir: EPE %f, abs_delta_d %f, Relu_delta_e %f, D1 %f, D3 %f, abs_delta_d3 %f, Relu_delta_e_5 %f, Relu_delta_e_3 %f, Relu_delta_e_1 %f"
          % (epe, abs_delta_d, Relu_delta_e, d1, d3, abs_delta_d3, Relu_delta_e_5, Relu_delta_e_3, Relu_delta_e_1))
    return {'TartanAir-epe': epe,
            'TartanAir-d1': d1,
            'TartanAir-d3': d3,
            'TartanAir-abs_delta_d': abs_delta_d,
            'TartanAir-abs_delta_d3': abs_delta_d3,
            'TartanAir-Relu_delta_e_5': Relu_delta_e_5,
            'TartanAir-Relu_delta_e_3': Relu_delta_e_3,
            'TartanAir-Relu_delta_e_1': Relu_delta_e_1,
            'TartanAir-Relu_delta_e': Relu_delta_e,
            }

@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_temporal_things(args, model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_cleanpass', things_test=True, mode='temporal')

    def load(args, image1, image2, disp_gt, T):
        # load image & disparity
        image1 = read_gen(image1)
        image2 = read_gen(image2)
        image1 = np.array(image1)
        image2 = np.array(image2)
        disp_gt = read_gen(disp_gt)
        disp_gt = torch.from_numpy(np.array(disp_gt).astype(np.float32))
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        T = torch.from_numpy(T).float()

        T = T[None].cuda(args.device)
        image1 = image1[None].cuda(args.device)
        image2 = image2[None].cuda(args.device)
        disp_gt = disp_gt[None].cuda(args.device)  # 1,1,h,w
        return image1, image2, disp_gt, T

    out_list, out3_list, epe_list = [], [], []
    K = np.array([[1050., 0., 479.5],
                  [0., 1050., 269.5],
                  [0.0, 0.0, 1.0]])
    K_raw = torch.from_numpy(K).float().cuda(args.device)[None]
    baseline = torch.tensor(1.).float().cuda(args.device)[None]
    for val_id in tqdm(range(len(val_dataset))):
        image1_list, image2_list, flow_gt_list, pose_list = val_dataset[val_id]
        params = dict()
        flow_q = None
        fmap1 = None
        previous_T = None
        net_list = None
        for j, (image1, image2, disp_gt, T) in tqdm(enumerate(zip(image1_list, image2_list, flow_gt_list, pose_list))):
            image1, image2, disp_gt, T = load(args, image1, image2, disp_gt, T)
            padder = InputPadder(image1.shape, divis_by=32)
            imgs, K = padder.pad(image1, image2, K=K_raw)
            image1, image2 = imgs
            params.update({'K': K,
                           'T': T,
                           'previous_T': previous_T,
                           'last_disp': flow_q,
                           'last_net_list': net_list,
                           'fmap1': fmap1,
                           'baseline': baseline})

            with autocast(enabled=mixed_prec):
                testing_output = model(image1, image2, iters=iters, test_mode=True, params=params if (flow_q is not None) and args.temporal else None)

            disp_pr = -testing_output['flow']
            flow_q = testing_output['flow_q']
            net_list = testing_output['net_list']
            fmap1 = testing_output['fmap1']
            previous_T = T
            disp_pr, K = padder.unpad(disp_pr, K=K)
            val = (disp_gt.squeeze(0).abs().flatten() < 192)
            if (val == False).all():
                continue
            epe = torch.sum((disp_pr.squeeze(0) - disp_gt.squeeze(0)) ** 2, dim=0).sqrt()

            epe = epe.flatten()
            out = (epe > 1.0).float()[val].mean().cpu().item()
            out3 = (epe > 3.0).float()[val].mean().cpu().item()
            mask_rate = val.float().mean().cpu().item()
            epe_list.append(epe[val].mean().cpu().item())
            out_list.append(np.array([out * mask_rate, mask_rate]))
            out3_list.append(np.array([out3 * mask_rate, mask_rate]))

    epe_list = np.array(epe_list)
    out_list = np.stack(out_list, axis=0)
    out3_list = np.stack(out3_list, axis=0)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list[:, 0]) / np.mean(out_list[:, 1])
    d3 = 100 * np.mean(out3_list[:, 0]) / np.mean(out3_list[:, 1])
    print("Validation FlyingThings: EPE %f, D1 %f, D3 %f" % (epe, d1, d3))

    return {'things-epe': epe, 'things-d1': d1, 'things-d3': d3}


if __name__ == '__main__':
    import os
    import psutil
    pid = os.getpid()
    process = psutil.Process(pid)
    process.nice(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True,
                        choices=["kitti", "things", "TartanAir"])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--uncertainty_threshold', default=0.5, type=float, help='the threshold of uncertainty')
    parser.add_argument('--visualize', action='store_true', help='visualize the results')
    parser.add_argument('--device', default=0, type=int, help='the device id')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--temporal', action='store_true', help="temporal mode")  # TODO: MODEL temporal mode
    parser.add_argument('--temporal_window_size', type=int, default=1, help="temporal consistency metric window size")
    parser.add_argument('--init_thres', type=float, default=0.5, help="the threshold gap of contrastive loss for cost volume.")
    args = parser.parse_args()

    # if args.visualize:
    wandb.init(
        job_type="test",
        project="vis",
        entity="zengjiaxi"
    )
    # add the args to wandb
    wandb.config.update(args)
    model = TCStereo(args)
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint['model'], strict=True)
        logging.info(f"Done loading checkpoint")
    model = torch.nn.DataParallel(model, device_ids=[args.device])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    model.cuda(args.device)
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = False

    if args.dataset == 'kitti':
        submit_kitti(args, model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_temporal_things(args, model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'TartanAir':
        validate_tartanair(args, model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
