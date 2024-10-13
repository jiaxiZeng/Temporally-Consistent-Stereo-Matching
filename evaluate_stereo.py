from __future__ import print_function, division
import sys

sys.path.append('core')
import os
import wandb
import argparse
import time
import skimage
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.tc_stereo import TCStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from core.utils.frame_utils import readDispTartanAir, read_gen
import cv2
import pykitti
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


@torch.no_grad()
def validate_tartanair(args, model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    # test set
    keyword_list = []
    scene_list = ['abandonedfactory', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'office', 'office2',
                  'oldtown', 'soulcity']  # ablation study
    part_list = ['P002', 'P007', 'P003', 'P006', 'P001', 'P042', 'P006', 'P004', 'P006', 'P008']

    for i, (s, p) in enumerate(zip(scene_list, part_list)):
        keyword_list.append(os.path.join(s, 'Easy', p))
        keyword_list.append(os.path.join(s, 'Hard', p))

    val_dataset = datasets.TartanAir(aug_params, root='datasets', scene_list=scene_list, test_keywords=keyword_list,
                                     is_test=True, mode='temporal', load_flow=False)

    # camera parameters
    K = np.array([[320.0, 0, 320.0],
                  [0, 320.0, 240.0],
                  [0, 0, 1]])
    K_raw = torch.from_numpy(K).float().cuda(args.device)[None]
    baseline = torch.tensor(0.25).float().cuda(args.device)[None]

    # Evaluate Metrics list
    out_list, out3_list, epe_list = [], [], []

    # load function
    def load(args, image1, image2, disp_gt, T):
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

        T = T[None].cuda(args.device)
        image1 = image1[None].cuda(args.device)
        image2 = image2[None].cuda(args.device)
        disp_gt = disp_gt[None].cuda(args.device)  # 1,1,h,w
        return image1, image2, disp_gt, T

    # Testing
    for val_id in tqdm(range(len(val_dataset))):
        image1_list, image2_list, flow_gt_list, pose_list = val_dataset[val_id]
        # temporal parameters
        params = dict()
        flow_q = None
        fmap1 = None
        previous_T = None
        net_list = None

        for (image1, image2, disp_gt, T) in tqdm(zip(image1_list, image2_list, flow_gt_list, pose_list)):
            # load
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

            # epe evaluation
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.sum((disp_pr.squeeze(0) - disp_gt.squeeze(0)) ** 2, dim=0).sqrt()

            epe = epe.flatten()
            val = (disp_gt.squeeze(0).abs().flatten() < 192)
            if (val == False).all():
                continue
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

    print("Validation TartanAir: EPE %f, D1 %f, D3 %f" % (epe, d1, d3))
    return {'TartanAir-epe': epe, 'TartanAir-d1': d1, 'TartanAir-d3': d3}


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
    parser.add_argument('--init_thres', type=float, default=0.5, help="the threshold gap of contrastive loss for cost volume.")
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
