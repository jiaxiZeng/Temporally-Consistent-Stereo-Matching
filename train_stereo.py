from __future__ import print_function, division
import wandb
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
import random
import torch.optim as optim
from core.tc_stereo import TCStereo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluate_stereo import count_parameters, validate_tartanair, validate_temporal_things
from core.utils.geo_utils import disp2disp_gradient_xy,disp2disp_normal_xy
from core.utils.utils import MedianPool2d
import core.stereo_datasets as datasets
import torch.nn.functional as F
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def disp_grad_loss(disp_grad_preds, disp_grad_gt, valid, loss_weights, metric_name='grad_loss', scale=0.25, dense_gt=True):
    # gradient loss
    n_predictions = len(disp_grad_preds)
    assert n_predictions >= 1
    grad_loss = 0.0
    median_pool = MedianPool2d(kernel_size=int(1/scale), stride=int(1/scale), padding=0, same=False)
    disp_grad_gt = median_pool(disp_grad_gt)
    mask = (disp_grad_gt[:, :1] < 5) & (disp_grad_gt[:, 1:] < 5)
    if dense_gt:
        valid = (F.max_pool2d(valid.float(), 4, 4, 0)).bool() & mask
    else:
        valid = F.interpolate(valid.float(), scale_factor=scale, mode='bilinear', align_corners=True)
        valid = (valid==1) & mask

    for i in range(n_predictions):
        assert not torch.isnan(disp_grad_preds[i]).any() and not torch.isinf(disp_grad_preds[i]).any(), [i, disp_grad_preds[i]]
        disp_grad = disp_grad_preds[i]  # n,2,h,w
        i_loss = torch.mean((disp_grad - disp_grad_gt).abs(), dim=1, keepdim=True)  # N,2,H,W
        grad_loss += loss_weights[i] * i_loss[valid].mean()

    metrics = {
        metric_name: grad_loss.item()
    }
    return grad_loss, metrics


def disp_normal_loss(flow_q_preds, disp_norm_gt, valid, loss_weights, metric_name='norm_loss', scale=0.25, dense_gt=True):
    # disp_normal_loss
    n_predictions = len(flow_q_preds)
    assert n_predictions >= 1
    norm_loss = 0.0
    median_pool = MedianPool2d(kernel_size=4, stride=4, padding=0, same=False)

    disp_norm_gt = median_pool(disp_norm_gt)
    mask = (disp_norm_gt[:, :1]/disp_norm_gt[:, 2:] < 5) & (disp_norm_gt[:, 1:2]/disp_norm_gt[:, 2:] < 5)
    if dense_gt:
        valid = (F.max_pool2d(valid.float(), 4, 4, 0)).bool() & mask
    else:
        valid = F.interpolate(valid.float(), scale_factor=scale, mode='bilinear', align_corners=True)
        valid = (valid==1) & mask

    for i in range(n_predictions):
        flow_q, flow_q_refine = flow_q_preds[i]
        disp_norm, _ = disp2disp_normal_xy(-flow_q)  # n,3,h,w
        disp_refine_norm, _ = disp2disp_normal_xy(-flow_q_refine)  # n,3,h,w
        i_loss = 0.5*torch.mean((disp_norm - disp_norm_gt).abs(), dim=1, keepdim=True) + 0.5*(1-torch.sum(disp_norm*disp_norm_gt, dim=1, keepdim=True))  # N,1,H,W
        i_loss_refine = 0.5*torch.mean((disp_refine_norm - disp_norm_gt).abs(), dim=1, keepdim=True) + 0.5*(1-torch.sum(disp_refine_norm*disp_norm_gt, dim=1, keepdim=True))  # N,1,H,W
        norm_loss += loss_weights[i] * (i_loss[valid].mean() + 1.2*i_loss_refine[valid].mean())

    metrics = {
        metric_name: norm_loss.item()
    }
    return norm_loss, metrics


def sequence_loss(flow_mono, flow_init, flow_preds, flow_gt, valid, loss_weights):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()
    assert not torch.isnan(flow_init).any()
    assert not torch.isnan(flow_mono).any()
    flow_loss += 0.1*(flow_init - flow_gt).abs()[valid.bool()].mean()
    flow_loss += 0.1*(flow_mono - flow_gt).abs()[valid.bool()].mean()
    for i in range(n_predictions):
        flows, flows_refine = flow_preds[i]
        assert not torch.isnan(flows).any() and not torch.isinf(flows).any(), [i, flows]
        i_weight = loss_weights[i]
        i_loss = (flows - flow_gt).abs() + 1.2 * (flows_refine - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1][0] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    epe_refine = torch.sum((flow_preds[-1][1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    epe_init = torch.sum((flow_init - flow_gt) ** 2, dim=1).sqrt()
    epe_init = epe_init.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'epe_refine': epe_refine.mean().item(),
        'epe_init': epe_init.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        '1px_refine': (epe_refine < 1).float().mean().item(),
        '3px_refine': (epe_refine < 3).float().mean().item(),
        '5px_refine': (epe_refine < 5).float().mean().item(),
    }

    return flow_loss, metrics


def init_loss(cost_volume, flow_gt, valid, max_flow=700, k=1, scale=0.25, threshold=0.1):
    """ Loss function defined over sequence of flow predictions """
    assert not torch.isnan(cost_volume).any() and not torch.isinf(cost_volume).any(), [cost_volume]

    flow_gt = scale*F.interpolate(flow_gt, scale_factor=scale, mode='nearest')
    valid = F.interpolate(valid.float(), scale_factor=scale, mode='bilinear', align_corners=True)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid == 1) & (mag < max_flow*scale))

    def rho(d):  # ρ(d)
        d = torch.clip(d, 0, cost_volume.size(1) - 1)
        return torch.gather(cost_volume, dim=1, index=d)

    def phi(d):  # φ(d)
        df = torch.floor(d).long()
        d_sub_df = d - df
        return d_sub_df * rho(df + 1) + (1 - d_sub_df) * rho(df)

    disp_gt = - flow_gt
    index_gt = torch.arange(cost_volume.size(3)).view(1, 1, 1, -1).to(cost_volume.device)-disp_gt
    mask = (index_gt >= 0) & (index_gt <= cost_volume.size(1) - 1)
    mask = mask & valid
    index_gt = torch.clip(index_gt, 0, cost_volume.size(1) - 1)
    phi_gt = phi(index_gt)
    gt_loss = 1-phi_gt[mask].mean()

    index_range = torch.arange(cost_volume.size(1)).view(1, -1, 1, 1).to(cost_volume.device).repeat(cost_volume.size(0), 1, cost_volume.size(2), cost_volume.size(3))
    low=index_gt-1.5
    high=index_gt+1.5
    index_range_mask = ((index_range >= low) & (index_range < high) | (~mask)).bool()
    cv_nm = torch.masked_fill(cost_volume, index_range_mask, 0)
    cost_nm = torch.topk(cv_nm, k=k, dim=1, largest=True).values
    nm_loss = torch.clip(cost_nm + threshold - phi_gt.detach(), min=0)[mask.repeat(1,k,1,1)].mean()
    init_loss = gt_loss + nm_loss
    metrics = {
        'init_loss': init_loss.item(),
        'init_gt_loss': gt_loss.item(),
        'init_nm_loss': nm_loss.item(),
        'forward_mask_rate': ((cost_nm[:, :1] + 0.3 - phi_gt) > 0).float().mean().item()
    }

    return init_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, sum_freq, frame_length=1):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = wandb
        self.sum_freq = sum_freq * frame_length
        self.frame_length = frame_length

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format((self.total_steps + 1) // self.frame_length, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps // self.frame_length}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = wandb

        self.writer.log(self.running_loss, commit=False)

    def push(self, metrics):
        """
        this function is used to record the running metrics when training
        :param metrics:
        :return:
        """
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            for key in metrics:
                self.running_loss[key] /= self.sum_freq
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        """
        this function is used to record the running metrics when testing
        :return:
        """
        if self.writer is None:
            self.writer = wandb

        self.writer.log(results)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('set seed %d successfully' % seed)


def save_ckpt(args, model, optimizer, scheduler, total_steps):
    save_path = Path('checkpoints/%d_%s.pth' % ((total_steps + 1), args.pth_name))
    logging.info(f"Saving file {save_path.absolute()}")
    state = {
        'total_steps': total_steps,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, save_path)


def train(args):
    torch.use_deterministic_algorithms(True,warn_only=True)
    # ddp mode
    if args.ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        args.device = local_rank
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        model = TCStereo(args)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.device)
        else:
            model = model.cuda(args.device)
        if args.restore_ckpt is not None:
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(args.restore_ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'], strict=True)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        setup_seed(1234 + torch.distributed.get_rank())
    # single gpu mode
    else:
        setup_seed(1234)
        model = TCStereo(args)
        if args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth")
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint, strict=True)
            logging.info(f"Done loading checkpoint")

    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    print("dataloader length: %d" % len(train_loader))
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, 100, frame_length=args.frame_length)

    model.cuda(args.device)
    model.train()

    # # we set norm layer to none instead of freezing bn
    # if args.ddp:
    #     model.module.freeze_bn()  # We keep BatchNorm frozen
    # else:
    #     model.freeze_bn()

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    epoch = 0
    metrics = dict()

    # training loop
    while should_keep_training:
        epoch += 1
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            # temporal training
            if args.temporal:
                # assert args.frame_length > 1
                assert args.frame_length == data_blob[0].shape[1], [data_blob[0].shape[1]]
                # variables
                loss = 0
                flow_q, previous_T, fmap1, net_list = None, None, None, None
                params = dict()
                baseline = data_blob[-1].cuda(args.device)
                K = data_blob[-2].cuda(args.device)
                data_blob = data_blob[:-2]

                model.zero_grad()
                for i_seq in range(args.frame_length):
                    image1, image2, flow, valid, T = [x[:, i_seq] for x in data_blob]
                    params.update({'K': K,  # n,3,3
                                   'T': T,  # n,4,4
                                   'previous_T': previous_T,
                                   'last_disp': flow_q,
                                   'last_net_list': net_list,
                                   'fmap1': fmap1,
                                   'baseline': baseline,  # n
                                   })

                    assert model.training
                    training_output = model(image1, image2, iters=args.train_iters, params=params if flow_q is not None else None, frame_id=i_seq)
                    assert model.training

                    # losses
                    loss_gamma = 0.9
                    n_predictions = len(training_output['flow_predictions'])
                    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
                    loss_weights = [adjusted_loss_gamma ** (n_predictions - i - 1) for i in range(n_predictions)]

                    # exclude invalid pixels and extremely large diplacements
                    mag = torch.sum(flow ** 2, dim=1).sqrt()

                    # exclude extremly large displacements
                    valid = ((valid >= 0.5) & (mag < 700)).unsqueeze(1)

                    disp_grad_gt, _ = disp2disp_gradient_xy(-flow)  # n,2,h,w
                    disp_norm_gt = F.normalize(torch.cat((disp_grad_gt, -torch.ones_like(disp_grad_gt[:, :1])), dim=1), dim=1)  # n,3,h,w

                    # sequential loss
                    seq_loss, seq_metrics = sequence_loss(training_output['flow_mono'], training_output['flow_init'], training_output['flow_predictions'], flow,
                                                          valid, loss_weights)
                    loss += seq_loss.item()
                    metrics.update(seq_metrics)

                    # cost volume init loss
                    in_loss, init_metrics = init_loss(training_output['cost_volume'], flow, valid, k=args.init_k, scale=1/(2**args.n_downsample), threshold=args.init_thres)
                    loss += in_loss.item()
                    metrics.update(init_metrics)

                    # disparity normal loss
                    norm_loss, norm_metrics = disp_normal_loss(training_output['flow_q_predictions'], disp_norm_gt, valid, loss_weights,
                                                               scale=1 / (2 ** args.n_downsample), dense_gt=False if args.train_dataset == 'kitti_raw' else True)
                    loss += 0.25*norm_loss.item()
                    metrics.update(norm_metrics)

                    # disparity gradient loss
                    grad_loss, grad_metrics = disp_grad_loss(training_output['disp_grad_q_predictions'], disp_grad_gt, valid, loss_weights,
                                                             scale=1 / (2 ** args.n_downsample), dense_gt=False if args.train_dataset == 'kitti_raw' else True)
                    loss += 5*grad_loss.item()
                    metrics.update(grad_metrics)

                    # logging
                    logger.push(metrics)

                    # backward and accumulate the gradients
                    scaler.scale((seq_loss+in_loss+0.25*norm_loss+5*grad_loss) / args.frame_length).backward()

                    # temporal info update
                    previous_T = T
                    net_list = training_output['net_list']
                    flow_q = training_output['flow_q']
                    fmap1 = training_output['fmap1']

                # update parameters
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                logger.writer.log({"live_loss": loss / args.frame_length, 'learning_rate': optimizer.param_groups[0]['lr']})
            # validation & save ckpt
            if total_steps % validation_frequency == validation_frequency - 1:
                if local_rank == 0:
                    save_ckpt(args, model, optimizer, scheduler, total_steps)
                if args.train_dataset == 'TartanAir':
                    results = validate_tartanair(args, model.module if args.ddp else model, iters=args.valid_iters)
                    logger.write_dict(results)
                    model.train()
                elif args.train_dataset == 'sceneflow':
                    results = validate_temporal_things(args,model.module if args.ddp else model, iters=args.valid_iters)
                    logger.write_dict(results)
                    model.train()
                else:
                    pass

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000 and local_rank == 0:
            save_ckpt(args, model, optimizer, scheduler, total_steps)

    print("FINISHED TRAINING")
    # logger.close()
    if local_rank == 0:
        PATH = 'checkpoints/%s.pth' % args.name
        torch.save({'model': model.module.state_dict()}, PATH)

    return PATH


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='TC-Stereo', help="name of your experiment")
    parser.add_argument('--pth_name', default='', help="name of the checkpoint")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_dataset', default='sceneflow', choices=['TartanAir', 'sceneflow', "kitti_raw"], help="training dataset.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=5, help="number of updates to the disparity field in each training forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--ddp', action='store_true', help="DDP mode")
    parser.add_argument('--device', default=None, type=int, help='gpu id')
    parser.add_argument('--temporal', action='store_true', help="temporal mode")
    parser.add_argument('--frame_length', default=2, type=int, help='frame length for training')
    parser.add_argument('--sync_bn', action='store_true', help="using sync bn")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=5, help='number of flow-field updates during inference forward pass')

    # Architecure choices
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default='none', choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")

    # Loss parameters
    parser.add_argument('--init_thres', type=float, default=0.5, help="the threshold gap of contrastive loss for cost volume.")
    parser.add_argument('--init_k', type=int, default=3, help="the number of top k in training.")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)
    wandb.init(
        job_type="train",
        project=args.name,
        entity="zengjiaxi"
    )
    # add the args to wandb
    wandb.config.update(args)
    train(args)
