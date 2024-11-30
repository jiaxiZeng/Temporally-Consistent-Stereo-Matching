import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, Lightfuse, DispGradPredictor, DispRefine, DisparityCompletor, HiddenstateUpdater
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D
from core.utils.utils import coords_grid, upflow8, bilinear_sampler
from core.utils.geo_utils import cal_relative_transformation, warp, get_backward_grid, disp2disp_gradient_xy

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class TCStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        # args
        self.args = args
        self.scale_rate = 1 / (2 ** args.n_downsample)
        context_dims = args.hidden_dims

        # feature extractor
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # context convs
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=1) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

        # hidden state fusion
        self.previous_current_hideen_fuse = nn.ModuleList([Lightfuse(args.hidden_dims[i], args.hidden_dims[i]) for i in range(self.args.n_gru_layers)])
        # disparity completion
        self.disp_completor = DisparityCompletor()
        # gradient space refinement
        self.disp_grad_refine = DispGradPredictor(args)
        # gradient guided propagation
        self.disp_refine = DispRefine(args)
        # context convs for gradient
        self.context_zqr_convs_grad = nn.ModuleList([nn.Conv2d(context_dims[i], 64, 3, padding=1) for i in range(self.args.n_gru_layers)])
        # hidden state updater
        self.hiddenstate_update = HiddenstateUpdater(context_dims[0])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                print("freeze!")
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0[:, :1], coords1[:, :1]

    def upsample_flow(self, flow, mask, scale=True):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask_max = torch.max(mask, dim=2, keepdim=True)[0]
        mask = torch.softmax(mask - mask_max, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1) if scale else F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def fuse_previous_current_hidden_state(self, net_list, warp_net_list):
        fused_net_list = []
        for net, warp_net, conv_fuse in zip(net_list, warp_net_list, self.previous_current_hideen_fuse):
            fused_net_list.append(conv_fuse(net, warp_net))
        return fused_net_list

    def forward(self, image1, image2, iters=12, params=None, test_mode=False, frame_id=0):
        """ Estimate disparity maps from stereo sequences"""
        flow_predictions = []
        flow_q_predictions = []
        disp_grad_q_predictions = []
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

        # correlator initialization
        corr_block = CorrBlock1D
        fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels, thres=self.args.init_thres)

        # temporal information
        if params is not None:  # not the first frame
            # intrinsic matrix
            K = params['K']  # b,3,3
            K_scale = K * torch.tensor([self.scale_rate, self.scale_rate, 1]).view(1, 3, 1).to(K.device)
            K_scale_inv = torch.linalg.inv(K_scale)
            # pose
            T = params['T']
            previous_T = params['previous_T']
            relative_T = cal_relative_transformation(previous_T, T)
            # baseline
            baseline = params['baseline']
            # disparity of the last frame
            flow_init = params['last_disp']
            # hidden state of the last frame
            last_net_list = params['last_net_list']
            # feature map of the last frame
            last_fmap1 = params['fmap1']
            # warping disparity & matching features
            warped_disparity, warped_fmap1, sparse_mask = warp(-flow_init, last_fmap1, relative_T, K_scale, K_scale_inv, baseline)
            sparse_disp = warped_disparity
            cost = torch.sum(F.normalize(fmap1, dim=1) * F.normalize(warped_fmap1, dim=1), dim=1, keepdim=True)
            cost = cost * sparse_mask
        else:  # the first frame
            sparse_disp, cost, sparse_mask = corr_fn.argmax_disp()
            last_net_list = None

        # disparity completion
        with autocast(enabled=self.args.mixed_precision):
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            grad_list = [conv(i) for i, conv in zip(inp_list, self.context_zqr_convs_grad)]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
            net_list = [x[0] for x in cnet_list]

            disp_init, disp_mono, w, net_list = self.disp_completor(sparse_disp, cost.detach(), sparse_mask, net_list)

        # hidden state warping
        if last_net_list is None:  # zero initialization for the first frame
            warped_net_list = [torch.zeros_like(x[0]) for x in cnet_list]
        else:  # warp features from the last frame
            warped_net_list = []
            backward_grid = get_backward_grid(disp_init.detach(), cal_relative_transformation(T, previous_T), K_scale, K_scale_inv, baseline)
            assert not torch.isnan(backward_grid).any() and not torch.isinf(backward_grid).any(), [backward_grid, torch.max(backward_grid)]
            for (i, net) in enumerate(last_net_list):
                warped_net_list.append(bilinear_sampler(net.float(), backward_grid.permute(0, 2, 3, 1)))
                backward_grid = 0.5 * F.interpolate(backward_grid, scale_factor=0.5, mode='bilinear', align_corners=True)

        # hidden state fusion
        with autocast(enabled=self.args.mixed_precision):
            net_list = [torch.tanh(x) for x in net_list]
            net_list = self.fuse_previous_current_hidden_state(net_list, warped_net_list)

        # flow initialization
        coords0, coords1 = self.initialize_flow(fmap1)  # N,1,H,W
        coords1 = coords0 - disp_init.detach()

        # iterative update
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # N,candidate,C,H,W
            flows_x = coords1 - coords0  # flow along x-axis, disp = -flow_x

            # disparity space refinement
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers == 3, iter16=True, iter08=False, update=False)
                net_list, delta_flow = self.update_block(net_list, inp_list, corr, flows_x, iter32=self.args.n_gru_layers == 3,
                                                         iter16=self.args.n_gru_layers >= 2)
            coords1 = coords1 + delta_flow
            disp_q = coords0 - coords1

            # gradient space refinement
            disp_grad, _ = disp2disp_gradient_xy(disp_q.detach())
            with autocast(enabled=self.args.mixed_precision):
                # gradient space refinement: grad-> refined grad
                disp_grad, context = self.disp_grad_refine(disp_grad, disp_q, grad_list)
                # disparity propagation: grad-> update disp
                refined_disp, up_mask = self.disp_refine(disp_grad, disp_q, net_list[0], context, test_mode and itr < iters - 1)
                delta_disp = (refined_disp - disp_q).detach()
                # update hidden states
                net_list = [self.hiddenstate_update(net_list[0], delta_disp), net_list[1], net_list[2]]

            coords1 = coords0 - refined_disp  # update coords1

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flows_up = upflow8(-disp_q)
                flow_refine_up = upflow8(-refined_disp)
            else:
                flows_up = self.upsample_flow(-disp_q, up_mask.detach())
                flow_refine_up = self.upsample_flow(-refined_disp, up_mask)

            flow_predictions.append([flows_up, flow_refine_up])
            flow_q_predictions.append([-disp_q, -refined_disp])
            disp_grad_q_predictions.append(disp_grad)

        flow_q = -refined_disp
        net_list = [x.detach() for x in net_list]

        if test_mode:
            testing_output = {'flow': torch.clip(flow_refine_up, max=0),
                              'flow_q': torch.clip(flow_q, max=0),
                              'net_list': net_list,
                              'fmap1': fmap1.detach(),
                              }
            return testing_output
        else:
            training_output = {
                # losses
                'flow_mono': -4 * F.interpolate(disp_mono, scale_factor=4, mode='bilinear', align_corners=True),
                'flow_init': -4 * F.interpolate(disp_init, scale_factor=4, mode='bilinear', align_corners=True),
                'flow_predictions': flow_predictions,
                'flow_q_predictions': flow_q_predictions,
                'disp_grad_q_predictions': disp_grad_q_predictions,
                'cost_volume': corr_fn.get_cost_volume(),
                # temporal info for next frame
                'flow_q': torch.clip(flow_q.detach(), max=0),
                'net_list': net_list,
                'fmap1': fmap1.detach(),
            }
            return training_output
