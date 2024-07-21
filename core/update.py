import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.geo_utils import disp2disp_grad_candidates
from core.utils.basic_layers import Conv2x_IN
from core.utils.utils import coords_grid


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Lightfuse(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Lightfuse, self).__init__()
        self.convzr = nn.Conv2d(hidden_dim + input_dim, hidden_dim * 2, kernel_size=1, padding=0)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=1, padding=0)

    def forward(self, h, x):
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h]
        assert not torch.isnan(x).any() and not torch.isinf(x).any(), [x]
        hxm = torch.cat([h, x], dim=1)
        hxmz, hxmr = self.convzr(hxm).chunk(2, dim=1)
        z = torch.sigmoid(hxmz)
        r = torch.sigmoid(hxmr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = z * h + (1 - z) * q
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h]
        return h


class Hardfuse(nn.Module):
    def __init__(self):
        super(Hardfuse, self).__init__()

    def forward(self, h, x, mask):
        h = mask * h + (1 - mask) * x
        return h


class HiddenstateUpdater(nn.Module):
    def __init__(self, hidden_dim):
        super(HiddenstateUpdater, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(1, 64, 1, padding=0),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(64, 64, 1, padding=0))
        self.convzr = nn.Conv2d(hidden_dim + 64, hidden_dim * 2, 1, padding=0)
        self.convq = nn.Conv2d(hidden_dim + 64, hidden_dim, 1, padding=0)

    def forward(self, h, x):
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h]
        assert not torch.isnan(x).any() and not torch.isinf(x).any(), [x]
        x = self.convs(x)
        hxm = torch.cat([h, x], dim=1)
        hxmz, hxmr = self.convzr(hxm).chunk(2, dim=1)
        z = torch.sigmoid(hxmz)
        r = torch.sigmoid(hxmr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = z * h + (1 - z) * q
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h]
        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convzr = nn.Conv2d(hidden_dim + input_dim, hidden_dim * 2, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h, h.shape, torch.isnan(h).any(), torch.isinf(h).any()]
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z, r = self.convzr(hx).chunk(2, dim=1)
        z = torch.sigmoid(z + cz)
        r = torch.sigmoid(r + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q
        assert not torch.isnan(h).any() and not torch.isinf(h).any(), [h]
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            assert not torch.isnan(corr).any() and not torch.isinf(corr).any(), [corr]
            assert not torch.isnan(flow).any() and not torch.isinf(flow).any(), [flow]
            motion_features = self.encoder(flow, corr)
            assert not torch.isnan(motion_features).any() and not torch.isinf(motion_features).any(), [motion_features]
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])
        return net, delta_flow


class DispGradPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv_grad_stem = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 32, 3, 1, 1))
        self.conv_grad_candidate_stem = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                                      nn.ReLU(inplace=True),
                                                      nn.Conv2d(64, 64, 3, 1, 1))
        self.conv_4_4 = nn.Sequential(nn.Conv2d(32 + 64 + 64, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True))
        self.conv_4_8 = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1),
                                      nn.ReLU(inplace=True))
        self.conv_8_8 = nn.Sequential(nn.Conv2d(96 + 64, 96, 3, 1, 1),
                                      nn.ReLU(inplace=True))
        self.conv_8_16 = nn.Sequential(nn.Conv2d(96, 128, 3, 2, 1),
                                       nn.ReLU(inplace=True))
        self.conv_16_16 = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, 1, 1),
                                        nn.ReLU(inplace=True))
        self.conv_16_8 = Conv2x_IN(128, 96, deconv=True, concat=False, keep_concat=False, IN=False)
        self.conv_8_4 = Conv2x_IN(96, 64, deconv=True, concat=False, keep_concat=False, IN=False)
        self.residual_head = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(128, 2, 3, 1, 1))
        self.conv_out = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))

    def forward(self, disp_grad, disp, clist):
        disp_grad = 5 * disp_grad.detach()
        disp = disp.detach()
        N, _, H, W = disp.shape
        grad_cands = disp2disp_grad_candidates(disp, level=2)  # n,2,16,h,w
        x4_grad = self.conv_grad_stem(disp_grad)
        x4_candidate = self.conv_grad_candidate_stem(grad_cands.reshape(N, -1, H, W))
        # DispGrad encoder-decoder
        x4 = self.conv_4_4(torch.cat((x4_grad, x4_candidate, clist[0]), dim=1))  # 64
        x8 = self.conv_4_8(x4)  # 96
        x8 = self.conv_8_8(torch.cat((x8, clist[1]), dim=1))
        x16 = self.conv_8_16(x8)  # 128
        x16 = self.conv_16_16(torch.cat((x16, clist[2]), dim=1))
        x8_up = self.conv_16_8(x16, x8)
        x4_up = self.conv_8_4(x8_up, x4)
        grad_refine = (disp_grad + self.residual_head(x4_up)) / 5
        return grad_refine, self.conv_out(x4_up)


class DispRefine(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # propagation kernels
        kernel_prop = torch.zeros((9, 1, 3, 3))
        vus = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for i, vu in enumerate(vus):
            v, u = vu
            kernel_prop[i, :, v, u] = 1
        self.kernel_prop = kernel_prop

        # difference kernels
        kernel_diff = torch.zeros((9, 1, 3, 3))
        vus = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        kernel_diff[:, :, 1, 1] = 1
        for i, vu in enumerate(vus):
            v, u = vu
            kernel_diff[i, :, v, u] = kernel_diff[i, :, v, u] - 1
        self.kernel_diff = kernel_diff

        self.context_compress = nn.Sequential(
            nn.Conv2d(128 + 64, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1)
        )
        self.disp_f_stem = nn.Sequential(nn.Conv2d(27, 96, 1, 1, 0),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(96, 96, 1, 1, 0))
        self.conv_fuse = nn.Sequential(nn.Conv2d(192, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True))
        self.w_head = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 9, 1, 1, 0))
        factor = 2 ** self.args.n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))

    def propagate_disparity(self, disparity_grad, disparity_map):
        """
        Args:
        - disparity_grad: tensor of shape [N, 2, H, W], disparity gradients along x and y axes
        - disparity_map: tensor of shape [N, 1, H, W], initial disparity map

        Returns:
        - propagated_disparities: tensor of shape [N, 9, H, W], disparity candidates in 8-neighborhood
        """
        N, _, H, W = disparity_grad.size()

        # pad
        disparity_grad = F.pad(disparity_grad, pad=(1, 1, 1, 1))
        disparity_map = F.pad(disparity_map, pad=(1, 1, 1, 1), mode='replicate')
        coords = coords_grid(N, H + 2, W + 2).to(disparity_grad.device)  # n,2,h+2,w+2

        # prop
        cat_prop = torch.cat((disparity_map, disparity_grad), dim=1).reshape(-1, 1, H + 2, W + 2)
        cat_prop = F.conv2d(cat_prop.repeat(1, 9, 1, 1), self.kernel_prop.to(disparity_grad.device), padding=0, groups=9).reshape(N, 3, 9, H, W)  # n,3,9,h,w
        disparity_map_prop, disparity_grad_prop = cat_prop[:, :1], cat_prop[:, 1:]

        # diff
        cat_diff = torch.cat((disparity_grad, coords), dim=1).reshape(-1, 1, H + 2, W + 2)  # n,4,h+2,w+2
        cat_diff = F.conv2d(cat_diff.repeat(1, 9, 1, 1), self.kernel_diff.to(disparity_grad.device), padding=0, groups=9).reshape(N, -1, 9, H, W)  # n,4,9,h,w
        grad_diff, coords_diff = cat_diff[:, :2], cat_diff[:, 2:]

        #  propagate
        propagated_disparities = disparity_map_prop + disparity_grad_prop[:, :1] * coords_diff[:, :1] + disparity_grad_prop[:, 1:] * coords_diff[:, 1:]   # n,1,9,h,w
        matrix = grad_diff.reshape(N, -1, H, W).abs()

        return propagated_disparities.squeeze(1), matrix.detach()

    def forward(self, disp_grads, disp, context_disp, context_grad, test_mode=False):
        disp = disp.detach()
        context = self.context_compress(torch.cat((context_disp, context_grad), dim=1))
        disp_candidates, matrix = self.propagate_disparity(disp_grads, disp)  # N, 9, H, W
        disp_f = self.disp_f_stem(torch.cat((disp_candidates.detach(), matrix), dim=1))
        fused_f = self.conv_fuse(torch.cat((disp_f, context), dim=1))  # N, 9, H, W
        w = self.w_head(fused_f)
        w_max = torch.max(w, dim=1, keepdim=True)[0]
        w = torch.softmax(w - w_max, dim=1)
        refined_disparity = torch.sum(w * disp_candidates, dim=1, keepdim=True)
        if test_mode:
            mask = None
        else:
            mask = 0.25 * self.mask(fused_f)
        return refined_disparity, mask


class DisparityCompletor(nn.Module):
    def __init__(self):
        super(DisparityCompletor, self).__init__()
        # sparse disparity embedding
        self.conv_disp_stem = nn.Sequential(nn.Conv2d(1, 64, 1, 1, 0),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, 1, 1, 0))
        self.conv_cost_stem = nn.Sequential(nn.Conv2d(1, 32, 1, 1, 0),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 32, 1, 1, 0))
        self.conv_mask_stem = nn.Sequential(nn.Conv2d(1, 32, 1, 1, 0),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(32, 32, 1, 1, 0))
        self.conv_disp_fuse = nn.Sequential(nn.Conv2d(128, 128, 1, 1, 0),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 64, 1, 1, 0))
        # encoder
        self.conv_4_4 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                      nn.InstanceNorm2d(192),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 64, 3, 1, 1))
        self.conv_4_8 = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1),
                                      nn.InstanceNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1))
        self.conv_8_8 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                      nn.InstanceNorm2d(192),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(192, 64, 3, 1, 1))
        self.conv_8_16 = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, 3, 1, 1))
        self.conv_16_16 = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                        nn.InstanceNorm2d(192),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(192, 64, 3, 1, 1))
        # decoder
        self.conv_16_8 = Conv2x_IN(64, 64, deconv=True, concat=False, keep_concat=False, IN=True)
        self.conv_8_4 = Conv2x_IN(64, 64, deconv=True, concat=False, keep_concat=False, IN=True)
        self.disp_head = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 1, 3, 1, 1))
        self.w_head = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 1, 3, 1, 1),
                                    nn.Sigmoid())

        self.conv_out16_disp = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                             nn.InstanceNorm2d(192),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(192, 128, 3, 1, 1))
        self.conv_out8_disp = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                            nn.InstanceNorm2d(192),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(192, 128, 3, 1, 1))
        self.conv_out4_disp = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 1),
                                            nn.InstanceNorm2d(192),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(192, 128, 3, 1, 1))

    def forward(self, disp, cost, mask, context_list):
        assert not torch.isnan(disp).any() and not torch.isinf(disp).any(), [disp]
        assert not torch.isnan(cost).any() and not torch.isinf(cost).any(), [cost]
        # sparse disparity embedding
        mask = mask - 0.5
        disp = disp / 10  # scale
        disp_f4 = self.conv_disp_stem(disp)
        cost_f4 = self.conv_cost_stem(cost)
        mask_f4 = self.conv_mask_stem(mask)
        x4_disp = self.conv_disp_fuse(torch.cat((disp_f4, cost_f4, mask_f4), dim=1))
        # encoder
        x4 = self.conv_4_4(torch.cat((x4_disp, context_list[0]), dim=1))
        x8 = self.conv_4_8(x4)
        x8 = self.conv_8_8(torch.cat((x8, context_list[1]), dim=1))
        x16 = self.conv_8_16(x8)
        x16_out = self.conv_16_16(torch.cat((x16, context_list[2]), dim=1))
        # decoder
        x8_out = self.conv_16_8(x16_out, x8)
        x4_out = self.conv_8_4(x8_out, x4)
        disp_mono = self.disp_head(x4_out)
        w = self.w_head(x4_out)
        assert not torch.isnan(disp_mono).any() and not torch.isinf(disp_mono).any(), [disp_mono]
        assert not torch.isnan(w).any() and not torch.isinf(w).any(), [w]
        disp_completed = (w * disp + (1 - w) * disp_mono) * 10  # rescale
        disp_mono = disp_mono * 10  # rescale
        x4_out = torch.cat((x4_out, context_list[0]), dim=1)
        x8_out = torch.cat((x8_out, context_list[1]), dim=1)
        x16_out = torch.cat((x16_out, context_list[2]), dim=1)
        disp_net_list = [self.conv_out4_disp(x4_out), self.conv_out8_disp(x8_out), self.conv_out16_disp(x16_out)]

        return disp_completed, disp_mono, w, disp_net_list
