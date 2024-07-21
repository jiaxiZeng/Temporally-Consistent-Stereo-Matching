import numpy as np
import random
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid


class TemporalFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform_torch(self, seq1, seq2):
        """ Photometric augmentation """
        seq_len = len(seq1)

        seq1 = torch.cat(seq1, dim=1)  # 3,n*h,w
        seq2 = torch.cat(seq2, dim=1)

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            seq1 = self.photo_aug(seq1).float()
            seq2 = self.photo_aug(seq2).float()

        # symmetric
        else:
            image_stack = torch.cat([seq1, seq2], dim=1)
            image_stack = self.photo_aug(image_stack).float()
            seq1, seq2 = torch.split(image_stack, image_stack.shape[1]//2, dim=1)
        seq1 = torch.split(seq1, seq1.shape[1]//seq_len, dim=1)
        seq2 = torch.split(seq2, seq2.shape[1]//seq_len, dim=1)
        return seq1, seq2

    def color_transform(self, seq1, seq2):
        """ Photometric augmentation """
        seq_len = len(seq1)
        seq1 = np.concatenate(seq1, axis=0)
        seq2 = np.concatenate(seq2, axis=0)

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            seq1 = np.array(self.photo_aug(Image.fromarray(seq1)), dtype=np.uint8)
            seq2 = np.array(self.photo_aug(Image.fromarray(seq2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([seq1, seq2], axis=0)  # 2*h,w,3
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            seq1, seq2 = np.split(image_stack, 2, axis=0)

        seq1 = np.split(seq1, seq_len, axis=0)
        seq2 = np.split(seq2, seq_len, axis=0)

        return seq1, seq2

    def eraser_transform_torch(self, seq1, seq2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = seq1[0].shape[1:]
        # stack
        seq2 = torch.stack(seq2, dim=0)
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = torch.mean(seq2.reshape(-1, 3, ht * wd), dim=(0, 2))
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                seq2[:, :, y0:y0 + dy, x0:x0 + dx] = mean_color.view(1, 3, 1, 1)
        # unstack
        seq2 = [seq2[i] for i in range(seq2.shape[0])]
        return seq1, seq2

    def eraser_transform(self, seq1, seq2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = seq1[0].shape[:2]
        # stack
        seq2 = np.stack(seq2, axis=0)  # n,h,w,3
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(seq2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                seq2[:, y0:y0 + dy, x0:x0 + dx, :] = mean_color
        # unstack
        seq2 = [seq2[i] for i in range(seq2.shape[0])]
        return seq1, seq2

    def spatial_transform_torch(self, seq1, seq2, flow_seq, K):
        # randomly sample scale
        ht, wd = seq1[0].shape[1:]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        # scale
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            K = K * (torch.tensor([scale_x, scale_y, 1]).reshape((3, 1))).cuda(K.device)
            # rescale the images
            seq1 = torch.stack(seq1, dim=0)
            seq2 = torch.stack(seq2, dim=0)
            flow_seq = torch.stack(flow_seq, dim=0)

            seq1 = F.interpolate(seq1, scale_factor=(scale_y, scale_x), mode='bilinear')
            seq2 = F.interpolate(seq2, scale_factor=(scale_y, scale_x), mode='bilinear')
            flow_seq = F.interpolate(flow_seq, scale_factor=(scale_y, scale_x), mode='bilinear')
            flow_seq = flow_seq * torch.tensor([scale_x, scale_y]).view(1, 2, 1, 1).cuda(K.device)

        if self.yjitter:
            y0 = np.random.randint(2, seq1.shape[2] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, seq1.shape[3] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            seq1 = seq1[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, :, y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            shift = torch.zeros_like(K)
            shift[0, 2] = x0
            shift[1, 2] = y0
            K = K - shift

        else:

            y0 = np.random.randint(0, seq1.shape[2] - self.crop_size[0])
            x0 = np.random.randint(0, seq1.shape[3] - self.crop_size[1])

            seq1 = seq1[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            shift = torch.zeros_like(K)
            shift[0, 2] = x0
            shift[1, 2] = y0
            K = K - shift

        return seq1, seq2, flow_seq, K

    def spatial_transform(self, seq1, seq2, flow_seq, K):
        # randomly sample scale
        ht, wd = seq1[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            K = K * (np.array([scale_x, scale_y, 1]).reshape((3, 1)))
            # rescale the images
            img1_list = []
            img2_list = []
            flow_list = []
            for img1, img2, flow in zip(seq1, seq2, flow_seq):
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = flow * [scale_x, scale_y]

                img1_list.append(img1)
                img2_list.append(img2)
                flow_list.append(flow)
            seq1 = img1_list
            seq2 = img2_list
            flow_seq = flow_list

        # stack
        seq1 = np.stack(seq1, axis=0)
        seq2 = np.stack(seq2, axis=0)
        flow_seq = np.stack(flow_seq, axis=0)
        if self.yjitter:
            y0 = np.random.randint(2, seq1.shape[1] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, seq1.shape[2] - self.crop_size[1] - 2)
            shift = np.zeros((3, 3))
            shift[0, 2] = x0
            shift[1, 2] = y0

            y1 = y0 + np.random.randint(-2, 2 + 1)

            seq1 = seq1[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            K = K - shift

        else:
            y0 = np.random.randint(0, seq1.shape[1] - self.crop_size[0])
            x0 = np.random.randint(0, seq1.shape[2] - self.crop_size[1])
            shift = np.zeros((3, 3))
            shift[0, 2] = x0
            shift[1, 2] = y0

            seq1 = seq1[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            K = K - shift

        return seq1, seq2, flow_seq, K

    def __call__(self, seq1, seq2, flow_seq, K):
        '''
        :param seq1: n,h,w,3
        :param seq2: n,h,w,3
        :param flow_seq: n,h,w,2
        :return:
        '''
        seq1, seq2 = self.color_transform_torch(seq1, seq2)
        seq1, seq2 = self.eraser_transform_torch(seq1, seq2)

        seq1, seq2, flow_seq, K = self.spatial_transform_torch(seq1, seq2, flow_seq, K)

        seq1=seq1.contiguous()
        seq2=seq2.contiguous()
        flow_seq=flow_seq.contiguous()
        K=K.contiguous()

        return seq1, seq2, flow_seq, K  # n,h,w,c


class TemporalSparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.5 / 3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.0
        self.eraser_aug_prob = 0.5

    def color_transform_torch(self, seq1, seq2):
        """ Photometric augmentation """
        seq_len = len(seq1)

        seq1 = torch.cat(seq1, dim=1)  # 3,n*h,w
        seq2 = torch.cat(seq2, dim=1)

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            seq1 = self.photo_aug(seq1).float()
            seq2 = self.photo_aug(seq2).float()

        # symmetric
        else:
            image_stack = torch.cat([seq1, seq2], dim=1)
            image_stack = self.photo_aug(image_stack).float()
            seq1, seq2 = torch.split(image_stack, image_stack.shape[1]//2, dim=1)
        seq1 = torch.split(seq1, seq1.shape[1]//seq_len, dim=1)
        seq2 = torch.split(seq2, seq2.shape[1]//seq_len, dim=1)
        return seq1, seq2

    def color_transform(self, seq1, seq2):
        """ Photometric augmentation """
        seq_len = len(seq1)
        seq1 = np.concatenate(seq1, axis=0)
        seq2 = np.concatenate(seq2, axis=0)

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            seq1 = np.array(self.photo_aug(Image.fromarray(seq1)), dtype=np.uint8)
            seq2 = np.array(self.photo_aug(Image.fromarray(seq2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([seq1, seq2], axis=0)  # 2*h,w,3
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            seq1, seq2 = np.split(image_stack, 2, axis=0)

        seq1 = np.split(seq1, seq_len, axis=0)
        seq2 = np.split(seq2, seq_len, axis=0)

        return seq1, seq2

    def eraser_transform_torch(self, seq1, seq2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = seq1[0].shape[1:]
        # stack
        seq2 = torch.stack(seq2, dim=0)
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = torch.mean(seq2.reshape(-1, 3, ht * wd), dim=(0, 2))
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                seq2[:, :, y0:y0 + dy, x0:x0 + dx] = mean_color.view(1, 3, 1, 1)
        # unstack
        seq2 = [seq2[i] for i in range(seq2.shape[0])]
        return seq1, seq2

    def eraser_transform(self, seq1, seq2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = seq1[0].shape[:2]
        # stack
        seq2 = np.stack(seq2, axis=0)  # n,h,w,3
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(seq2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                seq2[:, y0:y0 + dy, x0:x0 + dx, :] = mean_color
        # unstack
        seq2 = [seq2[i] for i in range(seq2.shape[0])]
        return seq1, seq2

    def spatial_transform_torch(self, seq1, seq2, flow_seq, valid_seq, K):
        # randomly sample scale
        # seq: [3,h,w]*n
        ht, wd = seq1[0].shape[1:]
        assert ht >=320
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        # scale
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            K = K * (torch.tensor([scale_x, scale_y, 1]).reshape((3, 1))).cuda(K.device)
            # rescale the images
            seq1 = torch.stack(seq1, dim=0)
            seq2 = torch.stack(seq2, dim=0)
            flow_seq = torch.stack(flow_seq, dim=0)  # n,1,h,w
            valid_seq = torch.stack(valid_seq, dim=0)

            seq1 = F.interpolate(seq1, scale_factor=(scale_y, scale_x), mode='bilinear')
            seq2 = F.interpolate(seq2, scale_factor=(scale_y, scale_x), mode='bilinear')
            # Note: we use semi-dense pseudo disparity map from LEAStereo, so we don't need to use sparse resize for flow and valid
            # If you use sparse flow, please rewrite resize_sparse_flow_map_torch with K and use it here
            flow_seq = F.interpolate(flow_seq, scale_factor=(scale_y, scale_x), mode='bilinear')
            flow_seq = flow_seq * torch.tensor([scale_x, scale_y]).view(1, 2, 1, 1).cuda(K.device)
            valid_seq = (F.interpolate(valid_seq.float(), scale_factor=(scale_y, scale_x), mode='bilinear')==1).float()
        else:
            seq1 = torch.stack(seq1, dim=0)
            seq2 = torch.stack(seq2, dim=0)
            flow_seq = torch.stack(flow_seq, dim=0)  # n,1,h,w
            valid_seq = torch.stack(valid_seq, dim=0)
        # random crop
        y0 = np.random.randint(0, seq1.shape[2] - self.crop_size[0])
        x0 = np.random.randint(0, seq1.shape[3] - self.crop_size[1])
        y0 = np.clip(y0, 0, ht - self.crop_size[0])
        x0 = np.clip(x0, 0, wd - self.crop_size[1])
        seq1 = seq1[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        seq2 = seq2[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid_seq=valid_seq[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow_seq = flow_seq[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        shift = torch.zeros_like(K)
        shift[0, 2] = x0
        shift[1, 2] = y0
        K = K - shift
        return seq1, seq2, flow_seq, valid_seq, K

    def spatial_transform(self, seq1, seq2, flow_seq, valid_seq, K):
        # randomly sample scale
        ht, wd = seq1[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            K = K * (np.array([scale_x, scale_y, 1]).reshape((3, 1)))
            # rescale the images
            img1_list = []
            img2_list = []
            flow_list = []
            for img1, img2, flow in zip(seq1, seq2, flow_seq):
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                # flow_seq = cv2.resize(flow_seq, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                # flow_seq = flow_seq * [scale_x, scale_y]

                img1_list.append(img1)
                img2_list.append(img2)
                flow_list.append(flow)
            seq1 = img1_list
            seq2 = img2_list
            flow_seq = flow_list

        # stack
        seq1 = np.stack(seq1, axis=0)
        seq2 = np.stack(seq2, axis=0)
        flow_seq = np.stack(flow_seq, axis=0)
        if self.yjitter:
            y0 = np.random.randint(2, seq1.shape[1] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, seq1.shape[2] - self.crop_size[1] - 2)
            shift = np.zeros((3, 3))
            shift[0, 2] = x0
            shift[1, 2] = y0

            y1 = y0 + np.random.randint(-2, 2 + 1)

            seq1 = seq1[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            K = K - shift

        else:
            y0 = np.random.randint(0, seq1.shape[1] - self.crop_size[0])
            x0 = np.random.randint(0, seq1.shape[2] - self.crop_size[1])
            shift = np.zeros((3, 3))
            shift[0, 2] = x0
            shift[1, 2] = y0

            seq1 = seq1[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            seq2 = seq2[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow_seq = flow_seq[:, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            K = K - shift

        return seq1, seq2, flow_seq, K

    def resize_sparse_flow_map_torch(self, flow_seq, valid_seq, fx=1.0, fy=1.0):
        n, _, ht, wd = flow_seq.shape
        coords = torch.meshgrid(torch.arange(wd), torch.arange(ht))  # h,w,2
        coords = torch.stack(coords, dim=-1).reshape(-1,ht,wd,2).repeat(n, 1, 1, 1)
        # coords = np.meshgrid(np.arange(wd), np.arange(ht))
        # coords = np.stack(coords, axis=-1)

        coords = coords.reshape(n, -1, 2).float()
        flow_seq = flow_seq.reshape(n, -1, 2).float()
        valid_seq = valid_seq.reshape(n, -1, 1).float()

        coords0 = coords[(valid_seq >= 1).repeat(1,1,2)]
        flow0 = flow_seq[(valid_seq >= 1).repeat(1,1,2)]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        scale_rate = torch.tensor((fx, fy)).to(coords0.device)
        coords1 = coords0 * scale_rate
        flow1 = flow0 * scale_rate

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def __call__(self, seq1, seq2, flow_seq, valid_seq, K):
        '''
        :param seq1: n,h,w,3
        :param seq2: n,h,w,3
        :param flow_seq: n,h,w,2
        :return:
        '''
        seq1, seq2 = self.color_transform_torch(seq1, seq2)
        seq1, seq2 = self.eraser_transform_torch(seq1, seq2)

        seq1, seq2, flow_seq, valid_seq, K = self.spatial_transform_torch(seq1, seq2, flow_seq, valid_seq, K)

        seq1=seq1.contiguous()
        seq2=seq2.contiguous()
        flow_seq=flow_seq.contiguous()
        K=K.contiguous()

        return seq1, seq2, flow_seq, valid_seq, K  # n,h,w,c


if __name__ == '__main__':
    AG=TemporalFlowAugmentor([480, 640])
    seq1 = [torch.from_numpy(np.zeros((3, 480, 640),dtype=np.uint8)).cuda() for _ in range(2)]
    seq2 = [torch.from_numpy(np.zeros((3, 480, 640),dtype=np.uint8)).cuda() for _ in range(2)]
    flow_seq = [torch.from_numpy(np.zeros((2, 480, 640))).cuda() for _ in range(2)]
    K =torch.from_numpy(np.array([[320.0, 0, 320.0],
                 [0, 320.0, 240.0],
                 [0, 0, 1]])).cuda()
    a=AG(seq1,seq2,flow_seq,K)