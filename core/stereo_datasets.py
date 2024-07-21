# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import logging
import os
import copy
import random
from pathlib import Path
from glob import glob
import os.path as osp
import pykitti
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, TemporalFlowAugmentor, TemporalSparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, temporal=False, frame_sample_length=4, is_test=False, ddp=False, load_flow=False,index_by_scene=False):
        self.augmentor = None
        self.index_by_scene = index_by_scene
        self.sparse = sparse
        self.temporal = temporal
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.device = aug_params.pop("device", None) if aug_params is not None else None
        self.ddp = ddp
        self.load_flow = load_flow
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params) if not temporal else TemporalSparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params) if not temporal else TemporalFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = is_test
        self.init_seed = False
        self.flow_list = []
        self.pose_list = []
        self.frame_sample_length = frame_sample_length
        self.intrinsic_K = None
        self.baseline = None
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # set seed
        if not self.init_seed and not self.is_test:
            # worker_num = int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2
            worker_num = 4
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                id = worker_info.id + worker_num * self.device if self.ddp else worker_info.id
                print(f"worker_info.id:{worker_info.id},worker_num:{worker_num},self.device:{self.device},id:{id}")
                torch.manual_seed(id)
                np.random.seed(id)
                random.seed(id)
                self.init_seed = True

        if self.temporal:  # temporal loader: load data by image pair sequences

            # sample a sequence
            if self.index_by_scene:  # first, index by scene (or sequences), then index by frame slice
                # sequence path
                index = index % len(self.image_list)
                image1_list = self.image_list[index][0]
                image2_list = self.image_list[index][1]
                pose_list = self.pose_list[index]
                disp_list = self.disparity_list[index]
                # assert len(image1_list) == len(image2_list) == len(disp_list), [len(image1_list), len(image2_list), len(disp_list)]
                if self.is_test or self.augmentor is None:
                    if self.load_flow:
                        flow_list = self.flow_list[index]
                        assert len(image1_list) == len(flow_list), [len(image1_list), len(flow_list)]
                        return image1_list, image2_list, disp_list, pose_list, flow_list
                    else:
                        return image1_list, image2_list, disp_list, pose_list  # read image pairs online

                # sequences slicing
                frame_length = len(image1_list)
                low = np.random.randint(0, frame_length - self.frame_sample_length)
                high = low + self.frame_sample_length
                image1_list = image1_list[low:high]
                image2_list = image2_list[low:high]
                T_seq = np.stack(pose_list[low:high], axis=0).astype(np.float32)  # n,4,4
                disp_list = disp_list[low:high]
            else:  # index by frame slice; the image_list is already sliced, the sliced sequences from different videos are concatenated
                index = index % len(self.image_list)
                image1_list = self.image_list[index][0]
                image2_list = self.image_list[index][1]
                pose_list = self.pose_list[index]  # n,4,4
                T_seq = np.stack(pose_list, axis=0).astype(np.float32)  # n,4,4
                disp_list = self.disparity_list[index]
                assert len(image1_list) == len(image2_list) == len(disp_list), [len(image1_list), len(image2_list), len(disp_list)]
                assert not self.is_test and self.augmentor is not None, "test mode should set index_by_scene=True"

            # read, process and convert to tensor
            left_seq = []
            right_seq = []
            flow_seq = []
            valid_seq = []
            for (img1_path, img2_path, disp_path) in zip(image1_list, image2_list, disp_list):
                disp = self.disparity_reader(disp_path)
                if isinstance(disp, tuple):
                    disp, valid = disp  # h,w
                else:
                    valid = disp < 512
                img1 = frame_utils.read_gen(img1_path)
                img2 = frame_utils.read_gen(img2_path)

                img1 = torch.from_numpy(np.array(img1).astype(np.uint8)).permute(2, 0, 1).cuda(self.device)
                img2 = torch.from_numpy(np.array(img2).astype(np.uint8)).permute(2, 0, 1).cuda(self.device)

                disp = np.array(disp).astype(np.float32)
                flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)  # h,w,2
                flow = torch.from_numpy(flow).permute(2, 0, 1).float().cuda(self.device)  # 2,h,w
                valid = torch.from_numpy(valid)[None].float().cuda(self.device)

                # for grayscale images
                if len(img1.shape) == 2:
                    img1 = np.tile(img1[..., None], (1, 1, 3))
                    img2 = np.tile(img2[..., None], (1, 1, 3))
                else:
                    img1 = img1[:3]
                    img2 = img2[:3]

                left_seq.append(img1)
                right_seq.append(img2)
                flow_seq.append(flow)
                valid_seq.append(valid)

            # do augmentations
            if self.sparse:
                # if intrinsic_K is a list
                if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
                    K = self.intrinsic_K[index]
                else:
                    K = self.intrinsic_K
                left_seq, right_seq, flow_seq, valid_seq, K = self.augmentor(left_seq, right_seq, flow_seq, valid_seq, torch.from_numpy(K.copy()).cuda(self.device))
            else:
                if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
                    K = self.intrinsic_K[index]
                else:
                    K = self.intrinsic_K
                left_seq, right_seq, flow_seq, K = self.augmentor(left_seq, right_seq, flow_seq, torch.from_numpy(K.copy()).cuda(self.device))

            flow_seq = flow_seq[:, :1].float()
            K = K.float()
            T_seq = torch.from_numpy(T_seq).float().cuda(self.device)
            baseline = torch.tensor(self.baseline).float().cuda(self.device)  # 1

            if self.sparse:
                valid_seq = valid_seq.squeeze(1)  # n,h,w
            else:
                valid_seq = (flow_seq.abs() < 512).float().squeeze(1)


            return [image1_list[0], image1_list[-1]], left_seq, right_seq, flow_seq, valid_seq, T_seq, K, baseline

        else:  # single pair loader: load data by image pair
            if self.is_test:
                img1 = frame_utils.read_gen(self.image_list[index][0])
                img2 = frame_utils.read_gen(self.image_list[index][1])
                img1 = np.array(img1).astype(np.uint8)[..., :3]
                img2 = np.array(img2).astype(np.uint8)[..., :3]
                img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                return img1, img2, self.extra_info[index]

            index = index % len(self.image_list)
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            disp = np.array(disp).astype(np.float32)
            flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]

            if self.augmentor is not None:
                if self.sparse:
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if self.sparse:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)  # h , w

            flow = flow[:1]
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.pose_list = v * copy_of_self.pose_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        if self.intrinsic_K is not None and isinstance(self.intrinsic_K, list):
            copy_of_self.intrinsic_K = v * copy_of_self.intrinsic_K
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False,mode='single_frame', frame_sample_length=4, ddp=False):
        super(SceneFlowDatasets, self).__init__(aug_params,temporal=(mode == 'temporal'),
                                                frame_sample_length=frame_sample_length,is_test=things_test, ddp=ddp, load_flow=False,
                                                index_by_scene=things_test)
        self.root = root
        self.dstype = dstype
        self.intrinsic_K = []
        self.baseline = 1.
        if things_test:
            self._add_things("TEST",temporal=(mode == 'temporal'),frame_sample_length=frame_sample_length)
        else:
            self._add_things("TRAIN",temporal=(mode == 'temporal'),frame_sample_length=frame_sample_length)
            self._add_monkaa(temporal=(mode == 'temporal'),frame_sample_length=frame_sample_length)
            self._add_driving(temporal=(mode == 'temporal'),frame_sample_length=frame_sample_length)

    def _add_things(self, split='TRAIN',temporal=False,frame_sample_length=1):
        """ Add FlyingThings3D data """
        if not temporal:
            original_length = len(self.disparity_list)
            root = osp.join(self.root, 'FlyingThings3D')
            left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
            right_images = [im.replace('left', 'right') for im in left_images]
            disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

            # Choose a random subset of 400 images for validation
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(left_images))[:400])
            np.random.set_state(state)

            for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
                if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
            logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")
        else:
            root = osp.join(self.root, 'FlyingThings3D')
            pose_ps = sorted(glob(osp.join(root, 'pose', split, '*/*/camera_data.txt')))
            scenes = sorted(glob(osp.join(root, self.dstype, split, '**', '**')))
            pose_list = []
            left_image_list = []
            right_image_list = []
            disparity_list = []
            # index by slices
            for pose_p, scene in zip(pose_ps, scenes):
                poses = frame_utils.readsceneflow_pose(pose_p)
                left_images = sorted(glob(osp.join(scene, 'left/*.png')))
                right_images = [im.replace('left', 'right') for im in left_images]
                disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

                # remove the last frame if the number of frames is not enough
                if len(left_images) != len(poses):
                    if len(left_images)-len(poses)==1:
                        left_images = left_images[:-1]
                        right_images = right_images[:-1]
                        disparity_images = disparity_images[:-1]
                    else:
                        raise ValueError([len(left_images), len(poses), pose_p, left_images])

                assert len(left_images) == len(poses), [len(left_images), len(poses), pose_p,left_images]
                if split == 'TRAIN':  # index by slice
                    left_image_list += [left_images[i:i + frame_sample_length] for i in range(len(left_images) - frame_sample_length + 1)]
                    right_image_list += [right_images[i:i + frame_sample_length] for i in range(len(right_images) - frame_sample_length + 1)]
                    disparity_list += [disparity_images[i:i + frame_sample_length] for i in range(len(disparity_images) - frame_sample_length + 1)]
                    pose_list += [poses[i:i + frame_sample_length] for i in range(len(poses) - frame_sample_length + 1)]
                else:  # index by scene
                    left_image_list.append(left_images)
                    right_image_list.append(right_images)
                    disparity_list.append(disparity_images)
                    pose_list.append(poses)

            for idx, (img1, img2, disp, pose) in enumerate(zip(left_image_list, right_image_list, disparity_list, pose_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.pose_list += [pose]
                self.intrinsic_K += [np.array(
                    [[1050., 0., 479.5],
                     [0., 1050., 269.5],
                     [0.0, 0.0, 1.0]]
                )]


    def _add_monkaa(self,temporal=False,frame_sample_length=1):
        """ Add FlyingThings3D data """
        if not temporal:
            original_length = len(self.disparity_list)
            root = osp.join(self.root, 'Monkaa')
            left_images = sorted(glob(osp.join(root, self.dstype, '*/left/*.png')))
            right_images = [image_file.replace('left', 'right') for image_file in left_images]
            disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
            logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")
        else:
            root = osp.join(self.root, 'Monkaa')
            pose = sorted(glob(osp.join(root, 'pose', '*/camera_data.txt')))
            scenes = sorted(glob(osp.join(root, self.dstype, '**')))
            pose_list = []
            left_image_list = []
            right_image_list = []
            disparity_list = []
            # index by slices
            for pose, scene in zip(pose, scenes):
                poses = frame_utils.readsceneflow_pose(pose)
                left_images = sorted(glob(osp.join(scene, 'left/*.png')))
                right_images = [im.replace('left', 'right') for im in left_images]
                disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]
                assert len(left_images) == len(poses), [len(left_images), len(poses)]
                left_image_list += [left_images[i:i + frame_sample_length] for i in range(len(left_images) - frame_sample_length + 1)]
                right_image_list += [right_images[i:i + frame_sample_length] for i in range(len(right_images) - frame_sample_length + 1)]
                disparity_list += [disparity_images[i:i + frame_sample_length] for i in range(len(disparity_images) - frame_sample_length + 1)]
                pose_list += [poses[i:i + frame_sample_length] for i in range(len(poses) - frame_sample_length + 1)]

            for idx, (img1, img2, disp, pose) in enumerate(zip(left_image_list, right_image_list, disparity_list, pose_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.pose_list += [pose]
                self.intrinsic_K += [np.array(
                    [[1050., 0., 479.5],
                     [0., 1050., 269.5],
                     [0.0, 0.0, 1.0]]
                )]

    def _add_driving(self,temporal=False,frame_sample_length=1):
        """ Add FlyingThings3D data """
        if not temporal:
            original_length = len(self.disparity_list)
            root = osp.join(self.root, 'Driving')
            left_images = sorted(glob(osp.join(root, self.dstype, '*/*/*/left/*.png')))
            right_images = [image_file.replace('left', 'right') for image_file in left_images]
            disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

            for img1, img2, disp in zip(left_images, right_images, disparity_images):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
            logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")
        else:
            root = osp.join(self.root, 'Driving')
            pose = sorted(glob(osp.join(root, 'pose', '*/*/*/camera_data.txt')))
            scenes = sorted(glob(osp.join(root, self.dstype, '*/*/*')))
            pose_list = []
            left_image_list = []
            right_image_list = []
            disparity_list = []
            # index by slices
            for pose, scene in zip(pose, scenes):
                poses = frame_utils.readsceneflow_pose(pose)
                left_images = sorted(glob(osp.join(scene, 'left/*.png')))
                right_images = [im.replace('left', 'right') for im in left_images]
                disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]
                assert len(left_images) == len(poses), [len(left_images), len(poses)]
                left_image_list += [left_images[i:i + frame_sample_length] for i in range(len(left_images) - frame_sample_length + 1)]
                right_image_list += [right_images[i:i + frame_sample_length] for i in range(len(right_images) - frame_sample_length + 1)]
                disparity_list += [disparity_images[i:i + frame_sample_length] for i in range(len(disparity_images) - frame_sample_length + 1)]
                pose_list += [poses[i:i + frame_sample_length] for i in range(len(poses) - frame_sample_length + 1)]

            for idx, (img1, img2, disp, pose) in enumerate(zip(left_image_list, right_image_list, disparity_list, pose_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.pose_list += [pose]
                self.intrinsic_K += [np.array(
                    [[450.0, 0., 479.5],
                     [0., 450.0, 269.5],
                     [0.0, 0.0, 1.0]]
                )] if '15mm_focallength' in img1[0] else [np.array(
                    [[1050., 0., 479.5],
                     [0., 1050., 269.5],
                     [0.0, 0.0, 1.0]]
                )]


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training' else [osp.join(root,
                                                                                                                             'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(
            image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo',ddp=False):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo,ddp=ddp)

        image1_list = sorted(glob(osp.join(root, 'training/*_left/*/frame_*.png')))
        image2_list = sorted(glob(osp.join(root, 'training/*_right/*/frame_*.png')))
        disp_list = sorted(glob(osp.join(root, 'training/disparities/*/frame_*.png'))) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings',ddp=False):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings,ddp=ddp)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', scene_list=[], test_keywords=[], is_test=False, mode='single_frame', frame_sample_length=4,ddp=False, load_flow=False):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                         is_test=is_test, ddp=ddp, load_flow=load_flow,index_by_scene=True)
        assert os.path.exists(root)
        assert mode in ['single_frame', 'temporal']
        if mode == 'single_frame':
            image1_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/image_left/*_left.png')))
            image2_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/image_right/*_right.png')))
            disp_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/**/depth_left/*_left_depth.npy')))
            if is_test:
                _, image1_list = self.split_train_valid(image1_list, test_keywords)
                _, image2_list = self.split_train_valid(image2_list, test_keywords)
                _, disp_list = self.split_train_valid(disp_list, test_keywords)
            else:
                image1_list, _ = self.split_train_valid(image1_list, test_keywords)
                image2_list, _ = self.split_train_valid(image2_list, test_keywords)
                disp_list, _ = self.split_train_valid(disp_list, test_keywords)

            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
        elif mode == 'temporal':
            if load_flow:
                assert is_test, 'flow is only available in test mode'

            # ablation study, use all scenes and Hard and Easy
            frames_list = sorted(glob(os.path.join(root, 'TartanAir/**/**/**/P*')))

            image1_list = []
            image2_list = []
            disp_list = []
            pose_list = []
            flow_list = []

            if is_test:
                _, frames_list = self.split_train_valid(frames_list, test_keywords)
            else:
                frames_list, _ = self.split_train_valid(frames_list, test_keywords)

            for x in frames_list:
                disp_frames = sorted(glob(os.path.join(x, 'depth_left/*_left_depth.npy')))
                left_frames = sorted(glob(os.path.join(x, 'image_left/*_left.png')))
                right_frames = sorted(glob(os.path.join(x, 'image_right/*_right.png')))
                pose_frames = frame_utils.read_tartanair_extrinsic(os.path.join(x, 'pose_left.txt'), 'left')
                if load_flow:
                    flow_frames = sorted(glob(os.path.join(x.replace('TartanAir', 'TartanAir_flow'), 'flow/*_*_flow.npy')))
                    # add a fake flow for the last frame
                    flow_frames.append(flow_frames[-1])

                if is_test:  # no augmentation for test
                    argument_rate = 1
                else:  # augment the data to keep the same sampling probability for each scene (or video).
                    frame_len = len(disp_frames)
                    argument_rate = frame_len//300
                assert argument_rate>=1,['please check 300']
                for _ in range(argument_rate):
                    disp_list.append(disp_frames)
                    image1_list.append(left_frames)
                    image2_list.append(right_frames)
                    pose_list.append(pose_frames)
                    if load_flow:
                        flow_list.append(flow_frames)

            for img1, img2, disp, pose in zip(image1_list, image2_list, disp_list, pose_list):
                self.image_list += [[img1, img2]]  # (frames,2,images)
                self.disparity_list += [disp]
                self.pose_list += [pose]  # (frames,images) T
            if load_flow:
                self.flow_list = flow_list
            self.intrinsic_K = np.array([[320.0, 0, 320.0],
                                         [0, 320.0, 240.0],
                                         [0, 0, 1]])
            self.baseline = 0.25

    def split_train_valid(self, path_list, valid_keywords):
        path_list_init = path_list
        for kw in valid_keywords:
            path_list = list(filter(lambda s: kw not in s, path_list))
        train_path_list = sorted(path_list)
        valid_path_list = sorted(list(set(path_list_init) - set(train_path_list)))
        return train_path_list, valid_path_list


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', is_test=False, mode='single_frame', frame_sample_length=4, image_set='training',ddp=False, index_by_scene=False, num_frames=11):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                                    is_test=is_test, ddp=ddp, load_flow=False, index_by_scene=index_by_scene)
        assert os.path.exists(root)
        if is_test:
            if mode == 'single_frame':
                raise NotImplementedError
            else:  # temporal
                num_frames = num_frames
                scene_list = sorted(glob(os.path.join(root, image_set, 'sequences', '**')))  # imageset in 'kitti_seq/kitti2012_testings', 'kitti_seq/kitti2015_testings'
                image1_list = []
                image2_list = []
                pose_list = []
                disp_list = []
                for scene in scene_list:
                    image1_list.append(sorted(glob(os.path.join(scene, 'image_2', '*.png')))[:num_frames])
                    image2_list.append(sorted(glob(os.path.join(scene, 'image_3', '*.png')))[:num_frames])
                    pose_path = os.path.join(scene, 'orbslam3_pose.txt')
                    pose_list.append(frame_utils.read_kitti_extrinsic(pose_path)[:num_frames])
                    disp_list.append(scene)  # a fake disp_list, pass the scene path
                for idx, (img1, img2, disp, pose) in enumerate(zip(image1_list, image2_list, disp_list, pose_list)):
                    self.image_list += [[img1, img2]]  # scenes , 2, frames
                    self.disparity_list += [disp]  # scene path
                    self.pose_list += [pose]   # scenes, frames, numpy(4*4)

        else:
            if mode=='single_frame':
                image1_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_2/*_10.png')))
                image2_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_3/*_10.png')))
                disp_list = sorted(glob(
                    os.path.join(root, 'Kitti15', 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else \
                    [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)
                image1_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_0/*_10.png')))
                image2_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_1/*_10.png')))
                disp_list += sorted(
                    glob(os.path.join(root, 'Kitti12', 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else \
                    [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

                for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
            else:
                raise NotImplementedError


class KITTIraw(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/kitti_raw', is_test=False, mode='single_frame', frame_sample_length=4, ddp=False):
        super(KITTIraw, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, temporal=(mode == 'temporal'), frame_sample_length=frame_sample_length,
                         is_test=is_test, ddp=ddp, load_flow=False)
        assert os.path.exists(root)
        scenes_list = sorted(glob(os.path.join(root, '**')))
        image1_list = []
        image2_list = []
        disp_list = []
        intrinsic_list = []
        pose_list = []
        if is_test:  # index by scene
            for scene in scenes_list:
                intrinsic_path = os.path.join(scene, scene.split('/')[-1], '.txt')
                image1_list.append(sorted(glob(os.path.join(scene, 'image_2/*.png'))))
                image2_list.append(sorted(glob(os.path.join(scene, 'image_3/*.png'))))
                intrinsic_list.append(intrinsic_path)
                pose_list.append(frame_utils.read_kitti_extrinsic(os.path.join(scene, 'orbslam3_pose.txt')))
                disp_list.append([x.replace('image_2/data/', 'leastereo/data/') for x in image1_list[-1]])  # fake path
        else:  # index by image slice
            for scene in scenes_list:  # date
                intrinsic_path = os.path.join(scene, 'calib_cam_to_cam.txt')
                seqs_list = sorted(glob(os.path.join(scene, '*_sync')))
                for seq in seqs_list:  # sync
                    img1_seq = sorted(glob(os.path.join(seq, 'image_02/data/*.png')))
                    img2_seq = sorted(glob(os.path.join(seq, 'image_03/data/*.png')))
                    disp_seq = sorted(glob(os.path.join(seq, 'leastereo/data/*.png')))
                    pose_seq = frame_utils.read_kitti_extrinsic(os.path.join(seq, 'pose.txt'))  # n 4,4
                    if len(img1_seq) != len(disp_seq) or len(img1_seq) != len(img2_seq) or len(img1_seq) != len(pose_seq):
                        print(f"Warning: {seq} has different length of images, disparity or pose")
                        continue
                    intrinsic_list += [intrinsic_path]*len(img1_seq)
                    # assert len(pose_seq) == len(img1_seq) == len(disp_seq), [len(pose_seq), len(img1_seq),len(disp_seq),seq]
                    img1_seq_slices = [img1_seq[i:i + frame_sample_length] for i in range(len(img1_seq) - frame_sample_length + 1)]
                    img2_seq_slices = [img2_seq[i:i + frame_sample_length] for i in range(len(img2_seq) - frame_sample_length + 1)]
                    disp_seq_slices = [disp_seq[i:i + frame_sample_length] for i in range(len(disp_seq) - frame_sample_length + 1)]
                    pose_seq_slices = [pose_seq[i:i + frame_sample_length] for i in range(len(pose_seq) - frame_sample_length + 1)]
                    image1_list += img1_seq_slices  # (slices, frame length)
                    image2_list += img2_seq_slices  # (slices, frame length)
                    disp_list += disp_seq_slices  # (slices, frame length)
                    pose_list += pose_seq_slices  # (slices, frame length)

        self.intrinsic_K = []
        for idx, (img1, img2, disp, pose) in enumerate(zip(image1_list, image2_list, disp_list, pose_list)):
            self.image_list += [[img1, img2]]  # (slices, 2, frame length)
            self.disparity_list += [disp]  # (slices, frame length)
            self.pose_list += [pose]  # (slices, frame length)
            Pr2 = pykitti.utils.read_calib_file(intrinsic_list[idx])['P_rect_02']
            self.intrinsic_K += [np.array([[Pr2[0], 0,  Pr2[2]],
                                         [0, Pr2[5], Pr2[6]],
                                         [0,      0,     1]])]
        self.baseline = 0.54


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F',ddp=False):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury,ddp=ddp)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014":  # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(
                filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False,
                  'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
    if hasattr(args, "device") and args.device is not None:
        aug_params["device"] = args.device

    train_dataset = None
    dataset_name = args.train_dataset
    if dataset_name.startswith("middlebury_"):
        new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', ''),ddp=args.ddp)
    elif dataset_name == 'sceneflow':
        clean_dataset = SceneFlowDatasets(aug_params.copy(), dstype='frames_cleanpass',mode='temporal' if args.temporal else 'single_frame',
                                          frame_sample_length=args.frame_length,ddp=args.ddp)
        final_dataset = SceneFlowDatasets(aug_params.copy(), dstype='frames_finalpass',mode='temporal' if args.temporal else 'single_frame',
                                          frame_sample_length=args.frame_length,ddp=args.ddp)
        new_dataset = (clean_dataset * 4) + (final_dataset * 4)
        logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    elif dataset_name == 'kitti_raw':
        new_dataset = KITTIraw(aug_params,
                               mode='temporal' if args.temporal else 'single_frame',
                               frame_sample_length=args.frame_length,
                               ddp=args.ddp)
        logging.info(f"Adding {len(new_dataset)} samples from KITTI raw")
    elif 'kitti' in dataset_name:
        new_dataset = KITTI(aug_params,
                            mode='temporal' if args.temporal else 'single_frame',
                            frame_sample_length=args.frame_length,
                            ddp=args.ddp)
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    elif dataset_name == 'sintel_stereo':
        new_dataset = SintelStereo(aug_params,ddp=args.ddp) * 140
        logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
    elif dataset_name == 'falling_things':
        new_dataset = FallingThings(aug_params,ddp=args.ddp) * 5
        logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
    elif dataset_name == 'TartanAir':
        keyword_list = []
        scene_list = ['abandonedfactory', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'office', 'office2',
                      'oldtown', 'soulcity']
        part_list = ['P002', 'P007', 'P003', 'P006', 'P001', 'P042', 'P006', 'P004', 'P006', 'P008']

        for i, (s, p) in enumerate(zip(scene_list, part_list)):
            keyword_list.append(os.path.join(s, 'Easy', p))  # temporal stereo off
            keyword_list.append(os.path.join(s, 'Hard', p))
        if args.temporal:
            scale_factor = 100
        else:
            scale_factor = 1
        root = 'datasets'
        new_dataset = TartanAir(aug_params, root=root, scene_list=scene_list, test_keywords=keyword_list, mode='temporal' if args.temporal else 'single_frame',
                                frame_sample_length=args.frame_length, ddp=args.ddp) * scale_factor
        logging.info(f"Adding {len(new_dataset)} samples from TartanAir")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    train_dataset = new_dataset
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=False, num_workers=4, prefetch_factor=4, drop_last=True,
                                       sampler=train_sampler, persistent_workers=True)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, prefetch_factor=4, drop_last=True,
                                       shuffle=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
