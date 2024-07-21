from pykitti.utils import load_oxts_packets_and_poses,transform_from_rot_trans,read_calib_file
from glob import glob
import numpy as np
import os
root ='/data1/kitti_raw'
scene_list = sorted(glob(os.path.join(root, '**')))
for scene in scene_list:
    print(scene)
    seqs_list = sorted(glob(os.path.join(scene, '*_sync')))
    c2c = read_calib_file(os.path.join(scene, 'calib_cam_to_cam.txt'))
    Rrect0 = transform_from_rot_trans(c2c['R_rect_00'], np.zeros(3))
    i2v = read_calib_file(os.path.join(scene, 'calib_imu_to_velo.txt'))
    i2v = transform_from_rot_trans(i2v['R'], i2v['T'])
    v2c = read_calib_file(os.path.join(scene, 'calib_velo_to_cam.txt'))
    v2c = transform_from_rot_trans(v2c['R'], v2c['T'])
    c2i = np.linalg.inv(np.dot(v2c, i2v))
    for seq in seqs_list:
        # if seq !='/data1/kitti_raw/2011_09_28/2011_09_28_drive_0002_sync':
        #     continue
        frame_list = sorted(glob(os.path.join(seq, 'oxts/data/*.txt')))
        oxts = load_oxts_packets_and_poses(frame_list)
        pose_str = ''
        camera_pose_first_inv = None
        pose_imu_first_inv = None
        w2c_0 = None
        imupose_init_inv =None
        for i in range(len(oxts)):
            imupose = oxts[i].T_w_imu  # imu pose
            if imupose_init_inv is None:
                imupose_init_inv = np.linalg.inv(imupose)
            i2w = imupose_init_inv.dot(imupose)
            c2w = np.dot(i2w, c2i)  # c2w:inv(cam pose)  # 第一帧的相机坐标系转换到世界坐标系
            if w2c_0 is None:
                w2c_0 = np.linalg.inv(c2w)  # w2c:cam pose
            c2c_0 = w2c_0.dot(c2w)  # camera ->camera t0
            # pose = pose.dot(pose_first_inv)  # 将第一帧的相机坐标系，设置为世界坐标系，转换到当前帧的坐标系
            # # # 记录的pose是从相机坐标系到世界坐标系，与temporal stereo保持一致
            pose = c2c_0
            pose_str_line = ' '.join(format(num, '.9f') for row in pose[:-1] for num in row)
            pose_str = pose_str + pose_str_line.lstrip() + '\n'
        with open(os.path.join(seq, 'pose.txt'), 'w') as f:
            f.write(pose_str)