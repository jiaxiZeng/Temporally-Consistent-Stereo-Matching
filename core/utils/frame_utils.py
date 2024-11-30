import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import cv2
from scipy.spatial.transform import Rotation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readDispBooster(file_name):
    disp = np.load(file_name, encoding='bytes', allow_pickle=True)
    valid = disp > 0
    return disp, valid


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())


def writeFlow(filename, uv, v=None):
    """ Write optical flow_seq to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid


# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14))[..., 0]
    mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid


# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open('/'.join(file_name.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid


# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / (depth + 1e-5)
    valid = disp > 0
    return disp, valid

def readFlowTartanAir(file_path):
    data = np.load(file_path)
    mask_path = file_path.replace('flow.npy','mask.npy')
    mask = np.load(mask_path)
    mask = (mask == 0).astype(float)
    return data, maskS

def readDepthTartanAir(file_name):
    depth = np.load(file_name)
    return depth


def readFlowTartanAir(file_path):
    data = np.load(file_path)
    mask_path = file_path.replace('flow.npy', 'mask.npy')
    mask = np.load(mask_path)
    mask = (mask == 0).astype(float)
    return data, mask


def readNormalTartanAir(file_name):
    normal = np.load(file_name)
    # normal = np.concatenate((normal,-np.ones_like(normal[:1])),axis=0)
    # normal_L2 = np.sqrt(np.sum(normal**2,axis=0,keepdims=True))
    # normal = normal / normal_L2
    # normal = normal / np.linalg.norm(normal,axis=0,keepdims=True)
    return normal


def readDispMiddlebury(file_name):
    if basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    elif basename(file_name) == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


def read_tartanair_extrinsic(extrinsic_path, side='left'):
    data = []
    camera_id = {'left': 0, 'right': 1}
    with open(extrinsic_path, 'r') as fp:
        lines = fp.readlines()
    for lineid, line in enumerate(lines):
        frame = int(lineid)
        values = line.rstrip().split(' ')
        assert len(values) == 7, 'Pose must be quaterion format -- 7 params, but {} got'.format(len(values))
        pose = np.array([float(values[i]) for i in range(len(values))])
        tx, ty, tz, qx, qy, qz, qw = pose
        R = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
        t = np.array([tx, ty, tz])
        T = np.eye(4)
        T[:3, :3] = R.transpose()
        T[:3, 3] = -R.transpose().dot(t)
        # ned(z-axis down) to z-axis forward
        m_correct = np.zeros_like(T)
        m_correct[0, 1] = 1
        m_correct[1, 2] = 1
        m_correct[2, 0] = 1
        m_correct[3, 3] = 1

        # m_correct
        T = np.matmul(m_correct, T)
        data.append(T)
        lineid += 1

    return data


def readsceneflow_pose(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        tokens = line.split()
        if len(tokens)>0 and tokens[0]=='L':
            pose = np.array([float(x) for x in tokens[1:]]).reshape(4, 4)
            poses.append(np.linalg.inv(pose))  # cam2wolrd->world2cam
    return poses


def read_kitti_extrinsic(extrinsic_path):
    data = []
    with open(extrinsic_path, 'r') as fp:
        lines = fp.readlines()
    for lineid, line in enumerate(lines):
        values = line.rstrip().split(' ')
        assert len(values) == 12, 'Pose must be quaterion format -- 12 params, but {} got'.format(len(values))
        pose = np.array([float(values[i]) for i in range(len(values))]).reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        data.append(np.linalg.inv(pose))  # cam2wolrd->world2cam
    return data  # frames,4,4


if __name__ == '__main__':
    import skimage

    normal = readNormalTartanAir('/data/TartanAir_normal/abandonedfactory/abandonedfactory/Easy/P000/normal_left/000030_left_normal.npy')
    depth = readDepthTartanAir('/data/TartanAir/abandonedfactory/abandonedfactory/Easy/P000/depth_left/000003_left_depth.npy')
    print(depth)
    skimage.io.imsave('normal.png', np.uint8(normal.transpose(1, 2, 0) * 128))
    skimage.io.imsave('depth.png', np.uint8(depth * 60))
