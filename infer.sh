#bin/bash
CUDA_VISIBLE_DEVICES=4
CUBLAS_WORKSPACE_CONFIG=:4096:8 python infer_stereo.py \
--mixed_precision \
--valid_iters 5 \
--shared_backbone \
--temporal \
--context_norm none \
--restore_ckpt ./checkpoints/tartanair.pth \
--left_img_dir /home/smbu/jiaxi/workspace/InstantSplat/output_infer/sora/women_mirror/4_views/interp/ours_1000/renders \
--right_img_dir /home/smbu/jiaxi/workspace/InstantSplat/output_infer/sora/women_mirror/4_views/interp_right/ours_1000/renders \
--pose_path /home/smbu/jiaxi/workspace/InstantSplat/output_infer/sora/women_mirror/4_views/pose/ours_1000/pose_interpolated.npy \
--camera_path /home/smbu/jiaxi/workspace/InstantSplat/assets/sora/women_mirror/sparse_4/0/cameras.txt \
--output_dir /home/smbu/jiaxi/workspace/InstantSplat/output_infer/sora/women_mirror/4_views/interp_disp/ours_1000 \
