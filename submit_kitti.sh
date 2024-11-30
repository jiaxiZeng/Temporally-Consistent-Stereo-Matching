CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python evaluate_stereo.py --dataset kitti \
--mixed_precision \
--valid_iters 5 \
--shared_backbone \
--temporal \
--context_norm none \
--restore_ckpt ./checkpoints/kitti_raw.pth
