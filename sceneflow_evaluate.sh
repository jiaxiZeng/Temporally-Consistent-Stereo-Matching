CUBLAS_WORKSPACE_CONFIG=:4096:8 python evaluate_stereo.py --dataset things \
--mixed_precision \
--valid_iters 5 \
--shared_backbone \
--temporal \
--context_norm none \
--restore_ckpt ./checkpoints/sceneflow.pth