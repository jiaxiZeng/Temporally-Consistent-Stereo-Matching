CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=30278 train_stereo.py --ddp \
--pth_name sceneflow \
--mixed_precision \
--batch_size 4 \
--train_dataset sceneflow \
--lr 0.0002 \
--num_steps 200000 \
--image_size 320 720 \
--train_iters 5 \
--valid_iters 5 \
--shared_backbone \
--saturation_range 0.0 1.4 \
--spatial_scale -0.2 0.4  \
--name sceneflow_benchmark \
--temporal \
--init_thres 0.5 \
--frame_length 2 \
--noyjitter \
--context_norm none