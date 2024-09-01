CONFIG=lavis/projects/blip2/train/advqa_t5_elm_video_debug_rec_img.yaml

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=10041 \
    scripts/train.py --cfg-path $CONFIG