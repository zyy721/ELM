CODE_HOME=/cpfs01/user/huanglinyan/projects/EmbodiedScene
CONFIG=/cpfs01/user/huanglinyan/projects/EmbodiedScene/lavis/projects/blip2/train/advqa_t5_ego4d.yaml

# cp -r /cpfs01/user/huanglinyan/cache/* /home/huanglinyan/.cache
# cp -r /cpfs01/user/huanglinyan/cache/* ~/.cache

# cd到bevformer根目录
cd $CODE_HOME

python -m torch.distributed.run \
    --nproc_per_node=2 \
    train.py --cfg-path $CONFIG