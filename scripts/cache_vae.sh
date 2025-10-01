IMAGENET_PATH='/root/autodl-tmp/djepa-imagenet/dataset/imagenet100/image'
CACHED_PATH='/root/autodl-tmp/djepa-imagenet/dataset/imagenet100/cache'

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}