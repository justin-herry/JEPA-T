IMAGENET_PATH='/PATH/TO/ILSVRC2012'

model=djepa_huge

# NOTE: This batch size is for a single GPU. You must keep the total batch size equals to 2048 during training.
batch_size=32

vae=KL16_MAR
patch_size=1
diffloss="gauss"

exp_name=${model}_${vae}_${patch_size}_${diffloss}
results_dir=exps/$exp_name
mkdir -p ${results_dir}

torchrun \
--nnodes=1 \
--nproc_per_node=4 \
--node_rank=0 \
--master_addr=localhost \
--master_port=7778 \
main_djepa.py \
--img_size 256 \
--vae ${vae} \
--patch_size ${patch_size} \
--model ${model} \
--epochs 320 \
--warmup_epochs 100 \
--batch_size ${batch_size} \
--blr 1.0e-4 \
--diffusion_batch_mul 4 \
--output_dir ${results_dir} \
--data_path ${IMAGENET_PATH} \
--online_eval \
--cfg_scale 1.0 \
--temperature 0.94 \
--num_iter 64 \
--cfg_schedule linear \
--eval_freq 10 \
--precision bf16 \
--diffloss ${diffloss} \
--diffusion_weight 1.0 \
--jepa_weight 1.0 \
--eval_bsz 256 \
--grad_checkpointing \
--adamw_eps "1e-6" \
--num_images 1000 \
--compile \
--qk_norm 

# If you want to evaluate the model, add the following arguments.
# --evaluate
# --resume checkpoints/djepa_huge.pth

# If you want to use cached data, add the following arguments.
# --use_cached 
# --cached_path PATH/TO/CACHED/ILSVRC2012

# If you want to do classifier-guided free sampling, adjust following arguments.
# Different modal has different hyperparameters, please refert to our paper for details.
# --cfg_scale 3.0
# --temperature 0.98