import argparse
import copy
import datetime
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

import models
import util.misc as misc
from engine.jepat import evaluate, train_one_epoch
from models.pretrained_vae import get_pretrained_vae
from util.crop import center_crop_arr
from util.loader import CachedFolder
from util.misc import NativeScalerWithGradNormCount as NativeScaler

warnings.simplefilter("ignore")
torch._dynamo.config.suppress_errors = True  # type: ignore

def get_args_parser():
    parser = argparse.ArgumentParser(
        'D-JEPA training script', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus)')
    parser.add_argument('--epochs', default=400, type=int,
                        help='Total number of epochs to run.')

    # Model parameters
    parser.add_argument('--model', default='JEPAT_large', choices=["JEPAT_base", "JEPAT_large", "JEPAT_huge"], type=str, metavar='MODEL',
                        help='Name of model to train.')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='Input image size.')
    parser.add_argument('--vae', default="KL16_MAR", choices=["KL16_MAR"],
                        help='VAE encoder to use. KL16_MAR from MAR.')
    parser.add_argument("--local_diffusers_model_root",
                        default="pretrained_models",
                        help='Root directory for local diffusers models.')
    parser.add_argument('--patch_size', default=1, choices=[1, 2, 4], type=int,
                        help='Number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='Number of autoregressive iterations to generate an image.')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate.')
    parser.add_argument('--cfg_scale', default=1.0, type=float,
                        help="Classifier-free guidance scale.")
    parser.add_argument('--cfg_schedule', default="linear", choices=["linear", "constant"], type=str,
                        help='Configuration schedule type.')
    parser.add_argument('--label_drop_prob', default=0.1, type=float,
                        help='Label dropout probability.')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Evaluation frequency. The model will be saved at each evaluation point.')
    parser.add_argument('--save_last_freq', type=int,
                        default=5, help='Frequency to save the latest model.')
    parser.add_argument('--online_eval', action='store_true',
                        help='Enable online evaluation during training.')
    parser.add_argument('--evaluate', action='store_true',
                        help='If set, only execute evaluation.')
    parser.add_argument('--eval_bsz', type=int, default=64,
                        help='Batch size for generation during evaluation.')
    parser.add_argument("--samples_log_on_tb", default=4, type=int,
                        help="Number of generated samples to log on TensorBoard.")
    parser.add_argument('--temperature', default=0.95, type=float, 
                        help='Diffusion loss sampling temperature for Gauss diffusion. (Default: 0.95)')
    parser.add_argument("--time_shifting_factor", default=1.0, type=float,
                        help="Diffusion loss sampling time-shifting factor for flow match.")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay (default: 0.02).')
    parser.add_argument("--adamw_eps", default=1e-8, type=float,
                        help="AdamW optimizer epsilon (default: 1e-8).")

    parser.add_argument('--grad_checkpointing', action='store_true',
                        default=False, help="Enable gradient checkpointing to save memory at the cost of computation.")
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute).')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256.')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower learning rate bound for cyclic schedulers that might reach to zero.')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule.')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='Number of warmup epochs for learning rate.')
    parser.add_argument('--ema_rate', default=0.9999, type=float,
                        help='Exponential moving average rate.')

    # D-JEPA params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio for masking.')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clipping threshold.')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Attention dropout rate.')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='Projection dropout rate.')
    parser.add_argument('--buffer_size', type=int, default=64,
                        help="Buffer size for caching latents.")

    # Diffusion Loss parameters
    parser.add_argument('--num_sampling_steps', type=str, default="100",
                        help='Number of diffusion steps for sampling.')
    parser.add_argument('--diffusion_batch_mul', type=int, default=1,
                        help='Diffusion batch multiplier.')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to dataset.')
    parser.add_argument('--class_num', default=1000, type=int,
                        help='Number of classes in the dataset.')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs, provide empty string for no saving.')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Directory to save TensorBoard logs.')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing.')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for initialization.')
    parser.add_argument('--resume', default='',
                        help='Path to checkpoint file to resume training.')
    parser.add_argument("--resume_opt_state", default=False,
                        action="store_true", help="Resume optimizer state.")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch for resuming training.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers.')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='Disable pinning of CPU memory in DataLoader.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank for distributed training.')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Set this flag for distributed training on ITP.')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training.')

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp16",
                        help="Model training and inference precision (Default: fp16).")
    parser.add_argument("--compile", default=False,
                        action="store_true", help="Compile the model for better performance.")

    # caching latents
    parser.add_argument('--use_cached', action='store_true',
                        dest='use_cached', default=False, help='Use cached latents.')
    parser.add_argument('--cached_path', default='',
                        help='Path to cached latents when using cache.')

    # hyper settings about d-jepa
    parser.add_argument("--diffusion_weight", default=1.0, type=float,
                        help='Weight for diffusion loss.')
    parser.add_argument("--jepa_weight", default=0., type=float,
                        help='Weight for JEPA loss.')
    parser.add_argument("--diffloss", default="gauss", choices=[
                        "gauss", "flow", "none"], help="Type of diffusion loss to use. Options are gauss and flow.")
    parser.add_argument("--jepaloss", default="jepa",
                        choices=["jepa", "none"], help="Type of JEPA loss to use.")
    parser.add_argument("--qk_norm", action="store_true", default=False,
                        help="Add qk normalization in attention to stabilize training.")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.seed is not None:
        seed = args.seed + misc.get_rank()
        print(f"Set global seed as {seed}.")
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        print("Using random seed.")

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # define the vae and JEPAT model
    vae = get_pretrained_vae(args)

    args.precision = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.precision]
    print(f"Precision: {args.precision}")
    model = models.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        # only define loss if necessary.
        diffloss=args.diffloss if args.diffusion_weight > 0 else "none",
        # only define loss if necessary.
        jepaloss=args.jepaloss if args.jepa_weight > 0 else "none",
        grad_checkpointing=args.grad_checkpointing,
        qk_norm=args.qk_norm,
    )
    if args.compile:
        model = torch.compile(model)

    # following timm: set wd as 0 for bias and norm layers
    print("Trainable parameters: ")
    print("\n".join([f"{k}: {v:,}" for k, v in model.count_params().items()]))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            # dit block will contains same unused parameters.
            find_unused_parameters=False,
            bucket_cap_mb=256,
            gradient_as_bucket_view=True,
            broadcast_buffers=False)
        model_without_ddp = model.module

    ema_model_without_ddp = copy.deepcopy(model_without_ddp)

    # no weight decay on bias, norm layers, and diffloss MLP
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True, eps=args.adamw_eps)
    loss_scaler = NativeScaler()

    # resume training
    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        ema_model_without_ddp.load_state_dict(
            checkpoint['model_ema'], strict=True)
        print("Resume checkpoint %s" % args.resume)
        args.start_epoch = checkpoint['epoch'] + 1

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and args.resume_opt_state:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(ema_model_without_ddp, vae, args, 0, batch_size=args.eval_bsz,
                 log_writer=None, cfg_scale=args.cfg_scale, temperature=args.temperature, num_iter=args.num_iter, use_ema=True)
        return

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(
            pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(
            args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # type: ignore

        train_one_epoch(
            model, vae,
            model_without_ddp, ema_model_without_ddp,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # save checkpoint
        if (epoch + 1) % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            # save for resume training.
            misc.save_model(args=args,
                            model_without_ddp=model_without_ddp,
                            ema_model_without_ddp=ema_model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            epoch_name="last")

        # online evaluation
        if args.online_eval and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            # save for visualization
            misc.save_model(args=args,
                            model_without_ddp=model_without_ddp,
                            ema_model_without_ddp=ema_model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            epoch_name=f"epoch_{epoch}")
            torch.cuda.empty_cache()
            evaluate(
                ema_model_without_ddp,
                vae,
                args,
                epoch,
                batch_size=args.eval_bsz,
                log_writer=log_writer,
                cfg_scale=args.cfg_scale, temperature=args.temperature, num_iter=args.num_iter,
                use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)