import math
import os
import time
from glob import glob
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch_fidelity
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torchvision import transforms

import util.ema_sched as ema_sched
import util.lr_sched as lr_sched
import util.misc as misc
import util.wd_sched as wd_sched


def update_ema(target, source, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target.parameters(), source.parameters()):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    model_without_ddp, ema_model_without_ddp,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer,
                    args):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    num_batches = len(data_loader)  # type: ignore
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr = lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / num_batches + epoch, args)
        wd = wd_sched.adjust_weight_decay(
            optimizer, data_iter_step / num_batches + epoch, args
        )

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure FP32 in VAE.
        tic = time.time()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
            if not args.use_cached:
                # Map input images to latent space + normalize latents:
                posterior = vae.encode(samples)
            else:
                posterior = DiagonalGaussianDistribution(samples)

            x = posterior.sample().sub_(args.shift).mul_(args.scaling_factor)

        # forward
        with torch.cuda.amp.autocast(dtype=args.precision):
            if args.jepa_weight > 0.:
                with torch.no_grad():
                    ema_model_without_ddp.eval()
                    ema_x = ema_model_without_ddp.forward_ema_encoder(
                        x, labels)
            else:
                ema_x = None

            diffusion_loss, jepa_loss = model(x, labels, ema_x=ema_x)

        loss = diffusion_loss * args.diffusion_weight + jepa_loss * args.jepa_weight

        loss_value_reduce = misc.all_reduce_mean(loss.item())

        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters())
        optimizer.zero_grad()
        if math.isnan(loss_value_reduce):
            print("loss is nan")
            break
        torch.cuda.synchronize()

        ema_rate = ema_sched.adjust_ema_decay(
            data_iter_step / num_batches + epoch, args)
        update_ema(ema_model_without_ddp, model_without_ddp, rate=ema_rate)

        toc = time.time()
        images_per_second = len(x) / (toc-tic)

        metric_logger.update(ImPerSec=misc.all_reduce_sum(images_per_second))

        metric_logger.update(loss=loss_value_reduce)
        if args.diffusion_weight > 0.:
            metric_logger.update(diffusion_loss=diffusion_loss.item())
        if args.jepa_weight > 0.:
            metric_logger.update(jepa_loss=jepa_loss.item())

        metric_logger.update(lr=lr)

        if args.diffusion_weight > 0.:
            diffusion_value_reduce = misc.all_reduce_mean(
                diffusion_loss.item())
        if args.jepa_weight > 0.:
            jepa_value_reduce = misc.all_reduce_mean(jepa_loss.item())
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / num_batches + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            if args.diffusion_weight > 0.:
                log_writer.add_scalar(
                    "diffusion_loss", diffusion_value_reduce, epoch_1000x)
            if args.jepa_weight > 0.:
                log_writer.add_scalar(
                    "jepa_loss", jepa_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar("ema", ema_rate, epoch_1000x)
            log_writer.add_scalar("wd", wd, epoch_1000x)

            if grad_norm is not None:
                log_writer.add_scalar(
                    "grad_norm", grad_norm.item(), epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    image = transform(image)
    return image


def evaluate(model, vae, args, epoch, batch_size=16, log_writer=None, cfg_scale=1.0, temperature=0.95, num_iter=256, use_ema=True):
    model.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    
    save_folder = os.path.join(args.output_dir,
                            f"ariter{num_iter}-diffsteps{args.num_sampling_steps}-temp{temperature}-{args.cfg_schedule}cfg_scale{cfg_scale}-image{args.num_images}-epoch{epoch}")
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"

    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    class_num = args.class_num
    # number of images per class must be the same
    assert args.num_images % class_num == 0
    class_label_gen_world = np.arange(
        0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                           world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        start_time = time.time()

        # generation
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=args.precision):
            sampled_tokens = model.sample_tokens(
                bsz=batch_size, num_iter=num_iter, cfg_scale=cfg_scale,
                cfg_schedule=args.cfg_schedule, labels=labels_gen,
                temperature=temperature,
                time_shifting_factor=args.time_shifting_factor)

            sampled_images = vae.decode(
                sampled_tokens / args.scaling_factor + args.shift)

        # measure speed after the first generation batch
        if i >= 1:
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(
                gen_img_cnt, used_time, used_time / gen_img_cnt))

        dist.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * \
                sampled_images.size(0) * world_size + \
                local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(
                np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            filename = os.path.join(
                save_folder, '{}.png'.format(str(img_id).zfill(6)))
            cv2.imwrite(filename, gen_img)  # type: ignore

    dist.barrier()

    # compute FID and IS
    if log_writer is not None and torch_fidelity is not None:
        # log writer will not be set for `--evaluate`
        # NOTE: This only be used during training to obtain FID and IS for imagenet with size 256.
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError(
                "Only support 256x256 with `torch_fidelity` to evaluate online. Please use offline evaluation.")
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
            postfix = postfix + "_ema"
        postfix = postfix + "_cfg{}".format(cfg_scale)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(
            fid, inception_score))

    if args.samples_log_on_tb > 0 and log_writer is not None:
        images_to_log = sorted(
            glob(os.path.join(save_folder, "*.png")))[:args.samples_log_on_tb]
        images = [load_image(image_path) for image_path in images_to_log]

        for i, image in enumerate(images):
            filename = os.path.basename(images_to_log[i])
            log_writer.add_image(f"Samples/{filename}", image, epoch)

    dist.barrier()

def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args):
    def to_numpy(x):
        return x.detach().cpu().numpy()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 200

    for samples, _, paths in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path,
                     moments=to_numpy(moments[i]),
                     moments_flip=to_numpy(moments_flip[i])
                     )

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()