import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from accelerate import Accelerator

from common import CC2017_Dataset, Neuroclips, RidgeRegression
from Perception import Perception_Reconstruction, Inception_Extension

# # SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('generative_models/')
# from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils


def verify_necessary_files(args, verbose: bool = True):
    '''
    Before training or loading data, assure that all necessary data and model files are available.
    '''
    if verbose is True:
        print('\nVerifying if we have all necessary data and model files...')
    assert os.path.isfile(f'{args.data_dir}/subj0{args.subj}_train_fmri.pt')
    assert os.path.isfile(f'{args.data_dir}/subj0{args.subj}_test_fmri.pt')
    assert os.path.isfile(f'{args.data_dir}/GT_train_3fps.pt')
    assert os.path.isfile(f'{args.data_dir}/GT_test_3fps.pt')
    assert os.path.isfile(f'{args.model_dir}/sd_image_var_autoenc.pth')
    assert os.path.isfile(f'{args.model_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    if verbose is True:
        print('Verified: all necessary files are available!\n')
    return

def save_ckpt(ckpt_path):
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f'\n---saved {ckpt_path} ckpt!---\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Configuration')
    parser.add_argument(
        '--subj', type=int, default=1, choices=[1,2,3],
        help='Validate on which subject?',
    )
    parser.add_argument(
        '--use_prior',action=argparse.BooleanOptionalAction, default=False,
        help='whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=10,
        help='Batch size can be increased by 10x if only training retreival submodule and not diffusion prior',
    )
    parser.add_argument(
        '--mixup_pct', type=float, default=.33,
        help='proportion of way through training when to switch from BiMixCo to SoftCLIP',
    )
    parser.add_argument(
        '--blurry_recon',action=argparse.BooleanOptionalAction, default=True,
        help='whether to output blurry reconstructions',
    )
    parser.add_argument(
        '--blur_scale', type=float, default=.5,
        help='multiply loss from blurry recons by this number',
    )
    parser.add_argument(
        '--clip_scale', type=float, default=1.,
        help='multiply contrastive loss by this number',
    )
    parser.add_argument(
        '--prior_scale', type=float, default=30,
        help='multiply diffusion prior loss by this',
    )
    parser.add_argument(
        '--use_image_aug',action=argparse.BooleanOptionalAction, default=False,
        help='whether to use image augmentation',
    )
    parser.add_argument(
        '--num_epochs', type=int, default=150,
        help='number of epochs of training',
    )
    parser.add_argument(
        '--n_blocks', type=int, default=4,
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=4096,
    )
    parser.add_argument(
        '--lr_scheduler_type', type=str, default='cycle',choices=['cycle','linear'],
    )
    parser.add_argument(
        '--ckpt_saving',action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    parser.add_argument(
        '--max_lr', type=float, default=3e-4,
    )
    parser.add_argument(
        '--use_text',action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        '--fps', type=int, default=3,
    )
    parser.add_argument(
        '--data_dir', type=str, default='$NeuroClips_ROOT/data/',
    )
    parser.add_argument(
        '--model_dir', type=str, default='$NeuroClips_ROOT/checkpoints/',
    )
    parser.add_argument(
        '--neuroclips_model_dir', type=str, default='$NeuroClips_ROOT/outputs/checkpoints/',
    )

    args = parser.parse_args()

    ### Multi-GPU config ###
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print('LOCAL RANK ', local_rank)

    data_type = torch.float16 # change depending on your mixed_precision

    # First use 'accelerate config' in terminal and setup using deepspeed stage 2 with CPU offloading!
    accelerator = Accelerator(split_batches=False, mixed_precision='fp16')

    print('PID of this process =', os.getpid())
    device = accelerator.device
    print('device:', device)
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    num_devices = torch.cuda.device_count()
    if num_devices == 0 or not distributed:
        num_devices = 1
    num_workers = num_devices
    print(accelerator.state)

    print('distributed =', distributed, 'num_devices =', num_devices, 'local rank =', local_rank, 'world size =', world_size, 'data_type =', data_type)
    print = accelerator.print # only print if local_rank=0

    NeuroClips_ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    args.data_dir = args.data_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)
    args.model_dir = args.model_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)
    args.neuroclips_model_dir = args.neuroclips_model_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)

    verify_necessary_files(args)

    # seed all random functions
    utils.seed_everything(args.seed)

    model_name = f'video_subj0{args.subj}_low_level.pth'

    if args.use_image_aug or args.blurry_recon:
        import kornia
        from kornia.augmentation.container import AugmentationSequential

    if args.use_image_aug:
        img_augment = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
            same_on_batch=False,
            data_keys=['input'],
        )

    subj_list = [args.subj]

    if args.subj == 1:
        voxel_length = 13447
    elif args.subj == 2 :
        voxel_length = 14828
    elif args.subj == 3 :
        voxel_length = 9114

    clip_seq_dim = 256
    clip_emb_dim = 1664
    seq_len = 1

    voxel_train = torch.load(f'{args.data_dir}/subj0{args.subj}_train_fmri.pt', map_location='cpu')
    voxel_test = torch.load(f'{args.data_dir}/subj0{args.subj}_test_fmri.pt', map_location='cpu')
    voxel_test = torch.mean(voxel_test, dim = 1).unsqueeze(1)
    num_voxels_list = [voxel_train.shape[-1]]
    print('Loaded all train fMRI frames to cpu!', voxel_train.shape)
    print('Loaded all test fMRI frames to cpu!', voxel_test.shape)

    train_images = torch.load(f'{args.data_dir}/GT_train_3fps.pt', map_location='cpu')
    test_images = torch.load(f'{args.data_dir}/GT_test_3fps.pt', map_location='cpu')
    print('Loaded all train image frames to cpu!', train_images.shape)
    print('Loaded all test image frames to cpu!', test_images.shape)

    train_dl = {}
    train_dataset = CC2017_Dataset(voxel_train, train_images, istrain=True)
    test_dataset = CC2017_Dataset(voxel_test, test_images, istrain=False)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=0, drop_last=False)

    num_samples_per_epoch = len(train_dataset) // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // args.batch_size
    print('batch_size =', args.batch_size, 'num_iterations_per_epoch =', num_iterations_per_epoch, 'num_samples_per_epoch =', num_samples_per_epoch)

    if args.blurry_recon:
        from diffusers import AutoencoderKL
        from autoencoder.convnext import ConvnextXL

        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        ckpt = torch.load(f'{args.model_dir}/sd_image_var_autoenc.pth')
        autoenc.load_state_dict(ckpt)

        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)
        utils.count_params(autoenc)

        cnx = ConvnextXL(f'{args.model_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=['input'],
        )

    model = Neuroclips()

    model.ridge1 = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
    utils.count_params(model.ridge1)
    utils.count_params(model)

    model.backbone = Perception_Reconstruction(h=args.hidden_dim,
                                               in_dim=args.hidden_dim,
                                               seq_len=seq_len,
                                               n_blocks=args.n_blocks,
                                               clip_size=clip_emb_dim,
                                               out_dim=clip_emb_dim*clip_seq_dim,
                                               blurry_recon=args.blurry_recon,
                                               clip_scale=args.clip_scale)
    model.fmri = Inception_Extension(h=256,
                                     in_dim=voxel_length,
                                     out_dim=voxel_length,
                                     expand=args.fps*2,
                                     seq_len=seq_len)

    utils.count_params(model.backbone)
    utils.count_params(model.fmri)
    utils.count_params(model)

    for param in model.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.num_epochs * num_iterations_per_epoch))
        print('total_steps', total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2/args.num_epochs
        )

    print('\nDone with model preparations!')
    num_params = utils.count_params(model)

    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test = 0
    torch.cuda.empty_cache()

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)
    # leaving out test_dl since we will only have local_rank 0 device do evals

    print(f'{model_name} starting with epoch {epoch} / {args.num_epochs}')
    progress_bar = tqdm(range(epoch, args.num_epochs), ncols=1200, disable=(local_rank!=0))
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))

    if num_devices != 0 and distributed:
        model = model.module

    for epoch in progress_bar:
        model.train()

        fwd_percent_correct = 0.
        bwd_percent_correct = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.

        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0.

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        step = 0
        for train_i, (voxel, image) in enumerate(train_dl):
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                #text = text_iters[train_i].detach()
                image = image.reshape(len(image) * args.fps * 2, 3, 224, 224).to(device)
                voxel = voxel[:, epoch % 2, :].half().unsqueeze(1).to(device)

                if args.use_image_aug:
                    image = img_augment(image)

                voxel = model.fmri(voxel).unsqueeze(1)
                voxel_ridge = model.ridge1(voxel, 0)
                blurry_image_enc_ = model.backbone(voxel_ridge, time = args.batch_size * args.fps * 2)

                if args.blurry_recon:
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        voxel, perm, betas, select = utils.mixco(voxel)
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1] * (len(image_enc.shape)-1)
                        image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                            image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                    image_norm = (image - mean)/std
                    image_aug = (blur_augs(image) - mean)/std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_total += cont_loss.item()

                    loss += (loss_blurry + 0.1*cont_loss) * args.blur_scale #/.18215

                if args.blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        random_samps = np.random.choice(np.arange(len(image)), size=len(image), replace=False)
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        blurry_pixcorr += pixcorr.item()

                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()
                step += 1
                print(f'Training epoch: {epoch}, sample: {step * args.batch_size}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item():.4f}, loss_mean: {np.mean(losses[-(train_i+1):]):.4f}')

        model.eval()

        if local_rank==0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, (voxel, image) in enumerate(test_dl):
                    # all test samples should be loaded per batch such that test_i should never exceed 0

                    if test_image is None:
                        voxel = voxel.half()
                        image = image.reshape(len(image) * args.fps * 2, 3, 224, 224).cpu()

                    loss=0.

                    voxel = voxel.to(device)
                    image = image.to(device)

                    #clip_target = clip_img_embedder(image.float())

                    test_fwd_percent_correct = 0.
                    test_bwd_percent_correct = 0.
                    text_fwd_percent_correct = 0.

                    voxel = model.fmri(voxel).unsqueeze(1)
                    voxel_ridge = model.ridge1(voxel,0)
                    blurry_image_enc_ = model.backbone(voxel_ridge, time=40*args.fps*2)

                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    #random_samps = np.random.choice(np.arange(len(image)), size=len(image)//6, replace=False)

                    if args.blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image, blurry_recon_images)
                        test_blurry_pixcorr += pixcorr.item()

                        print('PixCorr:', pixcorr.item())
                        test_losses.append(pixcorr.item())

                print('-------------------------')

        # Save model checkpoint and reconstruct
        if test_blurry_pixcorr / len(test_dl) > best_test:
            best_test = test_blurry_pixcorr / len(test_dl)
            print('new best test correlation:', best_test)
            save_ckpt(f'{args.neuroclips_model_dir}/{model_name}')
        else:
            print('not best', test_blurry_pixcorr / len(test_dl), 'best test correlation is', best_test)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

    print('\n===Finished!===\n')
