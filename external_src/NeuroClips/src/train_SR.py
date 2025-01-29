import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from accelerate import Accelerator

from common import CC2017_Dataset, CLIPProj, Neuroclips, RidgeRegression
from Semantic import Semantic_Reconstruction

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
import_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(import_dir, 'generative_models/'))
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder from OpenCLIP

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
    assert os.path.isfile(f'{args.data_dir}/GT_train_caption_emb.pt')
    assert os.path.isfile(f'{args.data_dir}/GT_test_caption_emb.pt')
    assert os.path.isfile(f'{args.data_dir}/coco_tokens_avg_proj.pth')
    assert os.path.isfile(f'{args.model_dir}/sd_image_var_autoenc.pth')
    assert os.path.isfile(f'{args.model_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    assert os.path.isfile(f'{args.model_dir}/mindeyev2_final_subj01_pretrained_40sess_24bs_last.pth')
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

def load_ckpt(ckpt_path, load_lr=True, load_optimizer=True, load_epoch=True, strict=True, multisubj_loading=False):
    print(f'\n---loading {ckpt_path} ckpt---\n')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()['epoch'] = checkpoint['epoch']
        print('Epoch', epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Configuration')
    parser.add_argument(
        '--subj', type=int, default=1, choices=[1,2,3],
        help='Validate on which subject?',
    )
    parser.add_argument(
        '--use_prior', action=argparse.BooleanOptionalAction, default=False,
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
        '--blurry_recon', action=argparse.BooleanOptionalAction, default=False,
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
        '--use_image_aug', action=argparse.BooleanOptionalAction, default=False,
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
        '--lr_scheduler_type', type=str, default='cycle', choices=['cycle','linear'],
    )
    parser.add_argument(
        '--ckpt_saving', action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    parser.add_argument(
        '--max_lr', type=float, default=3e-4,
    )
    parser.add_argument(
        '--use_text', action=argparse.BooleanOptionalAction, default=False,
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

    if args.use_prior:
        model_name_prior = f'video_subj0{args.subj}_SR_backbone.pth'
        model_name = f'video_subj0{args.subj}_SR.pth'
    else:
        model_name = f'video_subj0{args.subj}_SR_backbone.pth'

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
    text_scale = 0.25
    text_scale_prior = 1.0

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

    train_text = torch.load(f'{args.data_dir}/GT_train_caption_emb.pt', map_location='cpu')
    test_text = torch.load(f'{args.data_dir}/GT_test_caption_emb.pt', map_location='cpu')
    print('Loaded all train image captions to cpu!', train_text.shape)
    print('Loaded all test image captions to cpu!', test_text.shape)

    train_dl = {}
    train_dataset = CC2017_Dataset(voxel_train, train_images, train_text, istrain=True)
    test_dataset = CC2017_Dataset(voxel_test, test_images, test_text, istrain=False)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=0, drop_last=False)

    num_samples_per_epoch = len(train_dataset) // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // args.batch_size
    print('batch_size =', args.batch_size, 'num_iterations_per_epoch =', num_iterations_per_epoch, 'num_samples_per_epoch =', num_samples_per_epoch)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch='ViT-bigG-14',
        version='laion2b_s39b_b160k',
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

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

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=['input'],
        )

    model = Neuroclips()

    model.backbone = Semantic_Reconstruction(h=args.hidden_dim,
                                             in_dim=args.hidden_dim,
                                             seq_len=seq_len,
                                             n_blocks=args.n_blocks,
                                             clip_size=clip_emb_dim,
                                             out_dim=clip_emb_dim*clip_seq_dim,
                                             blurry_recon=args.blurry_recon,
                                             clip_scale=args.clip_scale)
    utils.count_params(model.backbone)
    utils.count_params(model)

    if args.use_prior:
        from Semantic import PriorNetwork, BrainDiffusionPrior

        prior_network = PriorNetwork(
                dim=clip_emb_dim,
                depth=6,
                dim_head=52,
                heads=clip_emb_dim // 52,
                causal=False,
                num_tokens=clip_seq_dim,
                learned_query_mode='pos_emb',
            )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=clip_emb_dim,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        utils.count_params(model.diffusion_prior)
        utils.count_params(model)

    if not args.use_prior:
        print('\n---Not using prior. Resuming from mindeyev2 ckpt---\n')

        # Initialize the backbone with MindEye2.
        checkpoint = torch.load(f'{args.model_dir}/mindeyev2_final_subj01_pretrained_40sess_24bs_last.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

        model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
        model.clipproj = CLIPProj()
        utils.count_params(model.ridge)
        utils.count_params(model)

    else:
        print('\n---Using prior. Resuming from our prior training ckpt---\n')

        # Initialize the backbone with MindEye2.
        checkpoint = torch.load(f'{args.model_dir}/mindeyev2_final_subj01_pretrained_40sess_24bs_last.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

        # Initialize the regression and projector from prior training.
        model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
        model.clipproj = CLIPProj()
        utils.count_params(model.ridge)
        utils.count_params(model)

        checkpoint = torch.load(f'{args.neuroclips_model_dir}/{model_name_prior}', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

    # test on subject 1 with fake data
    if args.use_prior:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.diffusion_prior.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    if args.use_text:
        checkpoint = torch.load(f'{args.data_dir}/coco_tokens_avg_proj.pth')
        model.clipproj.load_state_dict(checkpoint)
        model.clipproj.requires_grad_(False)

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
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    train_dls = [train_dl]

    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
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
        loss_all = 0.

        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0.


        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = {} # empty dict because diff subjects have differing # of voxels
        image_iters = torch.zeros(num_iterations_per_epoch, args.batch_size, 3, 224, 224).float()
        text_iters = torch.zeros(num_iterations_per_epoch, args.batch_size, 1280).float()
        perm_iters, betas_iters, select_iters = {}, {}, {}

        for s, train_dl in enumerate(train_dls):
            with torch.cuda.amp.autocast(dtype=data_type):
                for iter_idx, (voxel, image, text) in enumerate(train_dl):
                    image = image[:, 2 + epoch % 2, :, :, :].float()
                    voxel = voxel[:, epoch % 2, :].half().unsqueeze(1)

                    image_iters[iter_idx, s * args.batch_size : s * args.batch_size + args.batch_size] = image
                    text_iters[iter_idx, s * args.batch_size : s * args.batch_size + args.batch_size] = text

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        voxel, perm, betas, select = utils.mixco(voxel)
                        perm_iters[f'subj0{subj_list[s]}_iter{iter_idx}'] = perm
                        betas_iters[f'subj0{subj_list[s]}_iter{iter_idx}'] = betas
                        select_iters[f'subj0{subj_list[s]}_iter{iter_idx}'] = select

                    voxel_iters[f'subj0{subj_list[s]}_iter{iter_idx}'] = voxel

                    if iter_idx >= num_iterations_per_epoch-1:
                        break

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        step = 0
        for train_i in range(num_iterations_per_epoch):
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                voxel_list = [voxel_iters[f'subj0{s}_iter{train_i}'].detach().to(device) for s in subj_list]
                voxel_list = [voxel_iters[f'subj0{s}_iter{train_i}'].detach() for s in subj_list]
                image = image_iters[train_i].detach()
                text = text_iters[train_i].detach()
                image = image.to(device)

                if args.use_image_aug:
                    image = img_augment(image)

                clip_target = clip_img_embedder(image)
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm_list = [perm_iters[f'subj0{s}_iter{train_i}'].detach().to(device) for s in subj_list]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f'subj0{s}_iter{train_i}'].detach().to(device) for s in subj_list]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f'subj0{s}_iter{train_i}'].detach().to(device) for s in subj_list]
                    select = torch.cat(select_list, dim=0)

                voxel_ridge= model.ridge(voxel_list[0],0)

                _, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)


                if args.clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)


                if args.use_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    loss_prior *= args.prior_scale
                    loss += loss_prior


                if args.clip_scale > 0:
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006,
                            perm=perm, betas=betas, select=select)
                    else:
                        epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=epoch_temp)

                    loss_clip_total += loss_clip.item()
                    loss_clip *= args.clip_scale
                    loss += loss_clip

                if args.use_text:
                    if args.use_prior:
                        text = text.to(device)
                        pred_text_norm = nn.functional.normalize(model.clipproj(prior_out).flatten(1), dim=-1)
                        target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                        loss += utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)* text_scale_prior
                    else:
                        text = text.to(device)
                        pred_text_norm = nn.functional.normalize(model.clipproj(clip_voxels).flatten(1), dim=-1)
                        target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                        loss += utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)* text_scale

                if args.blurry_recon:
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
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

                if args.clip_scale > 0:
                    # forward and backward top 1 accuracy
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                if args.blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
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
                print(f'Training epoch: {epoch}, sample: {step * args.batch_size}, lr: {optimizer.param_groups[0]['lr']}, loss_clip: {loss_clip.item():.4f}, loss: {loss.item():.4f}, loss_mean: {np.mean(losses[-(train_i+1):]):.4f}')

        model.eval()

        if local_rank==0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, (voxel, image, text) in enumerate(test_dl):
                    # all test samples should be loaded per batch such that test_i should never exceed 0

                    ## Average same-image repeats ##
                    if test_image is None:
                        voxel = voxel.half()
                        image = image[:,2,:,:,:].cpu()

                    loss = 0.

                    voxel = voxel.to(device)
                    image = image.to(device)

                    #clip_target = clip_img_embedder(image.float())

                    test_fwd_percent_correct = 0.
                    test_bwd_percent_correct = 0.
                    text_fwd_percent_correct = 0.


                    clip_target = clip_img_embedder(image.float())
                    voxel_ridge = model.ridge(voxel,0)
                    _, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

                    clip_voxels = clip_voxels.to(device)
                    clip_voxels /= 3
                    #backbone /= 3

                    if args.clip_scale > 0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)


                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//6, replace=False)

                    if args.use_prior:
                        loss_prior, prior_out = model.diffusion_prior(text_embed=clip_voxels[random_samps], image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss_prior *= args.prior_scale
                        loss += loss_prior

                    if args.use_text:
                        if not args.use_prior:
                            text = text.to(device)
                            pred_text_norm = nn.functional.normalize(model.clipproj(clip_voxels).flatten(1), dim=-1)
                            target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                            loss += utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)* text_scale
                            labels = torch.arange(len(pred_text_norm)).to(pred_text_norm.device)
                            text_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(pred_text_norm, target_text_norm), labels, k=5).item()
                        else:
                            text = text[random_samps].to(device)
                            pred_text_norm = nn.functional.normalize(model.clipproj(prior_out).flatten(1), dim=-1)
                            target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                            loss += utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)* text_scale
                            labels = torch.arange(len(pred_text_norm)).to(pred_text_norm.device)
                            text_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(pred_text_norm, target_text_norm), labels, k=5).item()



                    '''
                    if blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr += pixcorr.item()
                    '''

                    if args.clip_scale > 0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006)

                        test_loss_clip_total += loss_clip.item()
                        loss_clip = loss_clip * args.clip_scale
                        loss += loss_clip


                    if args.clip_scale > 0:
                        # forward and backward top 1 accuracy
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                        print('fwd:',test_fwd_percent_correct, 'bwd:',test_bwd_percent_correct, 'text fwd:', text_fwd_percent_correct)

                    utils.check_loss(loss)
                    loss_all += loss.item()
                    test_losses.append(loss.item())

                # if utils.is_interactive(): clear_output(wait=True)
                print('-------------------------')

        # Save model checkpoint and reconstruct
        if loss_all/4 < best_test_loss:
            best_test_loss = loss_all/4
            print('new best test loss:',best_test_loss)
            if not args.use_prior:
                save_ckpt(f'{args.neuroclips_model_dir}/{model_name}')
        else:
            print('not best:', loss_all/4, 'best test loss is', best_test_loss)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

    print('\n===Finished!===\n')
    if args.ckpt_saving and args.use_prior:
        save_ckpt(f'{args.neuroclips_model_dir}/{model_name}')
