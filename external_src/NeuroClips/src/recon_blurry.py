import os
import argparse

import torch
import torch.nn as nn
from accelerate import Accelerator

from common import CC2017_Dataset, Neuroclips, RidgeRegression
from Perception import Perception_Reconstruction, Inception_Extension

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
    assert os.path.isfile(f'{args.data_dir}/subj0{args.subj}_test_fmri.pt')
    assert os.path.isfile(f'{args.data_dir}/GT_test_3fps.pt')
    assert os.path.isfile(f'{args.model_dir}/sd_image_var_autoenc.pth')
    assert os.path.isfile(f'{args.model_dir}/mindeyev2_bigG_to_L_epoch8.pth')
    assert os.path.isfile(f'{args.model_dir}/mindeyev2_unclip6_epoch0_step110000.ckpt')
    if verbose is True:
        print('Verified: all necessary files are available!\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training Configuration')
    parser.add_argument(
        '--subj', type=int, default=1, choices=[1, 2, 3],
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
        '--mixup_pct', type=float, default=0.33,
        help='proportion of way through training when to switch from BiMixCo to SoftCLIP',
    )
    parser.add_argument(
        '--blurry_recon', action=argparse.BooleanOptionalAction, default=True,
        help='whether to output blurry reconstructions',
    )
    parser.add_argument(
        '--blur_scale', type=float, default=0.5,
        help='multiply loss from blurry recons by this number',
    )
    parser.add_argument(
        '--clip_scale', type=float, default=1.0,
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
        '--seed', type=int, default=42,
    )
    parser.add_argument(
        '--use_text', action=argparse.BooleanOptionalAction, default=False,
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
    parser.add_argument(
        '--output_frame_dir', type=str, default='$NeuroClips_ROOT/outputs/frames/',
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
    args.output_frame_dir = args.output_frame_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)

    verify_necessary_files(args)

    # seed all random functions
    utils.seed_everything(args.seed)

    model_name_PR = f'video_subj0{args.subj}_PR.pth'

    if args.subj == 1:
        voxel_length = 13447
    elif args.subj == 2 :
        voxel_length = 14828
    elif args.subj == 3 :
        voxel_length = 9114

    clip_seq_dim = 256
    clip_emb_dim = 1664
    seq_len = 1

    voxel_test = torch.load(f'{args.data_dir}/subj0{args.subj}_test_fmri.pt', map_location='cpu')
    voxel_test = torch.mean(voxel_test, dim = 1).unsqueeze(1)
    num_voxels_list = [voxel_test.shape[-1]]
    print('Loaded all test fMRI frames to cpu!', voxel_test.shape)

    test_images = torch.load(f'{args.data_dir}/GT_test_3fps.pt', map_location='cpu')
    print('Loaded all test image frames to cpu!', test_images.shape)

    test_dataset = CC2017_Dataset(voxel_test, test_images, istrain = False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    if args.blurry_recon:
        from diffusers import AutoencoderKL
        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        checkpoint = torch.load(f'{args.model_dir}/sd_image_var_autoenc.pth')
        autoenc.load_state_dict(checkpoint)

        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)
        utils.count_params(autoenc)

    model = Neuroclips()

    model.ridge1 = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
    utils.count_params(model.ridge1)
    utils.count_params(model)

    model.backbone = Perception_Reconstruction(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=seq_len,
        n_blocks=args.n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim*clip_seq_dim,
        blurry_recon=args.blurry_recon,
        clip_scale=args.clip_scale)

    model.fmri = Inception_Extension(
        h=256,
        in_dim=voxel_length,
        out_dim=voxel_length,
        expand=args.fps*2,
        seq_len=seq_len)

    utils.count_params(model.backbone)
    utils.count_params(model.fmri)
    utils.count_params(model)

    checkpoint = torch.load(f'{args.neuroclips_model_dir}/{model_name_PR}', map_location='cpu')['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint

    # epoch = 0
    # print(f'{model_name} starting with epoch {epoch} / {num_epochs}')
    # progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
    model.eval()
    all_blurryrecons = None

    if local_rank == 0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
            for voxel, image in test_dl:
                voxel = voxel.half().to(device)

                image = image.reshape(len(image) * args.fps * 2, 3, 224, 224).cpu()
                image = image.to(device)

                voxel = model.fmri(voxel).unsqueeze(1)
                voxel_ridge = model.ridge1(voxel, 0)
                blurry_image_enc_ = model.backbone(voxel_ridge, time=40*args.fps*2)

                if args.blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0, 1)

                for instance_idx in range(len(voxel)):
                    im = torch.Tensor(blurry_recon_images[instance_idx])
                    if all_blurryrecons is None:
                        all_blurryrecons = im[None].cpu()
                    else:
                        all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))

                print(all_blurryrecons.shape)

    torch.save(all_blurryrecons, f'{args.output_frame_dir}/video_subj0{args.subj}_all_true_blurryrecons.pt')