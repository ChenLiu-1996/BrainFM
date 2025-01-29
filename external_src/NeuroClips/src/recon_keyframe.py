import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

from common import CC2017_Dataset, CLIPProj, Neuroclips, RidgeRegression, CLIPConverter
from Semantic import Semantic_Reconstruction, PriorNetwork, BrainDiffusionPrior

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
import_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(import_dir, 'generative_models/'))
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

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
        "--batch_size", type=int, default=4,
    )
    parser.add_argument(
        '--subj', type=int, default=1, choices=[1, 2, 3],
        help='Validate on which subject?',
    )
    parser.add_argument(
        '--blurry_recon', action=argparse.BooleanOptionalAction, default=False,
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
        '--plotting', action=argparse.BooleanOptionalAction, default=False,
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

    accelerator = Accelerator(split_batches=False, mixed_precision='fp16')
    print('PID of this process =', os.getpid())
    device = accelerator.device
    print('device:', device)

    NeuroClips_ROOT = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    args.data_dir = args.data_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)
    args.model_dir = args.model_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)
    args.neuroclips_model_dir = args.neuroclips_model_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)
    args.output_frame_dir = args.output_frame_dir.replace('$NeuroClips_ROOT', NeuroClips_ROOT)

    verify_necessary_files(args)

    # seed all random functions
    utils.seed_everything(args.seed)

    model_name = f'video_subj0{args.subj}_SR.pth'

    # make output directory
    os.makedirs(args.output_frame_dir, exist_ok=True)

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
        ckpt = torch.load(f'{args.model_dir}/sd_image_var_autoenc.pth')
        autoenc.load_state_dict(ckpt)

        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)
        utils.count_params(autoenc)

    model = Neuroclips()
    model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim, seq_len=seq_len)
    model.clipproj = CLIPProj()

    model.backbone = Semantic_Reconstruction(
        h=args.hidden_dim,
        in_dim=args.hidden_dim,
        seq_len=seq_len,
        n_blocks=args.n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim*clip_seq_dim,
        blurry_recon=False)

    utils.count_params(model.backbone)
    utils.count_params(model)

    # setup diffusion prior network
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

    model.to(device)

    checkpoint = torch.load(f'{args.neuroclips_model_dir}/{model_name}', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint

    # setup text caption networks
    from transformers import AutoProcessor
    from modeling_git import GitForCausalLMClipEmb
    processor = AutoProcessor.from_pretrained('microsoft/git-large-coco')
    clip_text_model = GitForCausalLMClipEmb.from_pretrained('microsoft/git-large-coco')
    clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    clip_text_model.eval().requires_grad_(False)

    clip_convert = CLIPConverter(
        clip_seq_dim=clip_seq_dim,
        clip_emb_dim=clip_emb_dim,
        clip_text_seq_dim=257,
        clip_text_emb_dim=1024)
    state_dict = torch.load(f'{args.model_dir}/mindeyev2_bigG_to_L_epoch8.pth', map_location='cpu')['model_state_dict']
    clip_convert.load_state_dict(state_dict, strict=True)
    clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    del state_dict

    # prep unCLIP
    config = OmegaConf.load(f'{import_dir}/generative_models/configs/unclip6.yaml')
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config['model']['params']
    network_config = unclip_params['network_config']
    denoiser_config = unclip_params['denoiser_config']
    first_stage_config = unclip_params['first_stage_config']
    conditioner_config = unclip_params['conditioner_config']
    sampler_config = unclip_params['sampler_config']
    scale_factor = unclip_params['scale_factor']
    disable_first_stage_autocast = unclip_params['disable_first_stage_autocast']
    offset_noise_level = unclip_params['loss_fn_config']['params']['offset_noise_level']

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast)

    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    checkpoint = torch.load(f'{args.model_dir}/mindeyev2_unclip6_epoch0_step110000.ckpt', map_location='cpu')
    diffusion_engine.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    batch = {'jpg': torch.randn(1, 3, 1, 1).to(device), # jpg doesn't get used, it's just a placeholder
             'original_size_as_tuple': torch.ones(1, 2).to(device) * 768,
             'crop_coords_top_left': torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out['vector'].to(device)
    print('vector_suffix', vector_suffix.shape)

    # get all reconstructions
    model.to(device)
    model.eval().requires_grad_(False)

    all_blurryrecons = None
    all_recons = None
    all_predcaptions = []
    all_clipvoxels = None
    all_textvoxels = None

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch_idx, (voxel, image) in enumerate(tqdm(test_dl)):
            voxel = voxel.to(device)
            if device != torch.device('cpu'):
                voxel = voxel.half()

            voxel_ridge = model.ridge(voxel, 0) # 0th index of subj_list
            _, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)

            # Save retrieval submodule outputs
            if all_clipvoxels is None:
                all_clipvoxels = clip_voxels.cpu()
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))

            # Feed voxels through OpenCLIP-bigG diffusion prior
            prior_out = model.diffusion_prior.p_sample_loop(
                clip_voxels.shape,
                text_cond=dict(text_embed=clip_voxels),
                cond_scale=1.0,
                timesteps=20)

            prior_out = prior_out.to(device)

            pred_caption_emb = clip_convert(prior_out)
            generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_predcaptions = np.hstack((all_predcaptions, generated_caption))
            print(generated_caption)

            # Feed diffusion prior outputs through unCLIP
            for instance_idx in range(len(voxel)):
                samples = utils.unclip_recon(
                    prior_out[[instance_idx]],
                    diffusion_engine,
                    vector_suffix,
                    num_samples=1,
                    device = device)
                if all_recons is None:
                    all_recons = samples.cpu()
                else:
                    all_recons = torch.vstack((all_recons, samples.cpu()))

                if args.plotting and batch_idx % (len(test_dl) // 5) == 0:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.imshow(transforms.ToPILImage()(samples[0]))
                    ax.axis('off')
                    fig.tight_layout(pad=2)
                    fig.savefig(f'{args.output_frame_dir}/video_subj0{args.subj}_image_clean_batch_{batch_idx}.png')

                if args.blurry_recon:
                    blurred_image = (autoenc.decode(blurry_image_enc[0]/0.18215).sample/ 2 + 0.5).clamp(0, 1)

                    for instance_idx in range(len(voxel)):
                        im = torch.Tensor(blurred_image[instance_idx])
                        if all_blurryrecons is None:
                            all_blurryrecons = im[None].cpu()
                        else:
                            all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))

                        if args.plotting and batch_idx % (len(test_dl) // 5) == 0:
                            fig = plt.figure(figsize=(8, 8))
                            ax = fig.add_subplot(1, 1, 1)
                            ax.imshow(transforms.ToPILImage()(im))
                            ax.axis('off')
                            fig.tight_layout(pad=2)
                            fig.savefig(f'{args.output_frame_dir}/video_subj0{args.subj}_image_blurry_batch_{batch_idx}.png')

    # resize outputs before saving
    imsize = 256
    all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    if args.blurry_recon:
        all_blurryrecons = transforms.Resize((imsize, imsize))(all_blurryrecons).float()

    # saving
    print(all_recons.shape)

    if args.blurry_recon:
        torch.save(all_blurryrecons, f'{args.output_frame_dir}/video_subj0{args.subj}_all_blurryrecons.pt')
    torch.save(all_recons, f'{args.output_frame_dir}/video_subj0{args.subj}_all_recons.pt')
    torch.save(all_predcaptions, f'{args.output_frame_dir}/video_subj0{args.subj}_all_predcaptions.pt')
    torch.save(all_clipvoxels, f'{args.output_frame_dir}/video_subj0{args.subj}_all_clipvoxels.pt')
    print(f'saved video_subj0{args.subj} outputs!')
