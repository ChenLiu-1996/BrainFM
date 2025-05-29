import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from diffusers import StableDiffusionXLPipeline
from einops import rearrange

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from log import log
from data_split import split_dataset

sys.path.insert(0, import_dir + '/src/dataset/')
from dynamic_natural_vision import DynamicNaturalVisionDataset


ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])


def prepare_dataloaders(args):

    dataset_train_val = DynamicNaturalVisionDataset(
        subject_idx=None,
        fMRI_window_frames=args.fMRI_window_frames,
        base_folder=args.data_folder,
        brain_atlas_folder=args.brain_atlas_folder,
        graph_knn_k=args.graph_knn_k,
        target_image_dim=(224, 224),
        mode='train')

    dataset_test = DynamicNaturalVisionDataset(
        subject_idx=None,
        fMRI_window_frames=args.fMRI_window_frames,
        base_folder=args.data_folder,
        brain_atlas_folder=args.brain_atlas_folder,
        graph_knn_k=args.graph_knn_k,
        target_image_dim=(224, 224),
        mode='test')

    # Train/val/test split
    ratios = [float(c) for c in args.train_val_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])

    train_set, val_set = split_dataset(
        dataset=dataset_train_val,
        splits=ratios,
        random_seed=0)  # Fix the dataset.
    test_set = dataset_test

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def cossim_video(encoder, video_true, video_pred):
    cossim_list = []
    assert video_true.shape == video_pred.shape
    assert len(video_true.shape) == 5  # [B, H, W, 3, T]
    for t in range(video_true.shape[-1]):
        cossim_item = torch.nn.functional.cosine_similarity(
            encoder.embed(video_true[..., t].permute((0, 3, 1, 2))),
            encoder.embed(video_pred[..., t].permute((0, 3, 1, 2)))).mean().item()
        cossim_list.append(cossim_item)
    return np.mean(cossim_list)


def train_epoch(model, train_loader, optimizer, loss_fn, device, max_iter, epoch_idx):
    train_loss, train_cossim_clip, train_cossim_resnet, train_cossim_convnext = 0, 0, 0, 0

    total_batches = min(max_iter, len(train_loader))
    plot_freq = max(total_batches // args.n_plot_per_epoch, 1)
    for batch_idx, data_item in enumerate(tqdm(train_loader, total=total_batches)):
        if max_iter is not None and batch_idx > max_iter:
            break

        should_plot = batch_idx % plot_freq == 0

        # fMRI_graph = data_item[0].to(device)
        video_true = data_item[1].to(device)
        sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
        ).to(device)
        sdxl.enable_model_cpu_offload()

        num_frames = 6
        num_inference_steps = 25
        batch_size = 1
        num_images_per_prompt = 1

        sdxl.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sdxl.scheduler.timesteps

        recon_frames = None
        for frame_idx in range(num_frames):
            image = rearrange(video_true[..., frame_idx], 'b h w c -> b c h w').float()  # [1, 3, 224, 224]
            image = (image + 1) / 2  # Image are expected to be within [0, 1].

            latents = sdxl.vae.encode(image).latent_dist.sample() * sdxl.vae.config.scaling_factor

            recon_image = sdxl.vae.decode(latents / sdxl.vae.config.scaling_factor).sample
            recon_image = recon_image * 2 - 1  # Rescale to roughly [-1, 1].

            if recon_frames is None:
                recon_frames = recon_image.unsqueeze(-1)
            else:
                recon_frames = torch.cat((recon_frames, recon_image.unsqueeze(-1)), dim=-1)

        video_pred = rearrange(recon_frames, 'b c h w t -> b h w c t')

        plot_video_frames(video_true, video_pred, mode='train', epoch_idx=epoch_idx, batch_idx=batch_idx)

        if should_plot:
            plot_video_frames(video_true, video_pred, mode='train', epoch_idx=epoch_idx, batch_idx=batch_idx)

    return model, train_loss, train_cossim_clip, train_cossim_resnet, train_cossim_convnext


def plot_video_frames(video_true: torch.Tensor, video_pred: torch.Tensor, mode: str, epoch_idx: int, batch_idx: int):
    '''
    Will only plot the first batch and the first 6 frames.
    '''
    video_true = video_true.detach().cpu().numpy()
    video_pred = video_pred.detach().cpu().numpy()

    fig = plt.figure(figsize=(24, 8))
    for frame_idx in range(6):
        ax = fig.add_subplot(2, 6, frame_idx + 1)
        image_true = video_true[0, :, :, :, frame_idx]
        image_true = (image_true + 1) / 2
        image_true = np.clip(image_true, 0, 1)
        ax.imshow(image_true)
        ax.set_axis_off()
        if frame_idx == 0:
            ax.set_title('Ground truth video', fontsize=16)

        ax = fig.add_subplot(2, 6, frame_idx + 7)
        image_pred = video_pred[0, :, :, :, frame_idx]
        image_pred = (image_pred + 1) / 2
        image_pred = np.clip(image_pred, 0, 1)
        ax.imshow(image_pred)
        ax.set_axis_off()
        if frame_idx == 0:
            ax.set_title('Reconstructed video', fontsize=16)

    save_path = os.path.join(args.plot_folder, mode, f'epoch_{epoch_idx + 1}_batch_{batch_idx + 1}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(save_path)
    plt.close(fig)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-ratio', default='4:1', type=str)
    args.add_argument('--max-epochs', default=1, type=int)
    args.add_argument('--max-training-iters', default=5, type=int)
    args.add_argument('--max-validation-iters', default=1, type=int)
    args.add_argument('--n-plot-per-epoch', default=5, type=int)
    args.add_argument('--batch-size', default=1, type=int)
    args.add_argument('--desired-batch-size', default=16, type=int)
    args.add_argument('--fMRI-window-frames', default=1, type=int)
    args.add_argument('--graph-knn-k', default=5, type=int)
    args.add_argument('--learning-rate', default=1e-2, type=float)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--data-folder', default='$ROOT/data/Dynamic_Natural_Vision/', type=str)
    args.add_argument('--brain-atlas-folder', default='$ROOT/data/brain_parcellation/AAL3/', type=str)

    args = args.parse_known_args()[0]
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)
    args.brain_atlas_folder = args.brain_atlas_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    # Set up training tools.
    model = None
    optimizer = None
    scheduler = None
    loss_fn = torch.nn.MSELoss()

    curr_run_identifier = f'SDXL_VQVAE_window-{args.fMRI_window_frames}_lr-{args.learning_rate}_seed-{args.random_seed}'
    args.model_save_path = os.path.join(ROOT_DIR, 'results', curr_run_identifier, 'model.pt')
    args.log_file = os.path.join(ROOT_DIR, 'results', curr_run_identifier, 'log.txt')
    args.plot_folder = os.path.join(ROOT_DIR, 'results', curr_run_identifier, 'figures')
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # Log the config.
    config_str = 'Config: \n'
    for key in vars(args).keys():
        config_str += '%s: %s\n' % (key, vars(args)[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=args.log_file, to_console=True)

    log('Testing the VQVAE in Stable Diffusion XL.', filepath=args.log_file)
    log(f'Using device: {device}', filepath=args.log_file)
    best_val_loss = np.inf
    for epoch_idx in tqdm(range(args.max_epochs)):
        model, loss, cossim_clip, cossim_resnet, cossim_convnext = train_epoch(model, train_loader, optimizer, loss_fn, device, args.max_training_iters, epoch_idx)
        log(f'Epoch {epoch_idx}/{args.max_epochs}: Training Loss {loss:.3f}, CosSim CLIP|ResNet|ConvNext={cossim_clip:.3f}|{cossim_resnet:.3f}|{cossim_convnext:.3f}.',
            filepath=args.log_file)
