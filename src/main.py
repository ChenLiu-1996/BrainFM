import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.stgTransformer import SpatialTemporalGraphTransformer
from model.vision_encoders import VisionEncoder
from torch_geometric.loader import DataLoader

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from log import log
from data_split import split_dataset
from scheduler import LinearWarmupCosineAnnealingLR

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
        mode='train')

    dataset_test = DynamicNaturalVisionDataset(
        subject_idx=None,
        fMRI_window_frames=args.fMRI_window_frames,
        base_folder=args.data_folder,
        brain_atlas_folder=args.brain_atlas_folder,
        graph_knn_k=args.graph_knn_k,
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


def train_epoch(model, train_loader, optimizer, loss_fn, device, max_iter):
    train_loss, train_cossim_clip, train_cossim_resnet, train_cossim_convnext = 0, 0, 0, 0

    optimizer.zero_grad()
    batch_per_backprop = int(args.desired_batch_size / args.batch_size)

    # These encoders are for perceptual similarity.
    encoder_clip = VisionEncoder(pretrained_model='clip', device=device)
    encoder_resnet = VisionEncoder(pretrained_model='resnet18', device=device)
    encoder_convnext = VisionEncoder(pretrained_model='convnext_tiny', device=device)

    total_batches = min(max_iter, len(train_loader))
    plot_freq = max(total_batches // args.n_plot_per_epoch, 1)
    for batch_idx, data_item in enumerate(tqdm(train_loader, total=total_batches)):
        if max_iter is not None and batch_idx > max_iter:
            break

        should_plot = batch_idx % plot_freq == 0

        fMRI_graph = data_item[0].to(device)
        video_true = data_item[1].to(device)
        video_pred = model(fMRI_graph)

        train_cossim_clip += cossim_video(encoder_clip, video_true, video_pred)
        train_cossim_resnet += cossim_video(encoder_resnet, video_true, video_pred)
        train_cossim_convnext += cossim_video(encoder_convnext, video_true, video_pred)

        loss = loss_fn(video_pred.float(), video_true.float())

        loss_ = loss / batch_per_backprop
        loss_.backward()
        train_loss += loss.mean().item()

        # Simulate bigger batch size by batched optimizer update.
        if batch_idx % batch_per_backprop == batch_per_backprop - 1:
            optimizer.step()
            optimizer.zero_grad()

        if should_plot:
            video_pred = torch.randn(size=video_true.size())
            plot_video_frames(video_true, video_pred, mode='train', batch_idx=(batch_idx + 1))

    train_loss /= min(max_iter, len(train_loader))
    train_cossim_clip /= min(max_iter, len(train_loader))
    train_cossim_resnet /= min(max_iter, len(train_loader))
    train_cossim_convnext /= min(max_iter, len(train_loader))
    return model, train_loss, train_cossim_clip, train_cossim_resnet, train_cossim_convnext

@torch.no_grad()
def val_epoch(model, val_loader, loss_fn, device, max_iter):
    val_loss, val_cossim_clip, val_cossim_resnet, val_cossim_convnext = 0, 0, 0, 0

    # These encoders are for perceptual similarity.
    encoder_clip = VisionEncoder(pretrained_model='clip', device=device)
    encoder_resnet = VisionEncoder(pretrained_model='resnet18', device=device)
    encoder_convnext = VisionEncoder(pretrained_model='convnext_tiny', device=device)

    total_batches = min(max_iter, len(val_loader))
    plot_freq = max(total_batches // args.n_plot_per_epoch, 1)

    for batch_idx, data_item in enumerate(val_loader):
        if max_iter is not None and batch_idx > max_iter:
            break

        should_plot = batch_idx % plot_freq == 0

        fMRI_graph = data_item[0].to(device)
        video_true = data_item[1].to(device)
        video_pred = model(fMRI_graph)

        val_cossim_clip += cossim_video(encoder_clip, video_true, video_pred)
        val_cossim_resnet += cossim_video(encoder_resnet, video_true, video_pred)
        val_cossim_convnext += cossim_video(encoder_convnext, video_true, video_pred)

        loss = loss_fn(video_pred.float(), video_true.float())
        val_loss += loss.mean().item()

        if should_plot:
            plot_video_frames(video_true, video_pred, mode='val', batch_idx=(batch_idx + 1))

    val_loss /= min(max_iter, len(val_loader))
    val_cossim_clip /= min(max_iter, len(val_loader))
    val_cossim_resnet /= min(max_iter, len(val_loader))
    val_cossim_convnext /= min(max_iter, len(val_loader))
    return model, val_loss, val_cossim_clip, val_cossim_resnet, val_cossim_convnext

@torch.no_grad()
def test_model(model, test_loader, loss_fn, device):
    test_loss, test_cossim_clip, test_cossim_resnet, test_cossim_convnext = 0, 0, 0, 0

    # These encoders are for perceptual similarity.
    encoder_clip = VisionEncoder(pretrained_model='clip', device=device)
    encoder_resnet = VisionEncoder(pretrained_model='resnet18', device=device)
    encoder_convnext = VisionEncoder(pretrained_model='convnext_tiny', device=device)

    for data_item in test_loader:
        fMRI_graph = data_item[0].to(device)
        video_true = data_item[1].to(device)
        video_pred = model(fMRI_graph)

        test_cossim_clip += cossim_video(encoder_clip, video_true, video_pred)
        test_cossim_resnet += cossim_video(encoder_resnet, video_true, video_pred)
        test_cossim_convnext += cossim_video(encoder_convnext, video_true, video_pred)

        loss = loss_fn(video_pred.float(), video_true.float())
        test_loss += loss.mean().item()

    test_loss /= len(test_loader)
    test_cossim_clip /= len(test_loader)
    test_cossim_resnet /= len(test_loader)
    test_cossim_convnext /= len(test_loader)
    return model, test_loss, test_cossim_clip, test_cossim_resnet, test_cossim_convnext

def plot_video_frames(video_true: torch.Tensor, video_pred: torch.Tensor, mode: str, batch_idx: int):
    '''
    Will only plot the first batch and the first 6 frames.
    '''
    video_true = video_true.detach().cpu().numpy()
    video_pred = video_pred.detach().cpu().numpy()

    fig = plt.figure(figsize=(24, 8))
    for frame_idx in range(6):
        ax = fig.add_subplot(2, 6, frame_idx + 1)
        ax.imshow(video_true[0, :, :, :, frame_idx])
        if frame_idx == 0:
            ax.set_title('Ground truth video')

        ax = fig.video_pred(2, 6, frame_idx + 7)
        ax.imshow(video_true[0, :, :, :, frame_idx])
        if frame_idx == 0:
            ax.set_title('Predicted video')

    save_path = os.path.join(args.plot_folder, mode, f'batch_{batch_idx}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(save_path)
    plt.close(fig)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-ratio', default='4:1', type=str)
    args.add_argument('--max-epochs', default=50, type=int)
    args.add_argument('--max-training-iters', default=512, type=int)
    args.add_argument('--max-validation-iters', default=256, type=int)
    args.add_argument('--n-plot-per-epoch', default=4, type=int)
    args.add_argument('--batch-size', default=2, type=int)
    args.add_argument('--desired-batch-size', default=16, type=int)
    args.add_argument('--fMRI-window-frames', default=3, type=int)
    args.add_argument('--graph-knn-k', default=5, type=int)
    args.add_argument('--learning-rate', default=1e-3, type=float)
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

    model = SpatialTemporalGraphTransformer(
        in_channels=1,
        hidden_dim=16,
        nhead=1,
        num_transformer_layers=1,
        input_frames=args.fMRI_window_frames,
        temporal_upsampling=6,
    )
    model.eval()
    model.to(device)

    # Set up training tools.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=min(10, args.max_epochs),
        warmup_start_lr=args.learning_rate * 1e-2,
        max_epochs=args.max_epochs)
    loss_fn = torch.nn.MSELoss()

    curr_run_identifier = f'model-STGT_window-{args.fMRI_window_frames}_lr-{args.learning_rate}_seed-{args.random_seed}'
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

    log('[SpatialTemporalGraphTransformer] Training begins.', filepath=args.log_file)
    log(f'Using device: {device}', filepath=args.log_file)
    best_val_loss = np.inf
    for epoch_idx in tqdm(range(args.max_epochs)):
        model.train()
        model, loss, cossim_clip, cossim_resnet, cossim_convnext = train_epoch(model, train_loader, optimizer, loss_fn, device, args.max_training_iters)
        scheduler.step()
        log(f'Epoch {epoch_idx}/{args.max_epochs}: Training Loss {loss:.3f}, CosSim CLIP|ResNet|ConvNext={cossim_clip:.3f}|{cossim_resnet:.3f}|{cossim_convnext:.3f}.',
            filepath=args.log_file)

        model.eval()
        model, loss, cossim_clip, cossim_resnet, cossim_convnext = val_epoch(model, val_loader, loss_fn, device, args.max_validation_iters)
        log(f'Validation Loss {loss:.3f}, CosSim CLIP|ResNet|ConvNext={cossim_clip:.3f}|{cossim_resnet:.3f}|{cossim_convnext:.3f}.',
            filepath=args.log_file)

        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), args.model_save_path)
            log('Model weights successfully saved.', filepath=args.log_file)

    model.eval()
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model, loss, cossim_clip, cossim_resnet, cossim_convnext = test_model(model, test_loader, loss_fn, device)
    log(f'\n\nTest Loss {loss:.3f}, CosSim CLIP|ResNet|ConvNext={cossim_clip:.3f}|{cossim_resnet:.3f}|{cossim_convnext:.3f}.',
        filepath=args.log_file)
