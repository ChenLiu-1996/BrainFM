import torch
import torch.nn as nn


class CC2017_Dataset(torch.utils.data.Dataset):
    def __init__(self, voxel, image, text = None, istrain: bool = False):
        if istrain == True:
            self.length = 4320
        else:
            self.length = 1200
        self.voxel = voxel
        self.image = image
        self.text = text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.text is not None:
            return self.voxel[idx], self.image[idx], self.text[idx]
        else:
            return self.voxel[idx], self.image[idx]


class CLIPProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1664, 1280))
    def forward(self, x):
        x = torch.mean(x, dim = 1)
        x = x @ self.proj
        return x


class CLIPConverter(torch.nn.Module):
    def __init__(self,
                 clip_seq_dim: int,
                 clip_emb_dim: int,
                 clip_text_seq_dim: int,
                 clip_text_emb_dim: int):

        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


class Neuroclips(nn.Module):
    def __init__(self):
        super(Neuroclips, self).__init__()
    def forward(self, x):
        return x


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
        self.seq_len = seq_len

    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(self.seq_len)], dim=1)
        return out
