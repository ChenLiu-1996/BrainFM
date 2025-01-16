'''
Data for Neural Encoding and Decoding with Deep Learning for Dynamic Natural Vision Tests
https://purr.purdue.edu/publications/2809/1

We are downloading the one preprocessed by NeuroClips
https://github.com/gongzix/NeuroClips
'''

import os

data_dir = '/gpfs/radev/project/krishnaswamy_smita/cl2482/common_data/Neural_Natural_Vision/data'
model_dir = '/gpfs/radev/project/krishnaswamy_smita/cl2482/common_data/Neural_Natural_Vision/checkpoints'

download_commands = [
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_3fps.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_caption.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_caption_emb.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_3fps.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_caption.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_caption_emb.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/README.md',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/coco_tokens_avg_proj.pth',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj01_test_fmri.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj01_train_fmri.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj02_test_fmri.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj02_train_fmri.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj03_test_fmri.pt',
    f'wget -P {data_dir} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj03_train_fmri.pt',
    f'wget -P {model_dir} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/sd_image_var_autoenc.pth',
    f'wget -P {model_dir} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/convnext_xlarge_alpha0.75_fullckpt.pth',
    f'wget -O {model_dir}/final_subj01_pretrained_40sess_24bs_last.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj01_pretrained_40sess_24bs/last.pth',
]

if __name__ == '__main__':
    for command in download_commands:
        os.system(command)