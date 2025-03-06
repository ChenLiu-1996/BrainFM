'''
Data for Neural Encoding and Decoding with Deep Learning for Dynamic Natural Vision Tests
https://purr.purdue.edu/publications/2809/1

We are downloading the one preprocessed by NeuroClips
https://github.com/gongzix/NeuroClips
'''

import os

DATA_DIR = '/gpfs/radev/project/krishnaswamy_smita/cl2482/common_data/Neural_Natural_Vision/data'
MODEL_DIR = '/gpfs/radev/project/krishnaswamy_smita/cl2482/common_data/Neural_Natural_Vision/checkpoints'

download_commands = [
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_3fps.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_caption.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_test_caption_emb.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_3fps.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_caption.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/GT_train_caption_emb.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/README.md',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/coco_tokens_avg_proj.pth',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj01_test_fmri.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj01_train_fmri.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj02_test_fmri.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj02_train_fmri.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj03_test_fmri.pt',
    f'wget -P {DATA_DIR} https://huggingface.co/datasets/gongzx/cc2017_dataset/resolve/main/subj03_train_fmri.pt',
    f'wget -O {MODEL_DIR}/mindeyev2_sd_image_var_autoenc.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/sd_image_var_autoenc.pth',
    f'wget -O {MODEL_DIR}/mindeyev2_convnext_xlarge_alpha0.75_fullckpt.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/convnext_xlarge_alpha0.75_fullckpt.pth',
    f'wget -O {MODEL_DIR}/mindeyev2_final_subj01_pretrained_40sess_24bs_last.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/train_logs/final_subj01_pretrained_40sess_24bs/last.pth',
    f'wget -O {MODEL_DIR}/mindeyev2_bigG_to_L_epoch8.pth https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/bigG_to_L_epoch8.pth',
    f'wget -O {MODEL_DIR}/mindeyev2_unclip6_epoch0_step110000.ckpt https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/unclip6_epoch0_step110000.ckpt',
    f'wget -O {MODEL_DIR}/animatediff_v3_sd15_adapter.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt',
    f'wget -O {MODEL_DIR}/animatediff_v3_sd15_mm.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt',
    f'wget -O {MODEL_DIR}/animatediff_v3_sd15_sparsectrl_rgb.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt',
    f'wget -O {MODEL_DIR}/dreambooth_lora_realisticVisionV60B1_v51VAE.safetensors https://huggingface.co/moiu2998/mymo/resolve/3c3093fa083909be34a10714c93874ce5c9dabc4/realisticVisionV60B1_v51VAE.safetensors'
]

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='stable-diffusion-v1-5/stable-diffusion-v1-5', local_dir=f'{MODEL_DIR}/stable-diffusion-v1-5',
                      allow_patterns=['text_encoder/*', 'tokenizer/*', 'unet/*', 'vae/*'])
    return


if __name__ == '__main__':
    for command in download_commands:
        os.system(command)

    download_model()
