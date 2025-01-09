import os
import re
from glob import glob
from multiprocessing import Pool

data_folder = '/gpfs/radev/scratch/krishnaswamy_smita/cl2482/BrainFM_2024_HBN/zip/'

def download_url(url: str) -> None:
    os.system(f'wget -P {data_save_dir} {url}')


if __name__ == '__main__':
    for subfolder in ['MRI', 'EEG']:

        # Read the webscraping results and find all download links.
        webscraping_file_list = glob(f'./webscraping/web_HBN_{subfolder}_*.txt')
        url_write_file = f'./url_HBN_{subfolder}.txt'

        if not os.path.isfile(url_write_file):
            url_list = []

            for webscraping_file in webscraping_file_list:
                with open(webscraping_file, 'r') as f:
                    content = f.read()
                pattern = r'value="http.*?"'
                matches = re.findall(pattern, content)

                url_list.extend([item.replace('value=', '').replace('"', '') for item in matches])

            with open(url_write_file, 'w') as f:
                for url in url_list:
                    f.write(f'{url}\n')

        # Download the files via wget.
        url_list = []
        data_save_dir = f'{data_folder}/{subfolder}/'
        os.makedirs(data_save_dir, exist_ok=True)

        with open(url_write_file, 'r') as f:
            content = f.read()
            url_list = content.split('\n')
            url_list = [item for item in url_list if item]

        Pool(8).map(download_url, url_list)