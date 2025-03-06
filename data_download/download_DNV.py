'''
Data for Neural Encoding and Decoding with Deep Learning for Dynamic Natural Vision Tests
https://purr.purdue.edu/publications/2809/1

Here we are downloading the raw data and preprocessing ourselves.
'''

import os

DATA_DIR = '/gpfs/radev/project/krishnaswamy_smita/cl2482/common_data/Dynamic_Natural_Vision'

download_commands = [
    f'wget -O {DATA_DIR}/stimuli.zip https://purr.purdue.edu/publications/2808/serve/1?render=archive',
    f'wget -O {DATA_DIR}/subject01.zip ftp://purr.purdue.edu/10_4231_R7X63K3M.zip',
    f'wget -O {DATA_DIR}/subject02.zip ftp://purr.purdue.edu/10_4231_R7NS0S1F.zip',
    f'wget -O {DATA_DIR}/subject03.zip ftp://purr.purdue.edu/10_4231_R7J101BV.zip',
    f'wget -O {DATA_DIR}/author_source_code.zip https://purr.purdue.edu/publications/2816/serve/2?render=archive',
]


if __name__ == '__main__':
    for command in download_commands:
        os.system(command)
