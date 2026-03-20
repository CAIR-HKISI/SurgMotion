import os
import glob
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

datadir = '/mnt/tqy/'
dataset = 'SUN-SEG'

videos = os.listdir(os.path.join(datadir, dataset, 'SUN-Negative'))
# images = glob.glob(os.path.join(datadir, dataset, 'SUN-Negative/*/*.jpg'))
# images = sorted(images, key=lambda x: int(x.split('/')[-1].split('_')[-1][5:-4]))
# print(len(images))
# print(images[0])
# exit(0)


def index_frame(video, image, idx):
    save_dir = os.path.join(datadir, dataset, 'indexed', video)

    # print(image, video, idx, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, 'img_{:05}.jpg'.format(idx))):
        try:
            cv2.imwrite(os.path.join(save_dir, 'img_{:05}.jpg'.format(idx)),
                        cv2.imread(os.path.join(datadir, dataset, 'SUN-Negative', video, image)))
        except:
            print(os.path.join(datadir, dataset, 'SUN-Negative', video, image))


# for video in videos:
#     images = os.listdir(os.path.join(datadir, dataset, 'SUN-Negative', video))
#     for idx, image in tqdm(enumerate(images)):
#         index_frame(video, image, idx + 1)
#         exit(0)

for video in videos:
    print(f'Processing {video}...')
    path = os.path.join(datadir, dataset, 'SUN-Negative', video)
    images = os.listdir(path)
    # for source in images:
    #     index = int(source.split('_')[-1].split('.')[0][6:])
    #     destination = path + "/img_%05d.jpg"%index
    #     source = path + '/' + source
    #     os.rename(source, destination)
    n_jobs = 1
    Parallel(n_jobs=n_jobs)(delayed(index_frame)(video, image, idx + 1) for idx, image in tqdm(enumerate(images)))
