import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


def dynamic_padding_collate_fn(batch):
    """
    Custom collate function to dynamically pad all samples in a batch to the maximum sequence length.
    Args:
        batch (list of dict): List of samples, each with 'image', 'label', and optionally other metadata.

    Returns:
        dict: Batch with padded 'image' and 'label', and original 'case_name'.
    """
    # Find the maximum number of frames in the batch
    max_frames = max(sample['image'].shape[0] for sample in batch)

    # Prepare padded tensors
    images = []
    labels = []
    case_names = []

    for sample in batch:
        image = sample['image']  # shape: (T, H, W) or (T, C, H, W)
        label = sample['label']  # shape: (T, H, W) or (T, C, H, W)

        # Determine the padding sizes
        image_padding = ((0, max_frames - image.shape[0]),) + ((0, 0),) * (image.ndim - 1)
        label_padding = ((0, max_frames - label.shape[0]),) + ((0, 0),) * (label.ndim - 1)

        # Pad the tensors
        pad_image = np.pad(image, image_padding, mode='constant', constant_values=0)
        pad_label = np.pad(label, label_padding, mode='constant', constant_values=0)

        images.append(torch.tensor(pad_image))
        labels.append(torch.tensor(pad_label))
        case_names.append(sample.get('case_name', None))  # Optional case_name

    # Stack all padded samples into tensors
    batch_images = torch.stack(images)
    batch_labels = torch.stack(labels)

    return {
        'image': batch_images,
        'label': batch_labels,
        'case_name': case_names
    }