import os
from torchvision import transforms
from torch.utils.data import ConcatDataset
from .transforms import *
from .masking_generator import (
    TubeMaskingGenerator, RandomMaskingGenerator,
    TubeRowMaskingGenerator,
    RandomRowMaskingGenerator
)
from .mae import VideoMAE
from .kinetics import VideoClsDataset
from .kinetics_sparse import VideoClsDataset_sparse
from .ssv2 import SSVideoClsDataset, SSRawFrameClsDataset
from .lvu import LVU


DATASETS_CONFIG = {
    "Colonoscopic":
        {
        'root': '/mnt/tqy/Colonoscopic/',
        'setting': '/mnt/tqy/Colonoscopic/train_list.txt',
        'prefix': '/mnt/tqy/Colonoscopic/',
        },
    "LDPolypVideo": 
        {
        'root': '/mnt/tqy/LDPolypVideo/',
        'setting': '/mnt/tqy/LDPolypVideo/train_list.txt',
        'prefix': '/mnt/tqy/LDPolypVideo/',
        },
    "Hyper-Kvasir":
        {
        'root': '/mnt/tqy/Hyper-Kvasir/',
        'setting': '/mnt/tqy/Hyper-Kvasir/train_list.txt',
        'prefix': '/mnt/tqy/Hyper-Kvasir/',
        },
    "Kvasir-Capsule":
        {
        'root': '/mnt/tqy/Kvasir-Capsule/',
        'setting': '/mnt/tqy/Kvasir-Capsule/train_list.txt',
        'prefix': '/mnt/tqy/Kvasir-Capsule/',
        },
    "CholecT45": 
        {
        'root': '/mnt/tqy/CholecT45/',
        'setting': '/mnt/tqy/CholecT45/train_list.txt',
        'prefix': '/mnt/tqy/CholecT45/',
        },
    "EndoFM":
        {
        'root': '/mnt/tqy/EndoFM/',
        'setting': '/mnt/tqy/EndoFM/train_list.txt',
        'prefix': '/mnt/tqy/EndoFM/',
        },
    "SUN-SEG":
        {
        'root': '/mnt/tqy/SUN-SEG/',
        'setting': '/mnt/tqy/SUN-SEG/train_list.txt',
        'prefix': '/mnt/tqy/SUN-SEG/',
        },
    "GLENDAv1":
        {
        'root': '/mnt/tqy/GLENDA_v1.0/',
        'setting': '/mnt/tqy/GLENDA_v1.0/train_list.txt',
        'prefix': '/mnt/tqy/GLENDA_v1.0/',
        },
    "gastric_real":
        {
        'root': '/mnt/tqy/gastroc_real/',
        'setting': '/mnt/tqy/gastroc_real/train_list.txt',
        'prefix': '/mnt/tqy/gastroc_real/',
        },
    "EndoMapper":
        {
        'root': '/mnt/tqy/EndoMapper/',
        'setting': '/mnt/tqy/EndoMapper/train_list.txt',
        'prefix': '/mnt/tqy/EndoMapper/',
        },
    "ROBUST-MIS":
        {
        'root': '/mnt/tqy/ROBUST-MIS/',
        'setting': '/mnt/tqy/ROBUST-MIS/train_list.txt',
        'prefix': '/mnt/tqy/ROBUST-MIS/',
        },
    "Ours-Porcine":
        {
        'root': '/mnt/tqy/ours_porcine_clips/',
        'setting': '/mnt/tqy/ours_porcine_clips/train_list.txt',
        'prefix': '/mnt/tqy/ours_porcine_clips/',
        },
}

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        # self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
        # self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        if args.color_jitter > 0:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupColorJitter(args.color_jitter),
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'tube_row':
            self.masked_position_generator = TubeRowMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random_row':
            self.masked_position_generator = RandomRowMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type in 'attention':
            self.masked_position_generator = None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        if self.masked_position_generator is None:
            return process_data, -1
        else:
            return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_pretraining_mixed_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    datasets = []
    for dataset_name in args.mix_datasets:
        dataset_config = DATASETS_CONFIG[dataset_name]
        dataset = VideoMAE(
            root=dataset_config['root'],
            setting=dataset_config['setting'],
            prefix=dataset_config['prefix'],
            split=args.split,
            video_ext='mp4',
            is_color=True,
            modality='rgb',
            num_segments=args.num_segments,
            new_length=args.num_frames,
            new_step=args.sampling_rate,
            transform=transform,
            temporal_jitter=False,
            video_loader=True,
            use_decord=args.use_decord,
            lazy_init=False,
            num_sample=args.num_sample)
        datasets.append(dataset)
    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    if args.data_set in [
            'Kinetics',
            'Kinetics_sparse',
            'mitv1_sparse'
        ]:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if 'sparse' in args.data_set:
            func = VideoClsDataset_sparse
        else:
            func = VideoClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        nb_classes = args.nb_classes
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if args.use_decord:
            func = SSVideoClsDataset
        else:
            func = SSRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set in [
            'LVU',
            'COIN',
            'Breakfast'
        ]:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        func = LVU

        dataset = LVU(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            trimmed=args.trimmed,
            time_stride=args.time_stride,
        )
        
        if args.data_set == "Breakfast":
            nb_classes = 10
        elif args.data_set == "COIN":
            nb_classes = 180
        elif args.data_set == "LVU":
            if "relation" in args.data_path.lower():
                nb_classes = 4
            elif "speak" in args.data_path.lower():
                nb_classes = 5
            elif "scene" in args.data_path.lower():
                nb_classes = 6
            elif "director" in args.data_path.lower():
                nb_classes = 10
            elif "genre" in args.data_path.lower():
                nb_classes = 4
            elif "writer" in args.data_path.lower():
                nb_classes = 10
            elif "year" in args.data_path.lower():
                nb_classes = 9
            else:
                nb_classes = -1

    else:
        print(f'Wrong: {args.data_set}')
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
