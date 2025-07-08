# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append('/data/wjl/vjepa2')
sys.path.append('/data/wjl/vjepa2/src')

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True,
    subset_file=None,
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,
    log_dir=None,
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
        )
    
    elif data.lower() == "surgical_videodataset":
        from src.datasets.video_surgical import make_surgical_videodataset
        dataset, data_loader, dist_sampler = make_surgical_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
        )

    return (data_loader, dist_sampler)

if __name__ == "__main__":
    
    
    from src.masks.multiseq_multiblock3d import MaskCollator
    from app.vjepa.transforms import make_transforms
    # from src.datasets.transforms import make_transforms
    
    cfgs_mask = {
        "mask": [
            {
                "aspect_ratio": [0.75, 1.5],
                "full_complement": False,
                "max_keep": None,
                "max_temporal_keep": 1.0,
                "num_blocks": 8,
                "spatial_scale": [0.15, 0.15],
                "temporal_scale": [1.0, 1.0],
            },
            {
                "aspect_ratio": [0.75, 1.5],
                "full_complement": False,
                "max_keep": None,
                "max_temporal_keep": 1.0,
                "num_blocks": 2,
                "spatial_scale": [0.7, 0.7],
                "temporal_scale": [1.0, 1.0],
            }
        ]
    }
    crop_size = 256
    patch_size = 16
    tubelet_size = 2
    dataset_fpcs=[16]
    ar_range = [0.75, 1.35]
    rr_scale = [0.3, 1.0]
    reprob = 0.0
    use_aa = False
    motion_shift = False
    dataset_type = "videodataset"
    
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=["/data/wjl/vjepa2/data_process/train_ssv2.csv"],
        batch_size=2,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=4,
        transform=transform,
        rank=0,
        world_size=1,
        datasets_weights=[1.0],
        persistent_workers=True,
        collator=mask_collator,
        num_workers=8,
        pin_mem=True,
        log_dir=None,
    )

    unsupervised_sampler.set_epoch(0)
    loader = iter(unsupervised_loader)
    sample = next(loader)
    import pdb; pdb.set_trace()
    
    
