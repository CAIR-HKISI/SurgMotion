# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from logging import getLogger
from multiprocessing import Value

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        strategy_selection="random",  # "all": use all strategies, "random": randomly select one, "weighted": select by weight
        strategy_weights=None,  # weights for each strategy (only used when strategy_selection="weighted")
    ):
        super(MaskCollator, self).__init__()

        self.strategy_selection = strategy_selection
        self.strategy_weights = strategy_weights
        if strategy_weights is not None:
            # Normalize weights
            total_weight = sum(strategy_weights)
            self.strategy_weights = [w / total_weight for w in strategy_weights]
        else:
            # Default: equal weights
            self.strategy_weights = [1.0 / len(cfgs_mask)] * len(cfgs_mask) if len(cfgs_mask) > 0 else []

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                    num_windows=m.get("num_windows", None),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def _select_strategy_indices(self, num_strategies):
        """
        Select which strategy indices to use based on strategy_selection mode
        :param num_strategies: number of available strategies
        :return: list of strategy indices to use
        """
        if num_strategies == 0:
            return []
        
        if self.strategy_selection == "all":
            # Use all strategies
            return list(range(num_strategies))
        elif self.strategy_selection == "random":
            # Randomly select one strategy
            return [random.randint(0, num_strategies - 1)]
        elif self.strategy_selection == "weighted":
            # Select one strategy based on weights
            selected_idx = random.choices(range(num_strategies), weights=self.strategy_weights[:num_strategies], k=1)[0]
            return [selected_idx]
        else:
            # Default: use all strategies
            logger.warning(f"Unknown strategy_selection mode: {self.strategy_selection}, using 'all'")
            return list(range(num_strategies))

    def __call__(self, batch):

        # Batch: [buffer, label, clip_indices, case_id]
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            fpc = len(sample[-1][-1])
            filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_pred, collated_masks_enc = [], []
            
            # Select which strategies to use
            num_strategies = len(self.mask_generators[fpc])
            selected_indices = self._select_strategy_indices(num_strategies)
            
            for i in selected_indices:
                mask_generator = self.mask_generators[fpc][i]
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            
            fpc_collations += [(collated_batch, collated_masks_enc, collated_masks_pred)]

        return fpc_collations


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
        num_windows=None,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes
        self.inv_block = inv_block
        self.num_windows = num_windows  # number of windows for window-predict strategy (None means disabled)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def _generate_window_predict_masks(self, batch_size, generator):
        """
        Generate masks for window-predict strategy: divide frames into N windows,
        each window predicts the next window's tokens (like BERT next token prediction)
        :param batch_size: batch size
        :param generator: torch generator for reproducibility
        :return: (encoder_masks, predictor_masks)
        """
        num_windows = self.num_windows
        if num_windows is None or num_windows < 2:
            raise ValueError("num_windows must be >= 2 for window-predict strategy")
        
        # Calculate window size in temporal dimension
        # Each window spans: window_duration frames
        window_duration = self.duration // num_windows
        
        if window_duration == 0:
            raise ValueError(f"Cannot divide {self.duration} frames into {num_windows} windows")
        
        # For each sample, we generate num_windows-1 mask pairs:
        # - Window 1 predicts Window 2
        # - Window 2 predicts Window 3
        # - ...
        # - Window (num_windows-1) predicts Window num_windows
        # Each sample must have the same number of mask pairs for proper collation
        
        num_pairs = num_windows - 1
        collated_masks_pred, collated_masks_enc = [], []
        
        for _ in range(batch_size):
            # Generate masks for each window-predict pair
            sample_masks_pred, sample_masks_enc = [], []
            
            for win_idx in range(num_pairs):
                # Context window (visible to encoder)
                context_start = win_idx * window_duration
                context_end = (win_idx + 1) * window_duration
                
                # Target window (to be predicted)
                target_start = (win_idx + 1) * window_duration
                # Handle last window which might be larger if duration doesn't divide evenly
                if win_idx == num_pairs - 1:
                    target_end = self.duration  # Last window includes all remaining frames
                else:
                    target_end = (win_idx + 2) * window_duration
                
                # Create masks for this window pair
                # mask_e: encoder mask (context window) - shape (duration, height, width)
                mask_e = torch.zeros((self.duration, self.height, self.width), dtype=torch.int32)
                if context_end > context_start:  # Ensure valid range
                    mask_e[context_start:context_end, :, :] = 1
                mask_e = mask_e.flatten()  # Flatten to 1D: [duration * height * width]
                
                # mask_p: predictor mask (target window) - shape (duration, height, width)
                mask_p = torch.zeros((self.duration, self.height, self.width), dtype=torch.int32)
                if target_end > target_start:  # Ensure valid range
                    mask_p[target_start:target_end, :, :] = 1
                mask_p = mask_p.flatten()  # Flatten to 1D
                
                # Get indices where mask == 1 (visible/predict)
                mask_e_indices = torch.argwhere(mask_e == 1).squeeze(-1)
                mask_p_indices = torch.argwhere(mask_p == 1).squeeze(-1)
                
                # Ensure we have valid masks (all samples must have same number of pairs)
                if len(mask_e_indices) == 0:
                    # If context is empty, use first patch as context
                    mask_e_indices = torch.tensor([0], dtype=torch.long)
                if len(mask_p_indices) == 0:
                    # If target is empty, use last patch as target
                    last_patch_idx = self.duration * self.height * self.width - 1
                    mask_p_indices = torch.tensor([last_patch_idx], dtype=torch.long)
                
                sample_masks_enc.append(mask_e_indices)
                sample_masks_pred.append(mask_p_indices)
            
            # Ensure all samples have the same number of pairs
            assert len(sample_masks_enc) == num_pairs, f"Expected {num_pairs} mask pairs, got {len(sample_masks_enc)}"
            assert len(sample_masks_pred) == num_pairs, f"Expected {num_pairs} mask pairs, got {len(sample_masks_pred)}"
            
            # Append all mask pairs for this sample
            collated_masks_enc.append(sample_masks_enc)
            collated_masks_pred.append(sample_masks_pred)
        
        # The masks are now lists of lists
        # For each sample: [mask1, mask2, ..., maskN] where N = num_windows - 1
        # We need to transpose to: [[all_mask1s], [all_mask2s], ..., [all_maskNs]]
        # Then collate each group
        
        if len(collated_masks_enc) == 0:
            raise ValueError("No valid masks generated")
        
        # Transpose: from [sample1_masks, sample2_masks, ...] 
        # to [[mask1_for_all_samples], [mask2_for_all_samples], ...]
        # All samples should have the same number of pairs (num_windows - 1)
        transposed_enc = [[] for _ in range(num_pairs)]
        transposed_pred = [[] for _ in range(num_pairs)]
        
        for sample_enc, sample_pred in zip(collated_masks_enc, collated_masks_pred):
            # All samples should have the same number of pairs
            assert len(sample_enc) == num_pairs and len(sample_pred) == num_pairs, \
                f"All samples must have {num_pairs} mask pairs"
            for pair_idx in range(num_pairs):
                transposed_enc[pair_idx].append(sample_enc[pair_idx])
                transposed_pred[pair_idx].append(sample_pred[pair_idx])
        
        # Collate each group
        collated_enc = [torch.utils.data.default_collate(enc_group) for enc_group in transposed_enc]
        collated_pred = [torch.utils.data.default_collate(pred_group) for pred_group in transposed_pred]
        
        if self.inv_block:
            return collated_pred, collated_enc  # predict context from block
        else:
            return collated_enc, collated_pred

    def _generate_block_mask_masks(self, batch_size, generator):
        """
        Generate masks for the default block mask strategy
        This is the original JEPA masking strategy that randomly masks spatial-temporal blocks
        :param batch_size: batch size
        :param generator: torch generator for reproducibility
        :return: (encoder_masks, predictor_masks)
        """
        # Sample block size using seed for reproducibility
        p_size = self._sample_block_size(
            generator=generator,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                # Generate mask by sampling npred blocks
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                # Extract predictor mask (masked regions) and encoder mask (visible regions)
                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # Apply max_keep constraint if specified
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        # Truncate masks to minimum size for efficient batching
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        
        # Handle complement masks
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        # Collate masks for batch processing
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc  # predict context from block
        else:
            return collated_masks_enc, collated_masks_pred

    def __call__(self, batch_size):
        """
        Main entry point for mask generation. Routes to appropriate masking strategy:
        
        Masking Strategies (in priority order):
        1. Window-predict strategy (num_windows): Divide frames into N windows, each window predicts the next
           - Only used if num_windows is specified in config
        2. Block mask strategy (default): Randomly mask spatial-temporal blocks (original JEPA strategy)
           - Used when num_windows is None (backward compatible with old configs)
           - This is the default strategy for compatibility with existing configs
        
        :param batch_size: batch size
        :return: (encoder_masks, predictor_masks)
            - encoder_masks: indices of context tokens (visible to encoder)
            - predictor_masks: indices of target tokens (to be predicted)
        """
        # Initialize random generator with seed for reproducibility
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Strategy 1: Window-predict strategy (BERT-like next window prediction)
        # Only activate if num_windows is explicitly set in config
        if self.num_windows is not None:
            return self._generate_window_predict_masks(batch_size, g)
        
        # Strategy 2: Default block mask strategy (original JEPA)
        # This is the default behavior for backward compatibility
        # Old configs without num_windows will automatically use this strategy
        return self._generate_block_mask_masks(batch_size, g)
