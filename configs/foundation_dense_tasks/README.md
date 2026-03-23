# Foundation dense task config conventions

- `eval_name`: must match the dynamic eval package key consumed by `evals.scaffold` (e.g., `depth_estimation_frozen`, `segmentation_frozen`).
- `dataset_type`: use a stable dataset identifier so loaders can dispatch dataset-specific behavior without hard-coded paths.
- Scope: V-JEPA2-first config tree under `vjepa2/` (initially `C3VD/` and `CVC-12K/` only).
