#! /bin/bash


MASTER_PORT=1256 python -m evals.main \
                --fname configs/probing_cholec80/cholec80_vitl_cpt_attentive_64f_debug.yaml \
                --devices "cuda:2"  --val_only  \
                --checkpoint "logs/cpt_cholec80/cpt_vitl16-256px-64f_lr1e-4_epoch-20/latest.pt" 