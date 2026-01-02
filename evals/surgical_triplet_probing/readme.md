### Run the codes:
```markdown
configs/
├── Triplet/
|   |──cholect50_ivt.yaml
```
Modify configs:

**folder:**    
    <your_folder> -------- *e.g. SLURM_Logs_ivt/test_log*

**dataset_train:**    
    <your_train .csv file> -------- *e.g. - ./NSJepa-triplet_recognition/data/CholecT50/clips_16f/train_dense_16f_detailed.csv*

**dataset_val:**    
    <your_test .csv file> -------- *e.g. - ./NSJepa-triplet_recognition/data/CholecT50/clips_16f/test_dense_16f_detailed.csv*

**checkpoint:**     
    <your_pretained_weights, large/huge/giant> -------- *e.g. ./logs9/surgical_cpt_vitl16-256px-64f_lr1e-20_epoch_21-dataset/latest.pt*

```markdown
evals/
├── surgical_triplet_probing/
|   |── cholect50.py
|   |── data_loader.py
|   |── eval_ivt.py # for single epoch
|   |── eval_ivt_multiepoch.py # for multiple epoch (run this for 5 - 10 epochs)
|   |── models.py
|   |── utils.py
├── main_ivt.py
├── scaffold_ivt.py
├── srun_ivt.py # run this
```

Modify srun_ivt.py

**source**     
    <your_environment> -------- *e.g. ./NSJepa-triplet_recognition/.venv/bin/activate*

**--fname**     
    <your .yaml folder> -------- *e.g. "./NSJepa-triplet_recognition/configs/${TASK}/${FNAME}"*
