# To prepare new bleeding dataset for FDTN probing

# 1. Prepare the data
#python data_process/bleeding_prepare.py --step parse
#python data_process/bleeding_prepare.py --step extract
#python data_process/bleeding_prepare.py --step split
#python data_process/bleeding_prepare.py --step csv

# or 
python data_process/bleeding_prepare.py --step all

# 2. Prepare the config
python data_process/gen_clips.py --base_data_path data/Surge_Frames/Bleeding_Dataset --window_size 64 --stride 1 --fps 1

# Different split ratios v1
python data_process/bleeding_prepare.py --step parse
python data_process/bleeding_prepare.py --step split
ln -s /home/user01/NSJepa/data/Surge_Frames/Bleeding_Dataset/frames /home/user01/NSJepa/data/Surge_Frames/Bleeding_Dataset_70_30/frames
python data_process/bleeding_prepare.py --step csv
python data_process/gen_clips.py --base_data_path data/Surge_Frames/Bleeding_Dataset_70_30 --window_size 64 --stride 1 --fps 1

# Different split ratios v2
python data_process/bleeding_prepare.py --out_dir data/Surge_Frames/Bleeding_Dataset_60_40 --step parse
python data_process/bleeding_prepare.py --out_dir data/Surge_Frames/Bleeding_Dataset_60_40 --step split --split_ratios 0.60,0.40,0
ln -s /home/user01/NSJepa/data/Surge_Frames/Bleeding_Dataset/frames /home/user01/NSJepa/data/Surge_Frames/Bleeding_Dataset_60_40/frames
python data_process/bleeding_prepare.py --out_dir data/Surge_Frames/Bleeding_Dataset_60_40 --step csv --split_ratios 0.60,0.40,0
python data_process/gen_clips.py --base_data_path data/Surge_Frames/Bleeding_Dataset_60_40 --window_size 64 --stride 1 --fps 1