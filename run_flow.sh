# export CUDA_VISIBLE_DEVICES=0
# python motion_seg/raft_batch.py --input_dir data/SurgicalAction160 --output_dir data/flow_output/SurgicalAction160

# export CUDA_VISIBLE_DEVICES=0
# python motion_seg/raft_batch.py --input_dir data/autolaparo/task1/videos --output_dir  data/flow_output/AutoLaparo 

# GPU=1
# input_dir=data/micai2016
# output_dir=data/flow_output/M2CAI2016

GPU=1
input_dir=data/micai2016
output_dir=data/flow_output/M2CAI2016

export CUDA_VISIBLE_DEVICES=$GPU
python motion_seg/raft_batch.py --input_dir $input_dir --output_dir $output_dir
