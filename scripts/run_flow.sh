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


# python motion_seg/stat_flow_multi.py --videos data/flow_output/SurgicalAction160/01_abdominal_access/01_03_flow.mp4 data/flow_output/SurgicalAction160/02_injection/02_01_flow.mp4  data/flow_output/SurgicalAction160/03_cutting/03_03_flow.mp4 data/flow_output/SurgicalAction160/08_suction/08_04_flow.mp4  data/flow_output/SurgicalAction160/12_knotting/12_05_flow.mp4 data/flow_output/SurgicalAction160/15_endobag-in/15_04_flow.mp4  --output_dir results


# video1=data/flow_output/M2CAI2016/train_dataset/workflow_video_01_flow.mp4
# video2=data/flow_output/AutoLaparo/01_flow.mp4

# python motion_seg/stat_flow_multi.py --videos $video1 $video2 --output_dir results_v2