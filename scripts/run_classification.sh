export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

master_port=29501
num_process=4

# EthanolConcentration

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.02 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 96 \
  --padding 32 \
  --stride 32 \
  --use_multi_gpu \
  --seq_len 1751 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# FaceDetection

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --kernel_width 24 \
  --padding 4 \
  --stride 4 \
  --use_multi_gpu \
  --seq_len 62 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 2
  
# Handwriting

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 24 \
  --padding 4 \
  --stride 4 \
  --use_multi_gpu \
  --seq_len 152 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# Heartbeat

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.05 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 12 \
  --padding 4 \
  --stride 4 \
  --use_multi_gpu \
  --seq_len 405 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# JapaneseVowels

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.1 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 8 \
  --padding 2 \
  --stride 2 \
  --use_multi_gpu \
  --seq_len 29 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# PEMS-SF

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.02 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 32 \
  --padding 8 \
  --stride 8 \
  --use_multi_gpu \
  --seq_len 144 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# SelfRegulationSCP1

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 48 \
  --padding 16 \
  --stride 16 \
  --use_multi_gpu \
  --seq_len 896 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# SelfRegulationSCP2

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 32 \
  --padding 8 \
  --stride 8 \
  --use_multi_gpu \
  --seq_len 1152 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# SpokenArabicDigits

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 30 \
  --patience 10 \
  --kernel_width 3 \
  --padding 1 \
  --stride 1 \
  --use_multi_gpu \
  --seq_len 93 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4

# UWaveGestureLibrary

accelerate launch --multi_gpu --mixed_precision=no --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model GPTJ \
  --llm_dim 4096 \
  --data UEA \
  --batch_size 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10 \
  --kernel_width 128 \
  --padding 32 \
  --stride 32 \
  --use_multi_gpu \
  --seq_len 315 \
  --model_comment reg_classification \
  --gradient_accumulation_steps 4
