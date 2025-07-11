set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
cd /mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/internvl_chat
source /mnt/pfs-gv8sxa/tts/dhg/sunshibo/envs/intervl/bin/activate

conda activate /mnt/pfs-gv8sxa/tts/dhg/sunshibo/envs/intervl
which python

# /mnt/pfs-gv8sxa/tts/dhg/sunshibo/envs/intervl/bin/python -m pip install timm==0.9.12

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=20052
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export PYTHONPATH="/mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL:/mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/internvl_chat:${PYTHONPATH}"

OUTPUT_DIR='/mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
/mnt/pfs-gv8sxa/tts/dhg/sunshibo/envs/intervl/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=0.0.0.0 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/pretrained/InternVL2-8B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/mnt/pfs-gv8sxa/tts/dhg/sunshibo/InternVL/internvl_chat/shell/data/actplan.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
