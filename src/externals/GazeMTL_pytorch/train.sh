#!/bin/bash
# Modern PyTorch training script for GazeMTL

SOURCE=cxr2
TASK=gaze_mtl
DATA_DIR=/home/ubuntu/hung.nh2/gaze/dicom/dicom_raw

# Configuration based on source
if [ $SOURCE == "mets" ]; then
    LR=0.0001 
    L2=0.00001
    Vscale=0.2
    gaze_mtl_task=diffusivity
    tw="0.5"

elif [ $SOURCE == "cxr" ]; then
    LR=0.0001 
    L2=0.0001 
    Vscale=0.2
    gaze_mtl_task=loc
    tw="0.5"

elif [ $SOURCE == "cxr2" ]; then
    LR=0.0001 
    L2=0.01 
    Vscale=0.2
    gaze_mtl_task=diffusivity 
    tw="1.0" 
fi

# Set log path
if [ $TASK == "gaze_mtl" ]; then
    LOG_PTH=/home/ubuntu/hung.nh2/gaze/gaze-01-baseline/src/externals/GazeMTL_pytorch/logs/$TASK/$SOURCE/$gaze_mtl_task
else
    LOG_PTH=/home/ubuntu/hung.nh2/gaze/gaze-01-baseline/src/externals/GazeMTL_pytorch/logs/$TASK/$SOURCE
fi

# Train for multiple seeds
for seed in 0  # Add more seeds: 101 102 103 104 105
do
    python train.py \
        --source $SOURCE \
        --train_scale 1 \
        --val_scale $Vscale \
        --data_dir $DATA_DIR \
        --gaze_mtl_task $gaze_mtl_task \
        --pretrained \
        --task_weights "$tw" \
        --log_path $LOG_PTH/seed_$seed \
        --seed $seed \
        --n_epochs 15 \
        --lr $LR \
        --l2 $L2 \
        --batch_size 32 \
        --lr_scheduler cosine_annealing \
        --optimizer adam \
        --num_workers 8
done

