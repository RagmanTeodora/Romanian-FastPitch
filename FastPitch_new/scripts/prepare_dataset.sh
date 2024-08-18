#!/usr/bin/env bash

set -e

current_speaker="SOslow" # modify here, all files have the same structure

: ${DATA_DIR:=/home/tragman/licenta/FastPitch_new/new_speakers/${current_speaker}/wavs22}
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists /home/tragman/licenta/FastPitch_new/new_speakers/${current_speaker}/meta_4_pitch_mels_${current_speaker}.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
