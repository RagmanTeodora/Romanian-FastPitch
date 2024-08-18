#!/usr/bin/env bash

export OMP_NUM_THREADS=1

export FREEZE_ENC="False"
export FREEZE_DEC="False"

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=1}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=false}
: ${SEED:=""}

: ${LEARNING_RATE:=0.1}

# Audiopaths
TRAIN_FILELIST=/home/tragman/licenta/FastPitch_new/new_speakers/$SPEAKER/${SPEAKER}_metadata_train.txt
VAL_FILELIST=/home/tragman/licenta/FastPitch_new/new_speakers/$SPEAKER/${SPEAKER}_metadata_eval.txt

# Adjust these when the amount of data changes
: ${EPOCHS:=560}
: ${EPOCHS_PER_CHECKPOINT:= 15}
: ${WARMUP_STEPS:=1000}
: ${KL_LOSS_WARMUP:=100}

# Train a mixed phoneme/grapheme model
: ${PHONE:=false}
# Enable energy conditioning
: ${ENERGY:=true}
: ${TEXT_CLEANERS:=basic_cleaners}
# Add dummy space prefix/suffix is audio is not precisely trimmed
: ${APPEND_SPACES:=false}

: ${LOAD_PITCH_FROM_DISK:=true}
: ${LOAD_MEL_FROM_DISK:=true}

# For multispeaker models, add speaker ID = {0, 1, ...} as the last filelist column
: ${NSPEAKERS:=18}
: ${SAMPLING_RATE:=22050}

# Adjust env variables to maintain the global batch size: NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 256.
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

for FASTPITCH_config in "${configs[@]}"; do
    export FASTPITCH_config
    CHECKPOINT_PATH="./OUTPUT_MODELS/output_${FASTPITCH_config}_only/FastPitch_checkpoint_500.pt"
    
    if [ "${SPK_source}" = "ext" ]; then
	DATASET_PATH=/home/tragman/licenta/FastPitch_new/new_speakers/$SPEAKER/wavs22/
    else
	DATASET_PATH=/mnt/student-share/data4teodora/SWARA_mels_pitch_meta/
    fi
    
    OUTPUT_DIR=./OUTPUT_MODELS/voice_cloning/${SPEAKER}_finetuned_${FASTPITCH_config}/
    LOG_FILE=${OUTPUT_DIR}/nvlog.json # --ramane la fel dar trb mentionat dupa OUTPUT_DIR
    FASTPITCH=./OUTPUT_MODELS/output_${FASTPITCH_config}_only/FastPitch_checkpoint_500.pt

    echo "SPEAKER: $SPEAKER"
    echo "FASTPITCH CONFIGURATION: $FASTPITCH_config"
    echo "DATASET: $DATASET_PATH"
    echo "TRAIN_SET: $TRAIN_FILELIST"
    echo "VAL_SET: $VAL_FILELIST"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    
    ARGS=""
    ARGS+=" --cuda"
    ARGS+=" -o $OUTPUT_DIR"
    ARGS+=" --log-file $LOG_FILE"
    ARGS+=" --dataset-path $DATASET_PATH"
    ARGS+=" --training-files $TRAIN_FILELIST"
    ARGS+=" --validation-files $VAL_FILELIST"
    ARGS+=" -bs $BATCH_SIZE"
    ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
    ARGS+=" --optimizer adam"
    ARGS+=" --epochs $EPOCHS"
    ARGS+=" --checkpoint-path $CHECKPOINT_PATH"
    ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"
    #ARGS+=" --resume"
    ARGS+=" --warmup-steps $WARMUP_STEPS"
    ARGS+=" -lr $LEARNING_RATE"
    ARGS+=" --weight-decay 1e-6"
    ARGS+=" --grad-clip-thresh 1000.0"
    ARGS+=" --dur-predictor-loss-scale 0.1"
    ARGS+=" --pitch-predictor-loss-scale 0.1"
    ARGS+=" --pitch-mean 228.67"
    ARGS+=" --pitch-std 46.91"

    # Autoalign & new features
    ARGS+=" --kl-loss-start-epoch 0"
    ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"
    ARGS+=" --text-cleaners $TEXT_CLEANERS"
    ARGS+=" --symbol-set romanian_fullPred2"
    ARGS+=" --n-speakers $NSPEAKERS"

    [ "$AMP" = "true" ]                && ARGS+=" --amp"
    [ "$ENERGY" = "true" ]             && ARGS+=" --energy-conditioning"
    [ "$SEED" != "" ]                  && ARGS+=" --seed $SEED"
    [ "$LOAD_MEL_FROM_DISK" = true ]   && ARGS+=" --load-mel-from-disk"
    [ "$LOAD_PITCH_FROM_DISK" = true ] && ARGS+=" --load-pitch-from-disk"
    [ "$PITCH_ONLINE_DIR" != "" ]      && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch
    [ "$PITCH_ONLINE_METHOD" != "" ]   && ARGS+=" --pitch-online-method $PITCH_ONLINE_METHOD"
    [ "$APPEND_SPACES" = true ]        && ARGS+=" --prepend-space-to-text"
    [ "$APPEND_SPACES" = true ]        && ARGS+=" --append-space-to-text"

    if [ "$SAMPLING_RATE" == "44100" ]; then
	ARGS+=" --sampling-rate 44100"
	ARGS+=" --filter-length 2048"
	ARGS+=" --hop-length 512"
	ARGS+=" --win-length 2048"
	ARGS+=" --mel-fmin 0.0"
	ARGS+=" --mel-fmax 22050.0"

    elif [ "$SAMPLING_RATE" != "22050" ]; then
	echo "Unknown sampling rate $SAMPLING_RATE"
	exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    python3 train_voice_cloning.py $ARGS "$@"
done
