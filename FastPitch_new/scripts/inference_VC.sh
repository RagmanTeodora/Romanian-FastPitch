#!/usr/bin/env bash

export ANONYM_FACTOR=1
SPK=$SPEAKER

SPEAKER=0 # all speakers == $SPK, but an int value is required for the embedding matrix
: ${NUM_SPEAKERS:=18}

: ${HIFI:="0.MODELS/g_02500000"}
: ${BATCH_SIZE:=1}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=false}
: ${ENERGY:=true}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

for FASTPITCH_config in "${configs[@]}"; do
    FASTPITCH="./OUTPUT_MODELS/voice_cloning/${SPK}_finetuned_${FASTPITCH_config}/FastPitch_checkpoint_560.pt"
    OUTPUT_DIR="out_synth/voice_clones/${SPK}/${FASTPITCH_config}"
    LOG_FILE="$OUTPUT_DIR/nvlog_infer.json"

    echo "Current speaker: ${SPK}"
    echo "Phrases file: ${PHRASES}"
    echo "Model in use: ${FASTPITCH}"
    echo "Audio files saved to: ${OUTPUT_DIR}"
    
    ARGS=""
    #ARGS+=" --save-mels"
    ARGS+=" -i $PHRASES"
    ARGS+=" -o $OUTPUT_DIR"
    ARGS+=" --log-file $LOG_FILE"
    ARGS+=" --fastpitch $FASTPITCH"
    ARGS+=" --hifi-checkpoint-file $HIFI"
    ARGS+=" --batch-size $BATCH_SIZE"
    ARGS+=" --repeats $REPEATS"
    ARGS+=" --warmup-steps $WARMUP"
    ARGS+=" --speaker $SPEAKER"
    ARGS+=" --n-speakers $NUM_SPEAKERS"
    ARGS+=" --symbol-set romanian_fullPred2"
    ARGS+=" --text-cleaners basic_cleaners"
    #ARGS+=" --pitch-transform-amplify  0.8"
    #ARGS+=" --pace 0.8"
    #ARGS+=" --pitch-transform-flatten"
    #ARGS+=" --pitch-transform-shift +20"

    [ "$CPU" = false ]          && ARGS+=" --cuda"
    [ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
    [ "$AMP" = true ]           && ARGS+=" --amp"
    [ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 0.0"
    [ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
    [ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

    mkdir -p "$OUTPUT_DIR"
    
    python3 inference_hifigan.py $ARGS "$@"
done
