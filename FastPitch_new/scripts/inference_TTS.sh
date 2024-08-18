#!/usr/bin/env bash

: ${FASTPITCH:="./OUTPUT_MODELS/output_${FASTPITCH_config}_only/FastPitch_checkpoint_500.pt"}
: ${NUM_SPEAKERS:=18}

: ${HIFI:="0.MODELS/g_02500000"}
: ${BATCH_SIZE:=1}
#: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=false}
: ${ENERGY:=true}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

#if [ "$ANONYM_FACTOR" = '0' ]; then
#    SPEAKERS=(0)
#fi

if [ "${#SPEAKERS[@]}" -eq 1 ]; then
    SPEAKER=$SPEAKERS
    if [ "$OUTPUT_DIR" = "" ]; then
	if [ "$ANONYM_FACTOR" = '1' ]; then
	    OUTPUT_DIR="out_synth/TTS/${FASTPITCH_config}/$SPEAKER"
	else
	    OUTPUT_DIR="out_synth/anonym/${FASTPITCH_config}/$SPEAKER"
	fi
    fi
fi    
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}

for SPEAKER in "${SPEAKERS[@]}"; do
    
    if [ "${#SPEAKERS[@]}" -gt 1 ]; then
	if [ "$ANONYM_FACTOR" = '1' ]; then
	    OUTPUT_DIR="out_synth/TTS/${FASTPITCH_config}/$SPEAKER"
	else
	    OUTPUT_DIR="out_synth/anonym/${FASTPITCH_config}/$SPEAKER"
	fi
    fi
    
    echo "SPEAKER: $SPEAKER"
    echo "FASTPITCH CONFIGURATION: $FASTPITCH_config"
    echo "ANONYM FLAG: $ANONYM_FACTOR"
    echo "PHRASES: $PHRASES"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    
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
