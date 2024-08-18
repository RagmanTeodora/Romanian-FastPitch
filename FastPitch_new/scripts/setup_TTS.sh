#!/usr/bin/env bash

# Set current speaker or enable the process for all speakers:
SPEAKERS=(14)
#SPEAKERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)


# For anonymization set factor to 0
#export ANONYM_FACTOR=1
export ANONYM_FACTOR=0


# Set the fastpitch/model configuration:
export FASTPITCH_config="pred"
#export FASTPITCH_config="encoder"
#export FASTPITCH_config="decoder"


# Choose the phrases file:
#PHRASES="./phrases/phrases_TTS_clasic.txt"
PHRASES="./phrases/new_phr_anonym.txt"

# Customize output directory or leave it by default (out_synth/anonym*/option_FASTPITCHconfig/spk_SPEAKER):
OUTPUT_DIR=""

source ./scripts/inference_TTS.sh
