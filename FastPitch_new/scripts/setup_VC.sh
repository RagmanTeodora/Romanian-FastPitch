#!/usr/bin/env bash

# Set current speaker of enable the process for all speakers:
#SPEAKER="SO"
#SPEAKER="RT"
#SPEAKER="bal"
#SPEAKER="bgl"
SPEAKER="bim" ###FREEZE
#SPEAKER="bmm"
#SPEAKER="dol" ###FREEZE
#SPEAKER="nll"
#SPEAKER="SOslow"

export SPEAKER


# Speaker source: external or SWARA2.0
SW2=("bal" "bgl" "bim" "bmm" "dol" "nll")
if [[ " ${SW2[*]} " == *" $SPEAKER "* ]]; then
    SPK_source="int"
else
    SPK_source="ext"
fi

# Set the fastpitch/model configuration
configs=("pred")
#configs=("pred" "encoder")
#"decoder")


#Text file path for inference phrases:
#PHRASES="phrases/johannah3.txt"
PHRASES="phrases/VC_test.txt"


#!!! do not modify the following lines: they are responsible for running the finetuning process, followed by the inference one once the training was successfully completed

#source ./scripts/train_VC.sh
source ./scripts/train_freeze.sh

if [ $? -eq 0 ]; then
    echo -e "\n#######################################################\nThe model was successfully adapted to speaker ${SPEAKER}\n#######################################################\n"
    #source ./scripts/inference_VC.sh
    source ./scripts/inference_freeze.sh
else
    echo "The finetuning procedure failed. See error log for more details. Could not procees with inference."
fi
