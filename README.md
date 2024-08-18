# Romanian FastPitch & unseen speaker adaptation
This repository provides a version of the [FastPitch][1] model, adapted for Romanian and integrated with a module for TTS using unseen speaker identities.

## Model Overview & features
This project was built on top of the [FastPitch][1] structure, with the following alterations:
- this version is adapted for Romanian
- supports 18 learned speakers from the [SWARA Speech Corpus][2]
- _anonymous speaker feature_ - a **fully generated** voice can be used for privacy preservation
- _new speakers TTS feature_ - the TTS function can now be adapted to **unseen speakers**, requiring a short voice sample for reference and 300 additional training steps

## Prerequisites
###### For TTS and anonymous speaker:
All the necessary checkpoints are available in `./OUTPUT_MODELS`
###### For new speaker adaptation (VC)
This process requires resuming the training for an additional 300 steps. While the process is not computationally intensive, it runs on CUDA, so you must ensure that your setup includes a compatible version of [PyTorch](https://pytorch.org/) and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive), and that you have access to GPU resources before starting the training/inference processes. You can find a good starting point [here](https://pytorch.org/get-started/locally/). If you don't have access to GPU resources, we've included a Colab notebook where you can explore all the features of this project. Just make sure to go through `Demo_t1_preproc.ipynb` before `Demo_t2_finetune.ipynb`.

## Installation instructions

**1. Copy the repository content**
- Option 1: Directly download the zip file. The extracted file will become your root path (for example, FastPitch_new in this case, but you can rename it)
- Option 2: Create a copy of this repo:
   ```
   git clone https://github.com/RagmanTeodora/Romanian-FastPitch.git
   ```
**2. Install necessary packages**
- use the `./requirements.txt` file to install the dependencies on your machine/environment
  ```
  cd ./FastPitch_new
  pip install -r requirements.txt
  ```

## Usage instructions

### Text-to-speech synthesis
1. Open `./scripts/setup_TTS.sh` with a text editor
2. Modify the `SPEAKERS` variable with a number from 0 to 17. These IDs correspond to the speakers according to the following table:

| Speaker ID |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  | 13  | 14  | 15  | 16  | 17  |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SWARA ID   | bas | bea | cau | dcs | ddm | eme | fds | htm | ips | mar | pcs | pmm | pss | rms | sam | sds | sgs | tss |

--> [reference samples](https://speech.utcluj.ro/swarasc/samples/index.html) <--

3. Set the anonym flag:
   - for the **TTS** task -- `ANONYM_FACTOR=1`
   - for the **anonymous speaker** feature -- `ANONYM_FACTOR=0` 
4. Choose your preferred conditioning structure. We recommend using the predictors-based version of the model:
   ```
   FASTPITCH_config="pred"
   ```
5. Choose the phrases file path
   - `PHRASES`="./phrases/..."
     
6. Run the setup script from the root file
   ```
   ./scripts/setup_TTS.sh
   ```

### Anonymous speaker synthesis
Follow the same steps as for Text-to-speech synthesis, but set the anonymous flag to 0 at step 3
```
ANONYM_FACTOR=0
```

### Unseen speaker synthesis
This feature allows the user to reproduce the identity of new speakers based on a short 30second voice sample and a few additional training steps
1. Use the t1 Colab notebook to record/load your voice samples and prepare the custom dataset for the fine-tuning process
2. Download the generated files and load them accordingly on your machine/environment
   - create a new folder in `./new_speakers/{your_ID}` named exactly as your speaker ID
   - populate it with:
       - the 22kHz recordings -- `./new_speakers/{your_ID}/wavs22`
       - the embedding .npy file -- `./new_speakers/{your_ID}/{your_ID}_18x384.npy `
       - the dataset preparation metadata file -- `./new_speakers/{your_ID}/meta_4_pitch_mels_{your_ID}.txt`
       - the train and validation metadata files -- `./new_speakers/{your_ID}/{your_ID}_metadata_eval.txt/train.txt`
    - load your phrases file into the `./phrases/` directory
3. Open `./scripts/prepare_dataset.sh` with a text editor and edit the speaker ID -- run this file from the root directory
4. Open `./scripts/setup_VC.sh` and edit the following variables; then run the file from the root directory
   - `SPEAKER="{your_ID}"`
   - `configs="{pred/encoder/decoder}"` -- we recommend the "pred" option for best results (default option)
   - `PHRASES="./phrases/..."`
  ##### Optional steps:
  5.  Open `./scripts/train_freeze.sh` and edit the following variables:
     - you can choose which layers will be frozen during the fine-tuning process; we **strongly** recommend setting only `FREEZE_DUR` to `true` when using the "pred" configuration
     - you can modify the number of additional training steps through the `EPOCHS` variable. The default setting is 560.
  ##### **IMPORTANT -- inference w/out fine-tuning**
  If you already fine-tuned the model for the new speaker identity, and want to proceed with the inference process multiple time on different phrases file, go to `./scripts/train_freeze.sh` and comment the following lines:
  ```
  mkdir -p "$OUTPUT_DIR"
  python3 train_voice_cloning.py $ARGS "$@"
  ```

## Limitations
The models obtained within this project have the following issues:
- the support phrase must explicitly include the expanded form of numbers, particularly when the numeral (e.g., 1 = o/un, 2=doi/două) depends on the gender of the referenced object (as is the case in the Romanian language)
- clipped voice samples used as references for unseen speaker adaptation will lead to clipped generated output
- speaking at a faster pace may result in less intelligible output; try to maintain a consistent pace when recording your reference samples

[1]:https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
[2]:https://speech.utcluj.ro/swarasc/
