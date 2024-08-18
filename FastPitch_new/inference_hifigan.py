# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import json

from hifi_gan.env import AttrDict
from hifi_gan.meldataset import MAX_WAV_VALUE
from hifi_gan.models import Generator

import argparse
import models
import time
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import utils
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import TextProcessing
#from waveglow import model as glow
#from waveglow.denoiser import Denoiser

#sys.modules['glow'] = glow

import warnings
warnings.filterwarnings("ignore", message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true', help='')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--hifi-checkpoint-file', type=str,
                        help='Full path to the HIFI-GAN generator')
#    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
#                        help='WaveGlow sigma')
#    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
#                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--p-arpabet', type=float, default=0.0, help='')
    parser.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                        help='')
    parser.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                        help='')
    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['english_cleaners_v2'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the model.')

    return parser


def load_model_from_ckpt(checkpoint_path, ema, model):

    checkpoint_data = torch.load(checkpoint_path)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)

#    if model_name == "WaveGlow":
#        for k, m in model.named_modules():
#            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
#
#        model = model.remove_weightnorm(model)

    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_checkpoint_hifi(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c:f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    #order = np.argsort([-t.size(0) for t in fields['text']])
    order = [i for i,t in enumerate(fields['text'])]
    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    for t in fields['text']:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=cuda)


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.output is not None:
        Path(args.output).mkdir(parents=False, exist_ok=True)

    log_fpath = args.log_file or str(Path(args.output, 'nvlog_infer.json'))
#    log_fpath = unique_log_fpath(log_fpath)

    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.fastpitch != 'SKIP':
        generator = load_and_setup_model(
            'FastPitch', parser, args.fastpitch, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=args.ema,
            jitable=args.torchscript)
        print("Loadinggggg the modeeeel")

        if args.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None


    print("Loaded FastPitch model")

    ## HIFI GAN                                                                                                                                                                   
    config_file = os.path.join(os.path.split(args.hifi_checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    generator_hifi = Generator(h).to(device)
    state_dict_g = load_checkpoint_hifi(args.hifi_checkpoint_file, device)
    print ("Loading HiFi-GAN checkpoint from: ", args.hifi_checkpoint_file)
    generator_hifi.load_state_dict(state_dict_g['generator'])
    generator_hifi.eval()
    generator_hifi.remove_weight_norm()
    ## Done HifiGAN 

    print("Loaded hifigan")
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    fields = load_fields(args.input)
    batches = prepare_input_sequence(
        fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
        args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)


    gen_kw = {'pace': args.pace,
              'speaker': args.speaker,
              'pitch_tgt': None,
              'pitch_transform': build_pitch_transformation(args)}

    print("ARGS SPEAKER: ", args.speaker, args.n_speakers)
    if args.torchscript:
        gen_kw.pop('pitch_transform')
        print('NOTE: Pitch transforms are disabled with TorchScript')




    log_enabled = 0
    reps = 1
    import common.stft
    stft = common.stft.STFT(filter_length=1024,
                                hop_length=int(1025/4),
                                win_length=1024).cuda()

    for i,b in enumerate(batches):
        with torch.no_grad():
            mel, mel_lens, *_ = generator(b['text'], **gen_kw)
            y_g_hat = generator_hifi(mel)

            audio = y_g_hat
            audio = audio / torch.max(torch.abs(audio))
            fname = b['output'][i] if 'output' in b else f'audio_{str(i).zfill(3)}.wav'
            audio_path = Path(args.output, fname)
            print ("Wrote output to: ", audio_path)
            write(audio_path, args.sampling_rate, audio.cpu().numpy())



if __name__ == '__main__':
    main()
