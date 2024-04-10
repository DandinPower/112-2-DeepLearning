import glob, os
import numpy as np
from scipy.io import wavfile
from audiomentations import Compose, SomeOf, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Shift, AddBackgroundNoise, AddShortNoises, PolarityInversion, ApplyImpulseResponse
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
from tqdm import tqdm
import os 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

InPath = "datasets/transform/train"
OutPath = "datasets/noisy/train"
BACKGROUNDNOISE_FOLDER = "datasets/downloads/background_noise/ESC-50-master/audio"
IMPULSERESPONSE_FOLDER = "datasets/builtin/echo_thief_impulse_response"

def check_path():
    if os.path.exists(OutPath):
        os.system("rm -rf " + OutPath)
    os.makedirs(OutPath)

from multiprocessing import Pool

def process_file(args):
    file, augment1, augment2 = args
    if not file.endswith(".wav"):
        return
    samples, sample_rate = load_sound_file(
        os.path.join(InPath, file), sample_rate=None
    )
    augmented_samples1 = augment1.augment(samples)
    augmented_samples2 = augment2(samples=augmented_samples1[0], sample_rate=sample_rate)
    wavfile.write(
        os.path.join(OutPath, file), rate=sample_rate, data=augmented_samples2
    )

def main():
    augment1 = naf.Sometimes([
    naa.VtlpAug(sampling_rate=16000, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
    ], aug_p=0.4)

    augment2 = Compose([
        AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=30, p=0.2),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
        AddBackgroundNoise(
            sounds_path=BACKGROUNDNOISE_FOLDER,
            min_snr_in_db=10,
            max_snr_in_db=30.0,
            p=0.4),
        AddShortNoises(
            sounds_path=BACKGROUNDNOISE_FOLDER,
            min_snr_in_db=10,
            max_snr_in_db=30.0,
            noise_rms="relative_to_whole_input",
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            p=0.3),
        ApplyImpulseResponse(
                ir_path=IMPULSERESPONSE_FOLDER, p=0.4
            )
    ])

    with Pool() as p:
        p.map(process_file, [(file, augment1, augment2) for file in os.listdir(InPath)])

if __name__ == "__main__":
    check_path()
    main()