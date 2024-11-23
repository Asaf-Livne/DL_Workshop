import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
import torchaudio
from pydub import AudioSegment
from io import BytesIO
import wave
import torchaudio

MAX_WAV_VALUE = 32768.0


def degrade_quality_bytes(y, quality='16k'):
    buf_input = AudioSegment.from_file(y, format="wav")

    buf_mp3 = BytesIO()
    buf_input.export(buf_mp3, format="mp3", bitrate=quality)
    buf_mp3.seek(0)

    degraded_audio = AudioSegment.from_file(buf_mp3, format="mp3")

    buf_wav = BytesIO()
    degraded_audio.export(buf_wav, format="wav")
    buf_wav.seek(0)

    return buf_wav

def degrade_quality(audio_tensor, sample_rate=22050, quality='16k'):
    tensors2d = audio_tensor.split(1, dim = 0)
    squeezed_tensors = []
    degraded_tensors = []
    for tensor in tensors2d:
        squeezed = tensor.squeeze(0)
        squeezed_tensors.append(squeezed)
    
    for tensor in squeezed_tensors:
        scaled_tensor = (tensor * MAX_WAV_VALUE).to(torch.int16)
        
        wav_buffer = BytesIO()
        wavefile = wave.open(wav_buffer, 'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(sample_rate)
        wavefile.writeframes(scaled_tensor.cpu().numpy().tobytes())  
        wavefile.close()
        
        degraded_audio_buffer = degrade_quality_bytes(wav_buffer, quality)
        
        degraded_audio, _ = torchaudio.load(degraded_audio_buffer, format="wav")
        degraded_tensors.append(degraded_audio)

    degraded_audio = torch.stack(degraded_tensors, dim=0)

    return degraded_audio


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    ###################################
    ## Changes in the original model ##
    ###################################
    

    global mel_basis, hann_window


    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    '''device = y.device

    y = degrade_quality(y).to(device)
    y = y.squeeze(1)'''


    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, win_length=win_size,
                                                    hop_length=hop_size, n_mels=num_mels, center=center, f_min=fmin, f_max=fmax,
                                                    window_fn=torch.hann_window, pad_mode='reflect', power=2.0, normalized=False).to(y.device)
    spec = transform(y)

    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            ## Changes in the original model ##
            lq_audio, _ = load_wav(filename.replace('wavs', 'lq_wavs'))
            lq_audio = lq_audio / MAX_WAV_VALUE
            lq_audio = normalize(lq_audio) * 0.95
            lq_audio = torch.FloatTensor(lq_audio)
            lq_audio = lq_audio.unsqueeze(0)
            if self.split:
                if lq_audio.size(1) >= self.segment_size:
                    lq_audio = lq_audio[:, audio_start:audio_start+self.segment_size]
                else:
                    lq_audio = torch.nn.functional.pad(lq_audio, (0, self.segment_size - lq_audio.size(1)), 'constant')

            mel = mel_spectrogram(lq_audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
