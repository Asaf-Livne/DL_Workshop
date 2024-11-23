import matplotlib.pyplot as plt
import torchaudio
import torch
import os
import numpy as np
import json


hq, srhq = torchaudio.load('demo/hq.wav')
lq, srlq = torchaudio.load('demo/lq.wav')
gen, srg = torchaudio.load('demo/gen_newer.wav')

h = json.load(open('config_v1.json'))
transform = torchaudio.transforms.MelSpectrogram(sample_rate=h['sampling_rate'], n_fft=h['n_fft'], win_length=h['win_size'], hop_length=h['hop_size'], n_mels=h['num_mels'], center=False, f_min=h['fmin'], f_max=h['fmax'], window_fn=torch.hann_window, pad_mode='reflect', power=2.0, normalized=True)

lq_spec = transform(lq)
hq_spec = transform(hq)
gen_spec = transform(gen)

plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.imshow(np.log(lq_spec[0].numpy()), aspect='auto', origin='lower')
plt.title('24kbps mp3 Compressed Audio')
plt.subplot(3, 1, 2)
plt.imshow(np.log(gen_spec[0].numpy()), aspect='auto', origin='lower')
plt.title('Generated from Compressed Audio')
plt.subplot(3, 1, 3)
plt.imshow(np.log(hq_spec[0].numpy()), aspect='auto', origin='lower')
plt.title('Original Audio')
plt.show()