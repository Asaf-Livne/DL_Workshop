from meldataset import degrade_quality
import os
import sys
import torchaudio


dir_list = os.listdir('LJSpeech-1.1/wavs')
i = 0
for file in dir_list:
    if file.endswith('.wav'):
        audio, _ = torchaudio.load('LJSpeech-1.1/wavs/' + file)
        audio = audio.unsqueeze(0)
        degraded = degrade_quality(audio, 22050, '24k')
        degraded = degraded.squeeze(0)
        torchaudio.save('LJSpeech-1.1/lq_wavs/' + file, degraded, 22050, format='wav', bits_per_sample=16)
        i += 1
        if i % 1000 == 0:
            print(i)

print('Done')