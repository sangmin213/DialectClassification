import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr=librosa.load('./DGDQ20000604.wav',offset=128,duration=1) # 128초부터 5초간 load
mfccs=librosa.feature.mfcc(y=y,sr=sr)
melspectro=librosa.feature.melspectrogram(y=y,sr=sr)

# wave from 저장
fig=plt.figure()
img=librosa.display.waveshow(y,sr)
fig.savefig('604(128sec+1)_12314.jpg')

# mel spectrum 저장
img=librosa.display.specshow(data=melspectro,sr=sr)
fig.savefig('604(128sec+1)_mel.jpg')

# MFCC 저장
img=librosa.display.specshow(data=mfccs,sr=sr)
fig.savefig('604(128sec+1)_mfcc.jpg')