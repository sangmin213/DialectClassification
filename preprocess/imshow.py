import librosa
from matplotlib import pyplot  as plt
import librosa.display
def main():
    data_path = '../dataset/jeju/jeju_data_1/DZES20000002.wav'

    y, sr = librosa.load(data_path, offset=5, duration=3)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    fig = plt.figure()
    print("y shape : ",y.shape)
    print("chroma shape :",chroma.shape)
    print("sr : ",sr)

    librosa.display.specshow(chroma, sr=sr)
    fig.savefig('./chroma')



if __name__ == '__main__':
    main()