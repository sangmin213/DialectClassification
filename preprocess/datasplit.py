import json
import os
import pickle
import wave
import torch
import librosa
# tree of dataset of one region
# dataset
# ├── label_dir
# │   ├── 001.json
# │   ├── 001.txt
# │   └── 002.json
# └── wav_dir
#     ├── 001.wav
#     └── 002.wav

# tree of preprocessed dataset
# preprocessed_dir
# ├── chungcheong
# │   └── preprocessed_001
# │       ├── 001.json
# │       ├── 1.wav
# │       ├── 2.wav
# │       └── 3.wav
# ├── gangwon
# ├── gyungsang
# ├── jeju
# └── junla
class DataSpliter:

    # label_file_path : the path of the directory include label json files
    # wav_file_path : the path of the directory include wav files
    # save_region_dir : the path of the directory you want to save label file and chunked wav files.
    def __init__(self, label_file_path, wav_file_path, save_region_dir):
        self.label_file_path = label_file_path
        self.wav_file_path = wav_file_path
        self.save_region_dir = save_region_dir
        
        json_file_names = os.listdir(label_file_path)
        json_file_names = [j for j in json_file_names if j.endswith('.json')]

        self.json_file_names = json_file_names

        wav_file_names = os.listdir(wav_file_path)
        wav_file_names = [w for w in wav_file_names if w.endswith('.wav')]

        self.wav_file_names = wav_file_names

        wav_label_dict = dict()

        for json_file_name in json_file_names:
            matched_wav_file = list(filter(lambda x: x[:-3] == json_file_name[:-4], wav_file_names))

            if len(matched_wav_file) == 0: continue
            
            wav_label_dict[json_file_name] = matched_wav_file[0]
        
        self.wav_label_dict = wav_label_dict
        

    def loadJson(self, label_file_path):
        with open(label_file_path, "r") as f:
            j = json.load(f)
        return j
    
    def readWavFile(self, wave_file_path):
        f = wave.open(wave_file_path, "rb")
        return f

    def getParams(self, wave_fd):
        params = wave_fd.getparams() #[nchannels, sampwidth, framerate, nframes, comptype, compname]
        return params

    def saveWavFile(self, data, params, start, end, save_file_name):
        with wave.open(save_file_name, "w") as f:
            f.setparams(params)
            f.setnframes(int(len(data) / params[1]))
            f.writeframes(data)
    
    def splitWavFiles(self, input_file_path, save_dir, labels):
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        wav_fd = self.readWavFile(input_file_path)
        params = self.getParams(wav_fd)

        for i, label in enumerate(labels, 1):
            start, end, speaker_info = label
            wav_fd.setpos(int(start*params[2])) # params[2] == framerate
            data = wav_fd.readframes(int((end-start) * params[2]))

            save_file_path = os.path.join(save_dir, f"{i}.wav")
            self.saveWavFile(data, params, start, end, save_file_path)

        wav_fd.close()

    def preprocessingLabel(self, label_file_path):
        json_data = self.loadJson(label_file_path)

        speakers = json_data['speaker']
        speaker_ids = [speaker['id'] for speaker in speakers]
        speakers_dict = {
            speaker_id:{'sex':speakers[i]['sex'], 'age':speakers[i]['age'], 'residence':speakers[i]['principal_residence']} 
            for i, speaker_id in enumerate(speaker_ids)
        }

        dialog = json_data['utterance']

        time_slices_speaker_infos = [(d['start'], d['end'], speakers_dict[d['speaker_id']]) for d in dialog]
        
        return time_slices_speaker_infos

    def saveLabels(self, file, label_file_path):
        with open(label_file_path, "wb") as f:
            pickle.dump(file, f)
        
    def run(self):

        wav_label_dict = self.wav_label_dict
        
        save_base_dir = self.save_region_dir
        if not os.path.isdir(save_base_dir): os.mkdir(save_base_dir)


        for i, (label_file, wav_file) in enumerate(wav_label_dict.items(), 1):
            label_path = os.path.join(self.label_file_path, label_file)
            time_speaker_info = self.preprocessingLabel(label_path)

            save_dir = os.path.join(save_base_dir, f"preprocessed_{label_file[:-5]}")
            if not os.path.isdir(save_dir): os.mkdir(save_dir)

            # save preprocessed label as pickle
            save_label_path = os.path.join(save_dir, f"preprocessed_{label_file[:-5]}.pickle")
            self.saveLabels(time_speaker_info, save_label_path)

            wav_path = os.path.join(self.wav_file_path, wav_file)
            self.splitWavFiles(wav_path, save_dir, time_speaker_info)

            print(f"\r{i}/{len(wav_label_dict)}", end="")

        print()


def main():
    label_file_path = '../dataset/gangwon/gangwon_label'
    wav_file_path = '../dataset/gangwon/gangwon_data_1'
    save_region_dir = './gangwon_preprocessed'
    pp = DataSpliter(label_file_path, wav_file_path, save_region_dir)

    pp.run()

if __name__ == '__main__':
    main()