import os

class Config:
    def __init__(self, base_dir, region, label_dir_name, data_dir_names, save_region_dir, samplerate, drop_start_sec, drop_end_sec, img_save, modes):
        self.base_dir = base_dir
        self.region = region
        self.region_dir = os.path.join(base_dir, region)
        self.data_dirs = data_dir_names
        self.label_dir_name = label_dir_name
        self.label_dir = os.path.join(self.region_dir, label_dir_name)
        self.data_dir_names = data_dir_names
        self.data_dirs = [os.path.join(self.region_dir, data_dir_name) for data_dir_name in data_dir_names]
        self.save_region_dir = save_region_dir
        self.samplerate = samplerate
        self.n_fft = int(samplerate/40)
        self.hop_length = int(samplerate/100)
        self.drop_start_sec = drop_start_sec
        self.drop_end_sec = drop_end_sec
        self.resize = int((drop_start_sec + drop_end_sec) / 2 * samplerate)
        self.img_save = img_save
        self.modes = modes

