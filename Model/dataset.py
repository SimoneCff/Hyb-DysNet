# In dataset.py

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import re

class SandDataset(Dataset):
    def __init__(self, file_list: list, label_map: dict, target_sample_rate: int, is_training: bool = False):
        self.file_list = file_list
        self.label_map = label_map
        self.target_sample_rate = target_sample_rate
        self.is_training = is_training
        self.id_extractor = re.compile(r"(ID\d+)")
        
        # --- FIX 1: DEFINE MAX_LENGTH HERE ---
        self.max_length = target_sample_rate * 5  # 5 seconds
        # ------------------------------------
        
        if self.is_training:
            self.time_masking = T.TimeMasking(time_mask_param=80)

    
    def __len__(self):
         return len(self.file_list)
        
    def __getitem__(self, index):
        file_path = self.file_list[index]
        
        match = self.id_extractor.search(file_path.name)
        if not match:
            raise ValueError(f"Error extracting the id of {file_path.name}")
        id_element = match.group(1)

        label = self.label_map[id_element] - 1
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # --- FIX 2: TRUNCATE/PAD (This code is now correct) ---
            num_samples = waveform.shape[1]
            if num_samples > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif num_samples < self.max_length:
                pad_len = self.max_length - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            # ------------------------------------------------

            if self.is_training:
                noise_factor = 0.005
                noise = torch.randn_like(waveform) * noise_factor
                waveform = waveform + noise
                waveform = self.time_masking(waveform)
                
            waveform = waveform.squeeze(0)

            return waveform, label
        
        except Exception as e:
            print(f"Error for the file {file_path}: {e}")
            return torch.empty(0), -1 # This is correct