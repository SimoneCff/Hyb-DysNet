import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import re

class SandDataset(Dataset):
    def __init__(self, file_list: list, label_map: dict, target_sample_rate):
        self.file_list = file_list
        self.label_map = label_map
        
        self.target_sample_rate = target_sample_rate
        self.id_extractor = re.compile(r"(ID\d+)")
     
    
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
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            waveform = waveform.squeeze(0)

            return waveform, label
          
        except Exception as e:
            print(f"Error for the file {file_path}: {e}")
            return torch.zeros(1, 128, 100), -1