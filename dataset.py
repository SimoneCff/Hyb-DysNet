import re
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
import random

# --- Costanti per lo Spettrogramma ---
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SAMPLES = 80000 # 5 secondi * 16000 Hz
N_MELS = 128
N_FFT = 1024

MEL_SPECTROGRAM_TRANSFORM = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS
)
TO_DB_TRANSFORM = T.AmplitudeToDB()


class SandDataset(Dataset):
    
    def __init__(self, xlsx_file_path: str, sheet_name: str, audio_dir: str, 
                 is_training: bool = False, label_map_file: str = None):
        
        self.audio_dir = audio_dir
        self.target_sample_rate = TARGET_SAMPLE_RATE
        self.is_training = is_training
        
        self.id_extractor = re.compile(r"(ID\d+)")
        self.resampler_cache = {}
        
        if self.is_training:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=30)
            self.time_masking = T.TimeMasking(time_mask_param=100)
            
        self.task_names = [
            'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
            'rhythmKA', 'rhythmPA', 'rhythmTA'
        ]

        print(f"Costruzione label_map dal foglio 'SAND - TRAINING set - Task 1'...")
        try:
            label_file = label_map_file if label_map_file is not None else xlsx_file_path
            df_full = pd.read_excel(label_file, sheet_name="SAND - TRAINING set - Task 1")
            self.label_map = {row['ID']: row['Class'] for _, row in df_full.iterrows()}
            print(f"Label map creata con {len(self.label_map)} ID.")
        except Exception as e:
            print(f"Errore CRITICO: Impossibile leggere il foglio 'SAND - TRAINING set - Task 1' da {label_file}")
            raise e

        print(f"Costruzione subject list per il foglio: {sheet_name}...")
        try:
            df_split = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Errore CRITICO: Impossibile leggere il foglio '{sheet_name}'")
            raise e
        
        # Ogni elemento è una tupla: (subject_id, label_0_4, [lista_di_file_paths])
        self.subjects = []
        
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Scanning subjects for {sheet_name}"):
            subject_id = row['ID']
            label = self.label_map.get(subject_id)
            
            if label is None:
                print(f"Attenzione: ID {subject_id} dal foglio {sheet_name} non trovato nella label_map.")
                continue
                
            label_0_4 = label - 1
            
            subject_files = []
            for task in self.task_names:
                folder_name = f"{subject_id}_{task}"
                file_name = f"{folder_name}.wav"
                file_path = os.path.join(self.audio_dir, folder_name, file_name)
                subject_files.append(file_path)
            
            self.subjects.append((subject_id, label_0_4, subject_files))
        
        print(f"Dataset '{sheet_name}' creato con {len(self.subjects)} soggetti.")
        if len(self.subjects) == 0:
            print(f"ATTENZIONE: 0 soggetti trovati. Verifica che il percorso AUDIO_DIR '{self.audio_dir}' sia corretto.")

    def __len__(self):
        return len(self.subjects)

    def _get_resampler(self, orig_freq: int):
        if orig_freq not in self.resampler_cache:
            self.resampler_cache[orig_freq] = T.Resample(orig_freq=orig_freq, new_freq=self.target_sample_rate)
        return self.resampler_cache[orig_freq]

    def _load_audio_file(self, file_path: str):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != self.target_sample_rate:
                resampler = self._get_resampler(sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            current_len = waveform.shape[1]
            if current_len > TARGET_LEN_SAMPLES:
                waveform = waveform[:, :TARGET_LEN_SAMPLES]
            elif current_len < TARGET_LEN_SAMPLES:
                waveform = torch.nn.functional.pad(
                    waveform, (0, TARGET_LEN_SAMPLES - current_len)
                )

            mel_spec = MEL_SPECTROGRAM_TRANSFORM(waveform)
            mel_spec_db = TO_DB_TRANSFORM(mel_spec)
            
            mean = mel_spec_db.mean()
            std = mel_spec_db.std()
            mel_spec_normalized = (mel_spec_db - mean) / (std + 1e-6)

            if self.is_training:
                mel_spec_normalized = self.freq_masking(mel_spec_normalized)
                mel_spec_normalized = self.time_masking(mel_spec_normalized)
            
            mel_spec_3_channel = mel_spec_normalized.expand(3, -1, -1)
            
            return mel_spec_3_channel
        
        except Exception as e:
            # print(f"Errore nel caricare il file {file_path}: {e}")
            return torch.zeros((3, N_MELS, 157))
        
    def __getitem__(self, index: int):
        subject_id, label, file_paths = self.subjects[index]
        
        if not self.is_training:
            chosen_file = file_paths[0]
        else:
            chosen_file = file_paths[index % len(file_paths)]
        
        mel_spec = self._load_audio_file(chosen_file)
        
        return mel_spec, label

    def get_labels(self):
        return [label for (_, label, _) in self.subjects]