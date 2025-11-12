"""
dataset.py

(Versione Spettrogrammi, Auto-Sufficiente, Corretta)

Contiene la classe SandDataset per il training FSL.
Questa classe gestisce internamente la lettura dell'Excel per 
costruire la sua lista di file e la mappa delle etichette.
*Include get_labels() per easyfsl e corregge la logica di path-finding.*
"""

import re
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

# --- Costanti per lo Spettrogramma ---
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SAMPLES = 80000 # 5 secondi * 16000 Hz
N_MELS = 128 # Altezza dell'immagine (128 bande Mel)
N_FFT = 1024 # Dimensione della finestra FFT

# Trasformazioni globali
MEL_SPECTROGRAM_TRANSFORM = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS
)
TO_DB_TRANSFORM = T.AmplitudeToDB()


class SandDataset(Dataset):
    """
    Dataset auto-sufficiente. Legge l'.xlsx per costruire
    la file_list e la label_map.
    
    Ritorna uno Spettrogramma Mel (come immagine 3 canali) e l'etichetta.
    
    Args:
        xlsx_file_path (str): Percorso al file sand_task_1.xlsx
        sheet_name (str): Il foglio specifico da caricare (es. "Training Baseline - Task 1")
        audio_dir (str): Percorso alla cartella audio (es. "task1/training")
        is_training (bool): Se True, applica l'augmentation (SpecAugment).
    """
    
    def __init__(self, xlsx_file_path: str, sheet_name: str, audio_dir: str, is_training: bool = False):
        
        self.audio_dir = audio_dir
        self.target_sample_rate = TARGET_SAMPLE_RATE
        self.is_training = is_training
        
        self.id_extractor = re.compile(r"(ID\d+)")
        self.resampler_cache = {}
        
        if self.is_training:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=30)
            self.time_masking = T.TimeMasking(time_mask_param=100)
            
        task_names = [
            'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
            'diadochokinesisPA', 'diadochokinesisTA', 'diadochokinesisKA'
        ]

        # --- 1. Costruisci la Mappa delle Etichette ---
        print(f"Costruzione label_map dal foglio 'SAND - TRAINING set - Task 1'...")
        try:
            df_full = pd.read_excel(xlsx_file_path, sheet_name="SAND - TRAINING set - Task 1")
            self.label_map = {row['ID']: row['Class'] for _, row in df_full.iterrows()}
            print(f"Label map creata con {len(self.label_map)} ID.")
        except Exception as e:
            print(f"Errore CRITICO: Impossibile leggere il foglio 'SAND - TRAINING set - Task 1' da {xlsx_file_path}")
            raise e

        # --- 2. Costruisci la Lista dei File per questo split ---
        print(f"Costruzione file_list per il foglio: {sheet_name}...")
        try:
            df_split = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Errore CRITICO: Impossibile leggere il foglio '{sheet_name}'")
            raise e
            
        self.data = [] # Lista di tuple: (percorso_file, etichetta_0_4)
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Scanning files for {sheet_name}"):
            subject_id = row['ID']
            label = self.label_map.get(subject_id) 
            
            if label is None:
                print(f"Attenzione: ID {subject_id} dal foglio {sheet_name} non trovato nella label_map.")
                continue
                
            label_0_4 = label - 1 # Mappa a 0-4
            
            for task in task_names:
                folder_name = f"{subject_id}_{task}"
                file_name = f"{folder_name}.wav"
                file_path = os.path.join(self.audio_dir, folder_name, file_name)
                
                # --- CORREZIONE: Rimuovi 'if os.path.exists(file_path):' ---
                # Aggiungiamo il percorso assumendo che esista.
                # Sarà __getitem__ a gestire l'errore se il file non esiste.
                self.data.append( (file_path, label_0_4) )
        
        print(f"Dataset '{sheet_name}' creato con {len(self.data)} campioni audio.")
        if len(self.data) == 0:
            print(f"ATTENZIONE: 0 campioni trovati. Verifica che il percorso AUDIO_DIR '{self.audio_dir}' sia corretto.")

    def __len__(self):
         return len(self.data)

    def _get_resampler(self, orig_freq: int):
        if orig_freq not in self.resampler_cache:
            self.resampler_cache[orig_freq] = T.Resample(orig_freq=orig_freq, new_freq=self.target_sample_rate)
        return self.resampler_cache[orig_freq]

    def __getitem__(self, index: int):
        
        file_path, label = self.data[index]

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
            
            return mel_spec_3_channel, label
        
        except Exception as e:
            # print(f"Errore nel caricare il file {file_path}: {e}")
            return torch.zeros((3, N_MELS, 157)), -1 # Dimensione approssimativa

    # --- CORREZIONE: Aggiungi questa funzione ---
    def get_labels(self):
        """
        Richiesto da easyfsl.TaskSampler.
        Ritorna una lista di tutte le etichette (0-4) nel dataset.
        """
        # self.data è una lista di tuple: (file_path, label_0_4)
        # Ritorna l'etichetta (indice 1) per ogni elemento.
        return [label for (file_path, label) in self.data]