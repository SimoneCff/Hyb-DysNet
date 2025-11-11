"""
dataset.py

PyTorch Dataset class for SAND Challenge Task 1.
Handles audio loading, feature extraction (Parselmouth, Wav2Vec),
and returns a unified feature vector per subject.
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import parselmouth
from parselmouth.praat import call
from torch.utils.data import Dataset

# --- 1. Constants and Global Models ---
# Load models once to be efficient.

print("Loading Wav2Vec2 model (requires internet on first run)...")

# Set device (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

try:
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    WAV2VEC_MODEL = bundle.get_model().to(DEVICE).eval()
except Exception as e:
    print(f"Critical Error: Could not load Wav2Vec2. {e}")
    print("Check internet connection and torchaudio installation.")
    WAV2VEC_MODEL = None

# Resampler: 8kHz (dataset) -> 16kHz (Wav2Vec2)
RESAMPLER_8_TO_16 = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

# Fixed vector size for classic features from each of the 8 audio files
CLASSIC_FEATURE_VECTOR_SIZE = 10 


# --- 2. Parselmouth Utility Functions ---

def get_phonation_features(sound):
    """Extracts Jitter, Shimmer, HNR, and Formants (F1, F2)."""
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", pitch.get_center_frequency())
        
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
        
        features = np.array([jitter, shimmer, hnr, f1_mean, f2_mean])
        return np.nan_to_num(features, nan=0.0)
    except Exception:
        return np.zeros(5)

def get_ddk_features(sound):
    """Extracts pitch and intensity regularity (std dev)."""
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_std_dev = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        intensity = call(sound, "To Intensity", 100, 0.0)
        intensity_std_dev = call(intensity, "Get standard deviation", 0, 0)
        
        features = np.array([pitch_std_dev, intensity_std_dev])
        return np.nan_to_num(features, nan=0.0)
    except Exception:
        return np.zeros(2)

def get_classic_features(audio_file_path):
    """
    Main classic feature dispatcher.
    Loads audio, calls the correct helper (phonation/ddk),
    and returns a fixed-size vector.
    """
    output_features = np.zeros(CLASSIC_FEATURE_VECTOR_SIZE)
    try:
        sound = parselmouth.Sound(audio_file_path)
    except Exception:
        return output_features 

    if "phonation" in audio_file_path:
        features = get_phonation_features(sound)
        output_features[0:len(features)] = features
    elif "diadochokinesis" in audio_file_path:
        features = get_ddk_features(sound)
        output_features[5:5+len(features)] = features 
        
    return output_features


# --- 3. Main Dataset Class (Corrected) ---

class SandFeatureExtractorDataset(Dataset):
    """
    PyTorch Dataset for SAND Challenge Task 1.
    
    Args:
        xlsx_file_path (str): Path to the .xlsx file (e.g., "sand_task_1.xlsx")
        sheet_name (str): Name of the sheet to load (e.g., "Validation Baseline - Task 1")
        audio_dir (str): Path to the root audio folder (e.g., ".../audio_files_task_1")
    """
    
    def __init__(self, xlsx_file_path, sheet_name, audio_dir):
        try:
            # --- THIS IS THE CORRECTION ---
            # Use pd.read_excel and pass the sheet_name
            self.df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
            # ------------------------------
        except FileNotFoundError:
            print(f"Error: Excel file not found at {xlsx_file_path}")
            raise
        except Exception as e:
            print(f"Error reading sheet '{sheet_name}' from {xlsx_file_path}: {e}")
            raise
            
        self.audio_dir = audio_dir
        
        # Folder/task names
        self.task_names = [
            'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
            'diadochokinesisPA', 'diadochokinesisTA', 'diadochokinesisKA'
        ]

    def __len__(self):
        """Returns the number of subjects in the DataFrame."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets one subject's complete feature vector and label.
        """
        
        # 1. Get tabular data
        subject_row = self.df.iloc[idx]
        subject_id = subject_row['ID']
        age = subject_row['Age']
        sex = 1 if subject_row['Sex'] == 'M' else 0
        label = subject_row['Class']
        
        all_waveforms_16k = []
        classic_features_list = []
        
        # 2. Process all 8 audio tasks for the subject
        for task in self.task_names:
            
            folder_name = f"{subject_id}_{task}"
            file_name = f"{folder_name}.wav"
            file_path = os.path.join(self.audio_dir, folder_name, file_name)
            
            try:
                # Load, resample, and store for Wav2Vec
                waveform_8k, sr = torchaudio.load(file_path)
                waveform_16k = RESAMPLER_8_TO_16(waveform_8k)
                all_waveforms_16k.append(waveform_16k)
                
                # Get classic features
                task_features = get_classic_features(file_path) 
                classic_features_list.append(task_features)
                
            except FileNotFoundError:
                all_waveforms_16k.append(torch.zeros((1, 16000)))
                classic_features_list.append(np.zeros(CLASSIC_FEATURE_VECTOR_SIZE))
            except Exception:
                all_waveforms_16k.append(torch.zeros((1, 16000))) 
                classic_features_list.append(np.zeros(CLASSIC_FEATURE_VECTOR_SIZE)) 

        # --- 3. Process Deep Features (Wav2Vec) ---
        full_waveform_16k = torch.cat(all_waveforms_16k, dim=1).to(DEVICE)
        
        with torch.no_grad():
            features, _ = WAV2VEC_MODEL(full_waveform_16k)
        
        deep_features_vector = torch.mean(features, dim=1).squeeze(0)
        
        # --- 4. Process Classic Features (Parselmouth) ---
        classic_features_vector = np.concatenate(classic_features_list) # [80]
        
        # --- 5. Combine All Features ---
        age_sex_vector = np.array([age, sex]) # [2]
        deep_features_np = deep_features_vector.cpu().numpy() # [768]
        
        final_feature_vector = np.concatenate([
            deep_features_np,          # [768]
            classic_features_vector,   # [80]
            age_sex_vector             # [2]
        ]) # Total size: 850
        
        # Map labels 1-5 to 0-4
        final_label = label - 1 
        
        return final_feature_vector, final_label

# --- 6. Self-Test Block (Updated) ---
# This runs only if you execute: python dataset.py
if __name__ == "__main__":
    
    print("\n--- Running dataset.py self-test ---")
    
    # --- !!! UPDATE THESE PATHS !!! ---
    # Path to the single .xlsx file
    TEST_XLSX_PATH = "task1/sand_task_1.xlsx"
    # Name of the sheet to test (use the small validation one)
    TEST_SHEET_NAME = "Validation Baseline - Task 1"
    # Path to the main audio folder
    TEST_AUDIO_DIR = "task1/training"
    # ----------------------------------
    
    print(f"Test XLSX: {TEST_XLSX_PATH}")
    print(f"Test Sheet: {TEST_SHEET_NAME}")
    print(f"Test Audio Dir: {TEST_AUDIO_DIR}")

    if "percorso/del/tuo" in TEST_XLSX_PATH:
        print("\nWARNING: Update TEST_XLSX_PATH, TEST_SHEET_NAME, and TEST_AUDIO_DIR to run the self-test.")
    else:
        try:
            if WAV2VEC_MODEL is None:
                raise Exception("Wav2Vec2 model did not load. Aborting test.")
                
            test_dataset = SandFeatureExtractorDataset(
                xlsx_file_path=TEST_XLSX_PATH,
                sheet_name=TEST_SHEET_NAME,
                audio_dir=TEST_AUDIO_DIR
            )
            print(f"Dataset loaded. Subject count: {len(test_dataset)}")
            
            # Test fetching the first subject
            features, label = test_dataset[0]
            
            print(f"\nSuccessfully fetched subject 0 (ID: {test_dataset.df.iloc[0]['ID']}).")
            print(f"  Feature vector shape: {features.shape}") # Should be (850,)
            print(f"  Label (mapped 0-4): {label}")
            print("\nSelf-test PASSED!")
            
        except FileNotFoundError:
            print("\n--- TEST FAILED ---")
            print("File not found. Check your paths.")
        except Exception as e:
            print(f"\n--- TEST FAILED ---")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()