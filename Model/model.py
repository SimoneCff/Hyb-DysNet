import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines

class SANDClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_encoder=True):
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(waveforms)
        
        pooled_features = torch.mean(features,dim=1)

        return pooled_features
        

def crea_modello(num_classes=5, freeze=False):
    print(f"--- Model Creation (FSL Encoder Wav2Vec2) ---")
    print(f"Freeze Encoder: {freeze}")   
    model = SANDClassifier(num_classes=num_classes, freeze_encoder=freeze)
    return model