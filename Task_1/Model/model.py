import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines

class SANDClassifier (nn.Module):
    def __init__(self, num_classes=5, freeze_encoder=True):
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        in_features = 768 #Base features for WA2Vec2
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(waveforms)
        
        pooled_features = torch.mean(features,dim=1)
        
        logits = self.head(pooled_features)
        
        return logits
        
