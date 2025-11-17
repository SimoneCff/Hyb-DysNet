import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import Counter
# Import FSL
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler 
from easyfsl.modules import resnet18

try:
    from dataset import SandDataset
except ImportError:
    print("Errore: Assicurati che 'dataset.py' (Versione Spettrogrammi) sia nella stessa cartella.")
    exit(1)
    
class DropoutBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.dropout(features)


def main():
    DEVICE = torch.device("mps")
    
    # Percorsi
    TRAIN_XLSX = "task1/sand_task_1.xlsx"
    AUDIO_DIR = "task1/training"
    
    N_WAY = 5
    N_SHOT = 3 
    N_SHOT_TRAIN = 3
    N_QUERY_TRAIN = 1
    
    N_SHOT_VAL = 1 
    N_QUERY_VAL = 1
    
    N_EPOCHS = 100
    N_TASKS_PER_EPOCH = 300
    
    
    
    # --- 1. Inizializza i Dataset ---
    print("Inizializzazione Datasets (auto-sufficiente)...")
    
    train_dataset = SandDataset(
        xlsx_file_path=TRAIN_XLSX,
        sheet_name="Training Baseline - Task 1",
        audio_dir=AUDIO_DIR,
        is_training=True
    )
    
    # ✅ USA VALIDATION per monitorare durante training
    val_dataset = SandDataset(
        xlsx_file_path=TRAIN_XLSX,
        sheet_name="Validation Baseline - Task 1",  # ✅ Validation ha le label
        audio_dir=AUDIO_DIR,
        is_training=False
    )

    # --- ANALISI DATASET ---
    print("\n=== ANALISI DISTRIBUZIONE CLASSI ===")
    train_labels = train_dataset.get_labels()
    val_labels = val_dataset.get_labels()  # ✅ Cambiato da test_labels
    
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)  # ✅ Cambiato da test_dist
    
    print(f"\nTraining Set - Totale soggetti: {len(train_labels)}")
    for cls in range(5):
        print(f"  Classe {cls+1}: {train_dist.get(cls, 0)} soggetti")
    
    print(f"\nValidation Set - Totale soggetti: {len(val_labels)}")  # ✅ Cambiato
    for cls in range(5):
        print(f"  Classe {cls+1}: {val_dist.get(cls, 0)} soggetti")
    
    min_train_samples = min(train_dist.values())
    min_val_samples = min(val_dist.values())
    required_train_samples = N_SHOT_TRAIN + N_QUERY_TRAIN
    required_val_samples = N_SHOT_VAL + N_QUERY_VAL
    
    if min_train_samples < required_train_samples:
        print(f"\n⚠️  ATTENZIONE: Training - La classe con meno campioni ha {min_train_samples} soggetti,")
        print(f"   ma servono almeno {required_train_samples} (N_SHOT={N_SHOT_TRAIN} + N_QUERY={N_QUERY_TRAIN})!")
        return
    
    if min_val_samples < required_val_samples:
        print(f"\n⚠️  ATTENZIONE: Validation - La classe con meno campioni ha {min_val_samples} soggetti,")
        print(f"   ma servono almeno {required_val_samples} (N_SHOT={N_SHOT_VAL} + N_QUERY={N_QUERY_VAL})!")
        return
    
    print(f"\n✓ Configurazione valida: Train needs {required_train_samples}, Val needs {required_val_samples}\n")

    print("Inizializzazione Sampler FSL...")
    train_sampler = TaskSampler(
        train_dataset, 
        n_way=N_WAY, 
        n_shot=N_SHOT_TRAIN,  # ✅ Usa parametri training
        n_query=N_QUERY_TRAIN,  # ✅ Usa parametri training
        n_tasks=N_TASKS_PER_EPOCH
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        collate_fn=train_sampler.episodic_collate_fn,
        pin_memory=True
    )

    val_sampler = TaskSampler(
        val_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT_VAL,  # ✅ Usa parametri validation (più piccoli)
        n_query=N_QUERY_VAL,  # ✅ Usa parametri validation (più piccoli)
        n_tasks=50
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=4,
        collate_fn=val_sampler.episodic_collate_fn,
        pin_memory=True
    )
    # Verifica soggetti duplicati
    train_subjects = set([s[0] for s in train_dataset.subjects])
    val_subjects = set([s[0] for s in val_dataset.subjects])  # ✅ Cambiato

    overlap = train_subjects.intersection(val_subjects)
    if overlap:
        print(f"⚠️ ATTENZIONE: {len(overlap)} soggetti presenti sia in train che in validation!")
        print(f"Soggetti duplicati: {list(overlap)[:10]}...")
        val_dataset.subjects = [s for s in val_dataset.subjects if s[0] not in train_subjects]  # ✅ Cambiato
        print(f"✓ Rimossi i duplicati. Nuovo size validation: {len(val_dataset.subjects)}")

    # ... resto del codice rimane uguale ...
    backbone = resnet18()
    backbone_with_dropout = DropoutBackbone(backbone)
    model = PrototypicalNetworks(backbone=backbone_with_dropout).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Inizio Training FSL ({N_WAY}-way {N_SHOT}-shot) ---")
    best_val_acc = 0
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}") as pbar:
            for batch in pbar:
                support_images, support_labels, query_images, query_labels, _class_ids = batch

                support_images = support_images.to(DEVICE)
                support_labels = support_labels.to(DEVICE)
                query_images = query_images.to(DEVICE)
                query_labels = query_labels.to(DEVICE)
                
                model.process_support_set(support_images, support_labels)
                query_logits = model(query_images)
                loss = criterion(query_logits, query_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calcola accuracy durante il training
                _, predictions = torch.max(query_logits, 1)
                correct += (predictions == query_labels).sum().item()
                total += query_labels.size(0)
                
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{total_loss / (pbar.n + 1):.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
        
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Test Evaluation Epoch {epoch+1}") as pbar_val:
                for batch in pbar_val:
                    support_images, support_labels, query_images, query_labels, _class_ids = batch

                    support_images = support_images.to(DEVICE)
                    support_labels = support_labels.to(DEVICE)
                    query_images = query_images.to(DEVICE)
                    query_labels = query_labels.to(DEVICE)
                    
                    model.process_support_set(support_images, support_labels)
                    query_logits = model(query_images)

                    _, predictions = torch.max(query_logits, 1)
                    val_correct += (predictions == query_labels).sum().item()
                    val_total += query_labels.size(0)
                    
                    pbar_val.set_postfix(acc=f"{100 * val_correct / val_total:.2f}%")
        
        avg_val_acc = val_correct / val_total
        print(f"\nEpoch {epoch+1}: Train Acc = {100*correct/total:.2f}%, Test Acc = {100*avg_val_acc:.2f}%")
        
        if avg_val_acc > best_val_acc:
            print(f"✓ Nuova best accuracy: {100*avg_val_acc:.2f}% (salvataggio modello)")
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, "fsl_resnet18_backbone_best.pth")

    for i, batch in enumerate(train_loader):
        support_images, support_labels, query_images, query_labels, class_ids = batch
        
        print(f"\n=== Batch {i} ===")
        print(f"Support labels: {support_labels.cpu().numpy()}")
        print(f"Query labels: {query_labels.cpu().numpy()}")
        print(f"Class IDs: {class_ids}")
        print(f"Support shape: {support_images.shape}")
        print(f"Query shape: {query_images.shape}")
        
        # Verifica che non ci siano label duplicate nel query set
        unique_query = torch.unique(query_labels)
        if len(unique_query) < N_WAY:
            print(f"⚠️ PROBLEMA: Solo {len(unique_query)} classi uniche nel query set!")
        
        if i >= 2:  # Mostra solo i primi 3 batch
            break
    
    print(f"\n🎉 Training completato. Migliore accuratezza sul test set: {100*best_val_acc:.2f}%")

if __name__ == "__main__":
    main()