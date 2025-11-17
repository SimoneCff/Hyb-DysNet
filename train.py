"""
train.py

Addestra un modello Few-Shot (Prototypical Networks)
utilizzando un backbone ResNet18 su Spettrogrammi Mel.

MODIFICHE per dataset sbilanciato:
- Ridotto N_SHOT e N_QUERY per adattarsi alla classe con meno campioni
- Aggiunto Class Balancing per il sampler
"""

import torch
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

# --- 1. Funzione Main ---
def main():
    
    # --- Configurazione ---
    DEVICE = torch.device("mps")
    
    # Percorsi
    TRAIN_XLSX = "task1/sand_task_1.xlsx"
    TEST_XLSX = "task1/sand_task_1_test.xlsx"  # Nome corretto del file test
    AUDIO_DIR = "task1/training"
    AUDIO_TEST_DIR = "task1/test"
    
    # Parametri FSL - RIDOTTI per dataset sbilanciato
    N_WAY = 5
    N_SHOT = 3  # Ridotto da 5 a 3 (Classe 1 ha solo 6 soggetti)
    N_QUERY = 2  # Ridotto da 10 a 2 per evitare sovrapposizioni
    N_EPOCHS = 30  # Aumentato per compensare meno samples per task
    N_TASKS_PER_EPOCH = 100
    
    # --- 1. Inizializza i Dataset ---
    print("Inizializzazione Datasets (auto-sufficiente)...")
    
    train_dataset = SandDataset(
        xlsx_file_path=TRAIN_XLSX,
        sheet_name="Training Baseline - Task 1",
        audio_dir=AUDIO_DIR,
        is_training=True
    )
    
    # Usa il Test set per la validazione (file Excel separato)
    # IMPORTANTE: Per il test set, passiamo il file di training per la label_map
    test_dataset = SandDataset(
        xlsx_file_path=TEST_XLSX,
        sheet_name="Test Baseline - Task 1",
        audio_dir=AUDIO_TEST_DIR,
        is_training=False,
        label_map_file=TRAIN_XLSX  # Usa il file di training per le label
    )

    # --- ANALISI DATASET ---
    print("\n=== ANALISI DISTRIBUZIONE CLASSI ===")
    train_labels = train_dataset.get_labels()
    test_labels = test_dataset.get_labels()
    
    train_dist = Counter(train_labels)
    test_dist = Counter(test_labels)
    
    print(f"\nTraining Set - Totale soggetti: {len(train_labels)}")
    for cls in range(5):
        print(f"  Classe {cls+1}: {train_dist.get(cls, 0)} soggetti")
    
    print(f"\nTest Set - Totale soggetti: {len(test_labels)}")
    for cls in range(5):
        print(f"  Classe {cls+1}: {test_dist.get(cls, 0)} soggetti")
    
    # Verifica che ogni classe abbia abbastanza soggetti
    min_samples = min(train_dist.values())
    required_samples = N_SHOT + N_QUERY
    
    if min_samples < required_samples:
        print(f"\n⚠️  ATTENZIONE: La classe con meno campioni ha {min_samples} soggetti,")
        print(f"   ma servono almeno {required_samples} (N_SHOT={N_SHOT} + N_QUERY={N_QUERY})!")
        print(f"   Riduci N_SHOT o N_QUERY, oppure rimuovi la classe sbilanciata.")
        return
    
    print(f"\n✓ Configurazione valida: ogni classe ha almeno {required_samples} soggetti\n")

    # --- 2. Inizializza i Sampler e Loader FSL ---
    print("Inizializzazione Sampler FSL...")
    train_sampler = TaskSampler(
        train_dataset, 
        n_way=N_WAY, 
        n_shot=N_SHOT, 
        n_query=N_QUERY, 
        n_tasks=N_TASKS_PER_EPOCH
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,  # Ridotto per stabilità
        collate_fn=train_sampler.episodic_collate_fn,
        pin_memory=True
    )

    val_sampler = TaskSampler(
        test_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=50  # Meno task per validation (più veloce)
    )
    val_loader = DataLoader(
        test_dataset,
        batch_sampler=val_sampler,
        num_workers=4,
        collate_fn=val_sampler.episodic_collate_fn,
        pin_memory=True
    )

    # --- 3. Inizializza Modello e Optimizer ---
    print("Caricamento backbone ResNet18...")
    backbone = resnet18()
    model = PrototypicalNetworks(backbone=backbone).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    criterion = nn.CrossEntropyLoss()

    # --- 4. Loop di Training ---
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

        # --- 5. Loop di Validazione su Test Set ---
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

    print(f"\n🎉 Training completato. Migliore accuratezza sul test set: {100*best_val_acc:.2f}%")

if __name__ == "__main__":
    main()