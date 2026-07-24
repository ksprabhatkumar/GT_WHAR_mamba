import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataset.data_loader import get_loso_dataloaders, get_dataset_config
from models.mamba_whar import Mamba_WHAR
from utils.metrics import calculate_metrics
from utils.logger import log_results, log_summary

# ==========================================
# ⚙️ MAMBA HYPERPARAMETERS
# ==========================================
HYPERPARAMS = {
    "epochs": 100,             # Increased to 100 for the Accuracy vs Epoch loop
    "batch_size": 64,          # Optimal batch size
    "lr": 0.001,               # Mamba learning rate
    "hidden_size": 64
}

CHECKPOINT_PATH = "checkpoints/latest_mamba_checkpoint.pth"

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc, macro_f1 = calculate_metrics(all_labels, all_preds)
    return acc, macro_f1

def run_experiment(dataset_name, device, resume_state=None):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING MAMBA EXPERIMENT: {dataset_name.upper()} ({HYPERPARAMS['epochs']} Epochs)")
    print(f"{'='*50}")
    
    config = get_dataset_config(dataset_name)
    start_subject = 1
    loso_acc, loso_f1 = [], []
    
    # 🌟 NEW: Dictionary to store Accuracy per epoch for plotting
    learning_curve_history = {} 

    if resume_state and resume_state['dataset'] == dataset_name:
        start_subject = resume_state['subject_id']
        start_epoch = resume_state['epoch'] + 1
        loso_acc = resume_state['loso_acc']
        loso_f1 = resume_state['loso_f1']
        best_acc, best_f1 = resume_state['best_acc'], resume_state['best_f1']
        
        # Load previous learning curve history if it exists
        if os.path.exists("mamba_learning_curve.json"):
            with open("mamba_learning_curve.json", "r") as f:
                learning_curve_history = json.load(f)

        if start_epoch > HYPERPARAMS["epochs"]:
            loso_acc.append(best_acc)
            loso_f1.append(best_f1)
            log_results(dataset_name, start_subject, best_acc * 100, best_f1 * 100, HYPERPARAMS)
            start_subject += 1
            start_epoch = 1
            best_acc, best_f1 = 0.0, 0.0
            
        print(f"[!] Resuming {dataset_name.upper()} from Subject {start_subject}, Epoch {start_epoch}")
    else:
        resume_state = None

    for subject_id in range(start_subject, config['total_subjects'] + 1):
        print(f"\n--- FOLD {subject_id}/{config['total_subjects']}: Subject {subject_id} ---")
        train_loader, test_loader = get_loso_dataloaders(dataset_name, subject_id, batch_size=HYPERPARAMS["batch_size"])
        
        model = Mamba_WHAR(
            num_nodes=config['num_nodes'], 
            in_features=config['in_features'], 
            hidden_dim=HYPERPARAMS["hidden_size"],
            num_classes=config['num_classes']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
        
        subject_key = f"Subject_{subject_id}"
        if subject_key not in learning_curve_history:
            learning_curve_history[subject_key] = []
            
        if resume_state and subject_id == start_subject and start_epoch > 1:
            model.load_state_dict(resume_state['model_state_dict'])
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        else:
            start_epoch = 1
            best_acc, best_f1 = 0.0, 0.0
            
        for epoch in range(start_epoch, HYPERPARAMS["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            # 🌟 NEW: Track Accuracy for the graph!
            learning_curve_history[subject_key].append(test_acc * 100)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%")
            
            # Save the learning curve data dynamically
            with open("mamba_learning_curve.json", "w") as f:
                json.dump(learning_curve_history, f, indent=4)
            
            torch.save({
                'dataset': dataset_name,
                'subject_id': subject_id,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loso_acc': loso_acc,
                'loso_f1': loso_f1,
                'best_acc': best_acc,
                'best_f1': best_f1
            }, CHECKPOINT_PATH)
            
        print(f"✅ Best for Subject {subject_id}: Acc = {best_acc*100:.2f}%, F1 = {best_f1*100:.2f}%")
        
        log_results(dataset_name, subject_id, best_acc * 100, best_f1 * 100, HYPERPARAMS)
        loso_acc.append(best_acc)
        loso_f1.append(best_f1)
        resume_state = None 

    if len(loso_acc) > 0:
        avg_acc = np.mean(loso_acc) * 100
        avg_f1 = np.mean(loso_f1) * 100
        log_summary(dataset_name, avg_acc, avg_f1)
        print(f"\n🎉 FINISHED {dataset_name.upper()} | Avg Acc: {avg_acc:.2f}% | Avg F1: {avg_f1:.2f}%")
        
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba-WHAR Experiment Runner")
    parser.add_argument("--dataset", type=str, default="dsads", help="Choose dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on DEVICE: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    
    global_resume_state = None
    if os.path.exists(CHECKPOINT_PATH):
        global_resume_state = torch.load(CHECKPOINT_PATH)
        print(f"\n[!] Checkpoint Found! Left off at Dataset: {global_resume_state['dataset'].upper()}")

    run_experiment(args.dataset.lower(), device, resume_state=global_resume_state)