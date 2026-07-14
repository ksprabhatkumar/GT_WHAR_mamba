import os
import torch
from torch.utils.data import Dataset, DataLoader

class HARDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_loso_dataloaders(dataset_name, test_subject_id, batch_size=128):
    """Dynamically loads datasets based on the dataset name."""
    
    # Path construction based on dataset name
    base_path = f"dataset/processed/{dataset_name.lower()}"
    
    if not os.path.exists(f"{base_path}_x.pt"):
        raise FileNotFoundError(f"Could not find processed tensors for {dataset_name}! Ensure they are in dataset/processed/")

    x = torch.load(f"{base_path}_x.pt")
    y = torch.load(f"{base_path}_y.pt")
    p = torch.load(f"{base_path}_p.pt")

    test_mask = (p == test_subject_id)
    train_mask = (p != test_subject_id)

    train_dataset = HARDataset(x[train_mask], y[train_mask])
    test_dataset = HARDataset(x[test_mask], y[test_mask])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_dataset_config(dataset_name):
    """Returns dataset-specific configurations (Nodes, Features, Classes)."""
    configs = {
        "dsads": {"num_nodes": 5, "in_features": 9, "num_classes": 19, "total_subjects": 8},
        "opportunity": {"num_nodes": 5, "in_features": 9, "num_classes": 18, "total_subjects": 4},
        "realdisp": {"num_nodes": 9, "in_features": 9, "num_classes": 33, "total_subjects": 17}
    }
    return configs[dataset_name.lower()]