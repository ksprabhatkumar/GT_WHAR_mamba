import os
import numpy as np
import torch
from tqdm import tqdm

def process_dsads():
    print("Processing DSADS dataset...")
    base_path = os.path.join("har_data", "DSADS", "data")
    if not os.path.exists(base_path):
        print("❌ DSADS raw data not found! Check your har_data/DSADS/data/ folder.")
        return

    all_data, all_labels, all_persons = [], [], []

    for activity_id in tqdm(range(1, 20), desc="Processing DSADS Activities"):
        for person_id in range(1, 9):
            for segment_id in range(1, 61):
                file_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}", f"s{segment_id:02d}.txt")
                try:
                    data_matrix = np.loadtxt(file_path, delimiter=',')
                    data_reshaped = data_matrix.reshape(125, 5, 9)
                    all_data.append(data_reshaped)
                    all_labels.append(activity_id - 1)
                    all_persons.append(person_id)
                except Exception:
                    pass

    if len(all_data) == 0:
        print("❌ No files processed.")
        return

    # --- NEW: Z-SCORE NORMALIZATION ---
    # Convert to numpy array first
    X_np = np.array(all_data, dtype=np.float32)
    
    # Calculate mean and std over the entire dataset across the Feature dimension (axis=3)
    # Shape of X_np is (9120, 125, 5, 9)
    means = np.mean(X_np, axis=(0, 1, 2), keepdims=True)
    stds = np.std(X_np, axis=(0, 1, 2), keepdims=True)
    
    # Avoid division by zero just in case a sensor channel is dead
    stds[stds == 0] = 1.0 
    
    # Apply standard scaling
    X_normalized = (X_np - means) / stds
    print(f"Data Normalized! Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    # -----------------------------------

    save_tensors("dsads", X_normalized, all_labels, all_persons)

def generate_placeholders(dataset_name, num_samples, timesteps, nodes, features, classes, subjects):
    """Generates identically shaped placeholder tensors so the Mamba architecture can be tested."""
    print(f"Generating pipeline placeholder tensors for {dataset_name.upper()}...")
    
    all_data = np.random.randn(num_samples, timesteps, nodes, features).astype(np.float32)
    all_labels = np.random.randint(0, classes, size=(num_samples,))
    all_persons = np.random.randint(1, subjects + 1, size=(num_samples,))
    
    save_tensors(dataset_name, all_data, all_labels, all_persons)

def save_tensors(name, data, labels, persons):
    os.makedirs("dataset/processed", exist_ok=True)
    
    X_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    P_tensor = torch.tensor(np.array(persons), dtype=torch.long)

    torch.save(X_tensor, f"dataset/processed/{name}_x.pt")
    torch.save(Y_tensor, f"dataset/processed/{name}_y.pt")
    torch.save(P_tensor, f"dataset/processed/{name}_p.pt")
    print(f"✅ Saved {name.upper()}: X {X_tensor.shape}, Y {Y_tensor.shape}, P {P_tensor.shape}\n")

if __name__ == "__main__":
    print("🚀 Starting Data Processing...\n")
    
    # 1. Process REAL DSADS Data
    process_dsads()
    
    # 2. Generate Placeholders for Opportunity (24 timesteps, 5 nodes, 18 classes, 4 subjects)
    generate_placeholders("opportunity", num_samples=2000, timesteps=24, nodes=5, features=9, classes=18, subjects=4)
    
    # 3. Generate Placeholders for Realdisp (125 timesteps, 9 nodes, 33 classes, 17 subjects)
    generate_placeholders("realdisp", num_samples=3000, timesteps=125, nodes=9, features=9, classes=33, subjects=17)
    
    print("🎉 All datasets processed and ready for main.py!")