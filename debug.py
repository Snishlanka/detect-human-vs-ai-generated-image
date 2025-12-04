import os
import sys
import time

def print_status(step, status, message):
    symbol = "✅" if status else "❌"
    print(f"{symbol} [{step}] {message}")

print("--- STARTING SYSTEM DIAGNOSTIC ---\n")

# --- 1. LIBRARY CHECK ---
print("1. Checking Libraries...")
required_libs = [
    ('pandas', 'pandas'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('sklearn', 'sklearn'),
    ('PIL', 'PIL'),
    ('tqdm', 'tqdm'),
    ('logging', 'logging')
]

all_libs_pass = True
for lib_name, import_name in required_libs:
    try:
        __import__(import_name)
        print_status(lib_name, True, "Installed")
    except ImportError:
        print_status(lib_name, False, "MISSING! Run: pip install " + lib_name)
        all_libs_pass = False

if not all_libs_pass:
    print("\n❌ CRITICAL: Missing libraries. Install them and run this again.")
    sys.exit()

# Now safe to import
import torch
import pandas as pd
from PIL import Image

# --- 2. CUDA (GPU) CHECK ---
print("\n2. Checking GPU & CUDA...")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print_status("CUDA", True, f"Available! Device: {device_name}")
    print(f"   - CUDA Version: {cuda_version}")
    print(f"   - PyTorch Version: {torch.__version__}")
    
    # Simple Tensor Test
    try:
        x = torch.tensor([1.0]).cuda()
        print_status("Tensor Test", True, "Successfully created tensor on GPU.")
    except Exception as e:
        print_status("Tensor Test", False, f"Failed to use GPU: {e}")
else:
    print_status("CUDA", False, "Not available. You are running on CPU.")
    print(f"   - PyTorch Version: {torch.__version__}")
    print("   Note: Training will be slow. If you have an NVIDIA GPU, reinstall PyTorch with CUDA support.")

# --- 3. DATA CHECK ---
print("\n3. Checking Data & Paths...")
CSV_FILE = 'train.csv'
IMG_FOLDER = 'train_data'

# Check CSV
if os.path.exists(CSV_FILE):
    print_status("train.csv", True, "Found.")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"   - Found {len(df)} rows in CSV.")
        
        # Check Image Folder
        if os.path.exists(IMG_FOLDER):
            print_status("train_data/", True, "Folder found.")
            
            # Check First Image
            first_filename = df.iloc[0]['file_name'] # e.g., 'train_data/abc.jpg'
            
            # Logic: If CSV has 'train_data/img.jpg', we look for that relative to current dir
            # If CSV has 'img.jpg', we look inside IMG_FOLDER
            if IMG_FOLDER in first_filename:
                full_path = first_filename
            else:
                full_path = os.path.join(IMG_FOLDER, first_filename)
                
            print(f"   - Checking first image: {full_path}")
            
            if os.path.exists(full_path):
                print_status("Image Access", True, "File exists on disk.")
                try:
                    img = Image.open(full_path)
                    img.verify() # Verify it's not corrupt
                    print_status("Image Read", True, "Successfully opened and verified image.")
                except Exception as e:
                    print_status("Image Read", False, f"Corrupted or unreadable: {e}")
            else:
                print_status("Image Access", False, "Could not find the file!")
                print("   DEBUG: List of files in current dir:", os.listdir('.'))
                if os.path.exists(IMG_FOLDER):
                    print(f"   DEBUG: First 5 files in {IMG_FOLDER}:", os.listdir(IMG_FOLDER)[:5])
        else:
            print_status("train_data/", False, "Folder NOT found in this directory.")
            
    except Exception as e:
        print_status("CSV Read", False, f"Could not read CSV: {e}")
else:
    print_status("train.csv", False, "File NOT found in this directory.")

print("\n--- DIAGNOSTIC COMPLETE ---")