import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import multiprocessing
import logging
from datetime import datetime

# --- Configuration ---
IMAGE_DIR = '' 
CSV_FILE = 'train.csv'

# Generate a unique log filename based on the current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f'training_log_{current_time}.txt'

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 0. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. Custom Dataset Class ---
class AIDetectionDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['file_name']
        img_path = os.path.join(self.root_dir, img_name)
        label = row['label']

        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError) as e:
            logger.warning(f"Could not read {img_path}. Using blank image. Error: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# --- 2. Data Preparation ---
def get_dataloaders(csv_file, img_dir, batch_size):
    logger.info("Loading data configuration...")
    df = pd.read_csv(csv_file)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_dataset = AIDetectionDataset(train_df, img_dir, transform=train_transforms)
    val_dataset = AIDetectionDataset(val_df, img_dir, transform=val_transforms)

    num_workers = 2 if os.name != 'nt' else 0 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

# --- 3. Model Setup (Automatic Download) ---
def build_model():
    logger.info(f"Building EfficientNet B0 model on {DEVICE}...")
    
    # Automatic download of ImageNet weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Modify classifier for AI vs Human
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    return model.to(DEVICE)

# --- 4. Training Loop ---
def train_model():
    train_loader, val_loader = get_dataloaders(CSV_FILE, IMAGE_DIR, BATCH_SIZE)
    model = build_model()
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_val_acc = 0.0

    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # --- TRAINING PHASE ---
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        scheduler.step(avg_val_loss)

        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_version2.pth")
            logger.info(">>> Best model saved (version 2)!")
        
        logger.info("-" * 30)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # --- HARDWARE LOGGING BLOCK ---
    logger.info("--- SYSTEM & HARDWARE INFO ---")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        # Convert bytes to GB
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 
        
        logger.info(f"Device: GPU ({gpu_name})")
        logger.info(f"CUDA Version: {cuda_ver}")
        logger.info(f"Total VRAM: {gpu_mem:.2f} GB")
    else:
        logger.info("Device: CPU (Training will be slow)")
    logger.info("-" * 30)
    # ------------------------------

    train_model()