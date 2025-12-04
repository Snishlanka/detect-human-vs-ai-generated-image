import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

# --- Configuration ---
TEST_CSV = 'test.csv'
OUTPUT_FILE = 'submission_version_2.csv'
MODEL_PATH = 'best_model_version2.pth' 
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a unique log filename based on the current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f'prediction_log_{current_time}.txt'

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

# --- 1. Test Dataset Class ---
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['id']
        img_id = img_path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not read {img_path}. Using blank image. Error: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        return image, img_id

# --- 2. Load Model ---
def load_model(model_path):
    logger.info(f"Loading model from {model_path}...")
    
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: Model file '{model_path}' not found!")
        logger.error("   Make sure you are in the correct folder or trained the model successfully.")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. Main Prediction Loop ---
def make_predictions():
    logger.info(f"Starting Prediction Run. Log file: {LOG_FILE}")

    # 1. Load CSV
    if not os.path.exists(TEST_CSV):
        logger.error(f"Error: {TEST_CSV} not found!")
        return

    df = pd.read_csv(TEST_CSV)
    logger.info(f"Found {len(df)} test entries in {TEST_CSV}.")

    # 2. Check for Image Folder
    first_path = df.iloc[0]['id']
    folder_name = os.path.dirname(first_path)
    
    if not os.path.exists(folder_name):
        logger.error(f"CRITICAL ERROR: Folder '{folder_name}' not found!")
        logger.error(f"   The CSV expects images to be in a folder named: {folder_name}")
        logger.error(f"   Current directory contains: {[d for d in os.listdir() if os.path.isdir(d)]}")
        return
    else:
        logger.info(f"Found image folder: {folder_name}")

    # 3. Setup Data Loader
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(df, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Predict
    model = load_model(MODEL_PATH)
    
    predictions = []
    ids = []

    logger.info(f"Starting batch predictions on {DEVICE}...")
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            
            predictions.extend(preds.flatten().astype(int))
            ids.extend(img_ids)

    logger.info("Prediction loop complete.")

    # 5. Save Submission
    submission_df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    abs_path = os.path.abspath(OUTPUT_FILE)
    logger.info(f"Success! Submission saved to: {abs_path}")
    
    logger.info("Top 5 rows of submission:")
    logger.info("\n" + submission_df.head().to_string())

if __name__ == "__main__":
    make_predictions()