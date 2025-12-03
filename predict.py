import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Configuration ---
TEST_CSV = 'test.csv'
OUTPUT_FILE = 'submission.csv'
MODEL_PATH = 'best_model.pth' # Ensure this matches your saved model name
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Test Dataset Class ---
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # The 'id' column contains "test_data_v2/filename.jpg"
        img_path = self.df.iloc[idx]['id']
        
        # We use the full path from the CSV as the ID for submission
        img_id = img_path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Create a blank image if file is missing to prevent crash
            # (Prints a warning so you know)
            print(f"Warning: Could not read {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        return image, img_id

# --- 2. Load Model ---
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    
    # Initialize model architecture
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Change classifier to match training (1 output node)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("   Make sure you are in the correct folder or trained the model successfully.")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. Main Prediction Loop ---
def make_predictions():
    # 1. Load CSV
    if not os.path.exists(TEST_CSV):
        print(f"❌ Error: {TEST_CSV} not found!")
        return

    df = pd.read_csv(TEST_CSV)
    print(f"Found {len(df)} test entries in {TEST_CSV}.")

    # 2. Check for Image Folder
    # The first entry looks like 'test_data_v2/filename.jpg'
    first_path = df.iloc[0]['id']
    folder_name = os.path.dirname(first_path) # Extracts 'test_data_v2'
    
    if not os.path.exists(folder_name):
        print(f"\n❌ CRITICAL ERROR: Folder '{folder_name}' not found!")
        print(f"   The CSV expects images to be in a folder named: {folder_name}")
        print(f"   Current directory contains: {[d for d in os.listdir() if os.path.isdir(d)]}")
        print("   -> Please rename your test image folder to match or move it here.\n")
        return
    else:
        print(f"✅ Found image folder: {folder_name}")

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

    print(f"Starting predictions on {DEVICE}...")
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader):
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Convert logits to probability (0 to 1)
            probs = torch.sigmoid(outputs)
            
            # Convert to class (0 or 1)
            preds = (probs > 0.5).float().cpu().numpy()
            
            predictions.extend(preds.flatten().astype(int))
            ids.extend(img_ids)

    # 5. Save Submission
    submission_df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Success! Submission saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("Top 5 rows:")
    print(submission_df.head())

if __name__ == "__main__":
    make_predictions()