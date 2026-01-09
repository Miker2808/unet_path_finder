import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from vggunet import VGG_UNET
from resnet_unet_atten import RESNET_UNET_ATTEN
from dataset import MassachusettsRoadsDataset

from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    get_val_loss,
    EarlyStopping,
    CombinedLoss
)

# Hyperparameters etc.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 12
IMAGE_HEIGHT = 16*93
IMAGE_WIDTH = 16*93
PIN_MEMORY = True
LEARNING_RATE = 1e-4
TRAIN_VAL_SPLIT = 0.85
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001
DICE_BCE_ALPHA = 0.6

MODEL_PATH = "weights/"
LOAD_MODEL = True
SAVE_PREDICTIONS = True

# Single dataset directories
IMAGE_DIR = "dataset/large/images"
MASK_DIR = "dataset/large/masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for data, targets in loop:
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.amp.autocast(device_type=DEVICE):  # type: ignore
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=180, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.GridDistortion(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create full dataset and split
    full_dataset = MassachusettsRoadsDataset(IMAGE_DIR, MASK_DIR, transform=None)
    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with transforms using split indices
    train_dataset = Subset(
        MassachusettsRoadsDataset(IMAGE_DIR, MASK_DIR, transform=train_transform),
        train_subset.indices
    )
    val_dataset = Subset(
        MassachusettsRoadsDataset(IMAGE_DIR, MASK_DIR, transform=val_transform),
        val_subset.indices
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, shuffle=False
    )

    model = RESNET_UNET_ATTEN(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = CombinedLoss(alpha=DICE_BCE_ALPHA)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA)
    scaler = torch.amp.GradScaler('cuda')  # type: ignore

    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_PATH + model.name + ".pth.tar"), model)

    print("Initial validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        val_loss = get_val_loss(val_loader, model, loss_fn, DEVICE)
        print(f"Validation Loss: {val_loss:.5f}")
        
        scheduler.step(epoch + 1)
        check_accuracy(val_loader, model, device=DEVICE)
        
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)  # type: ignore
            break
        
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
        save_checkpoint(checkpoint, filename=MODEL_PATH + model.name + ".pth.tar")

    # Final save
    print("\nFinal validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)
    save_checkpoint({"state_dict": model.state_dict()}, filename=MODEL_PATH + model.name + ".pth.tar")
    if SAVE_PREDICTIONS:
        save_predictions_as_imgs(val_loader, model, "saved_images/{model.name}/final/", DEVICE)


if __name__ == "__main__":
    main()