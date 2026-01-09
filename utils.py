import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from dataset import MassachusettsRoadsDataset

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"Validation loss improved: {self.best_loss:.5f} -> {val_loss:.5f}")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        return self.early_stop
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Jaccard weight
        self.beta = beta    # Dice weight
        self.jaccard = JaccardLoss()
        self.dice = DiceLoss()
        
    def forward(self, inputs, targets):
        jaccard_loss = self.jaccard(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        return self.alpha * jaccard_loss + self.beta * dice_loss

def get_val_loss(loader, model, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            
            preds = model(x)
            
            # Resize predictions to match target if needed
            if preds.shape != y.shape:
                preds = torch.nn.functional.interpolate(
                    preds, size=y.shape[2:], mode='bilinear', align_corners=False
                )
            
            loss = loss_fn(preds, y)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(loader)

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # Add channel dimension to match model output
            
            preds = torch.sigmoid(model(x))
            
            # Resize predictions to match target if needed
            if preds.shape != y.shape:
                preds = torch.nn.functional.interpolate(
                    preds, size=y.shape[2:], mode='bilinear', align_corners=False
                )
            
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device).unsqueeze(1)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            
            # Resize predictions to match target if needed
            if preds.shape != y.shape:
                preds = torch.nn.functional.interpolate(
                    preds, size=y.shape[2:], mode='bilinear', align_corners=False
                )
            
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}/mask_{idx}.png")

    model.train()