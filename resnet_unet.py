import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models import ResNet50_Weights
from base_models import DoubleConv

class RESNET_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, pretrained=True
    ):
        super(RESNET_UNET, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Encoder from pretrained ResNet50
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels, /2
        self.maxpool = resnet.maxpool  # /2
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels, /2
        self.encoder4 = resnet.layer3  # 1024 channels, /2
        self.encoder5 = resnet.layer4  # 2048 channels, /2

        self.bottleneck = DoubleConv(2048, 1024)

        # Decoder - need 5 upsampling stages to get back to original size
        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(2048, 1024)  # 1024(skip from enc4) + 1024(up)
        
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(1024, 512)  # 512(skip from enc3) + 512(up)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 256(skip from enc2) + 256(up)
        
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64(skip from enc1) + 64(up)
        
        # Final upsample to match input resolution
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(32, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def name(self):
        return "res_unet_atten"

    def forward(self, x):
        # Store original size
        original_size = x.shape[2:]
        
        # Encoder with skip connections
        enc1 = self.encoder1(x)  # 64 channels, H/2
        enc1_pooled = self.maxpool(enc1)  # H/4
        enc2 = self.encoder2(enc1_pooled)  # 256 channels, H/4
        enc3 = self.encoder3(enc2)  # 512 channels, H/8
        enc4 = self.encoder4(enc3)  # 1024 channels, H/16
        enc5 = self.encoder5(enc4)  # 2048 channels, H/32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder with skip connections
        x = self.up1(bottleneck)
        x = TF.resize(x, size=enc4.shape[2:])
        x = torch.cat([enc4, x], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = TF.resize(x, size=enc3.shape[2:])
        x = torch.cat([enc3, x], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = TF.resize(x, size=enc2.shape[2:])
        x = torch.cat([enc2, x], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = TF.resize(x, size=enc1.shape[2:])
        x = torch.cat([enc1, x], dim=1)
        x = self.dec4(x)
        
        # Final upsample to original resolution
        x = self.up5(x)
        x = self.dec5(x)
        
        # Ensure output matches input size
        x = TF.resize(x, size=original_size)
        
        return self.final_conv(x)

if __name__ == "__main__":
    x = torch.randn((2, 3, 480, 480))
    model = RESNET_UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == (2, 1, 480, 480), f"Expected (2, 1, 480, 480), got {preds.shape}"
    print("Output shape matches input shape")