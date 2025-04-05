import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import Image


class TumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, image_stats=[[0],[1]], mask_stats=[[0],[1]]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_stats = image_stats
        self.mask_stats = mask_stats

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img = Image.open(os.path.join(self.image_dir, img_name)).resize((256,256))
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).resize((256,256))

        images = self.transform["image"](img)
        masks = self.transform["mask"](mask)

        return images, masks


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc_conv4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid() 

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv1(x)
        p1 = self.pool1(c1)
        c2 = self.enc_conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc_conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc_conv4(p3)
        p4 = self.pool4(c4)
        # Bottleneck
        bn = self.bottleneck(p4)
        # Decoder
        u4 = self.upconv4(bn)
        merge4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec_conv4(merge4)

        u3 = self.upconv3(d4)
        merge3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec_conv3(merge3)

        u2 = self.upconv2(d3)
        merge2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec_conv2(merge2)

        u1 = self.upconv1(d2)
        merge1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec_conv1(merge1)

        out_before_activation = self.out_conv(d1)
        out = self.activation(out_before_activation)
        return out


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for i, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if i % 10 == 0 or i == total_batches:
                print(f"Processed {i}/{total_batches} batches")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Val Loss: {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_unet.pth")
            print("Saved Best Model!")

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    return model


if __name__=="__main__":
 
    if torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3080 Laptop GPU':
        base_dir = "/home/n-iznat/Desktop/DL/fp/dataset"
    else:
        base_dir = "/home/nurkeldi.iznat/dataset"

    train_image_dir = os.path.join(base_dir, "Dataset_split/Training/images")
    train_mask_dir = os.path.join(base_dir, "Dataset_split/Training/masks")

    val_image_dir = os.path.join(base_dir, "Dataset_split/Validation/images")
    val_mask_dir = os.path.join(base_dir, "Dataset_split/Validation/masks")

    test_image_dir = os.path.join(base_dir, "Dataset_split/Testing/images")
    test_mask_dir = os.path.join(base_dir, "Dataset_split/Testing/masks")

    
    transform = {
        "image": transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.], [1.0])]),
        "mask": transforms.Compose([transforms.ToTensor()])}
      


    datasets = {
    "train": TumorSegmentationDataset(train_image_dir, train_mask_dir, transform=transform),
    "val": TumorSegmentationDataset(val_image_dir, val_mask_dir, transform=transform),
    # "test": TumorSegmentationDataset(test_image_dir, test_mask_dir, transform=transform),
    }


    bs = 16

    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=bs, shuffle=True, num_workers=2),
        "val": DataLoader(datasets["val"], batch_size=bs, shuffle=True, num_workers=2),
        # "test": DataLoader(datasets["test"], batch_size=bs, shuffle=False, num_workers=2),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Now using GPU if available
    model = UNet(in_channels=1, out_channels=1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_model(model, dataloaders["train"], dataloaders["val"], criterion, optimizer, scheduler, num_epochs=100)
