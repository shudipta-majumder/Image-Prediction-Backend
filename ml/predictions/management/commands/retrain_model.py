# # your_app/management/commands/retrain_model.py
# from django.core.management.base import BaseCommand
# import torch
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from torchvision import transforms
# import os
# from django.conf import settings
# from predictions.models import Cifar10CnnModel  # Import your model

# # Path to save misclassified images
# MISCLASSIFIED_DIR = os.path.join(settings.MEDIA_ROOT, 'misclassified')
# trained_model_path = os.path.join(settings.BASE_DIR, 'predictions', 'trained_model.pth')
# retrained_model_path = os.path.join(settings.BASE_DIR, 'predictions', 'retrained_model.pth')

# # A dictionary mapping class names to integer labels (you should define this based on your dataset)
# class_name_to_label = {
#     'airplane': 0,
#     'automobile': 1,
#     'bird': 2,
#     'cat': 3,
#     'deer': 4,
#     'dog': 5,
#     'frog': 6,
#     'horse': 7,
#     'ship': 8,
#     'truck': 9
# }

# class MisclassifiedDataset(Dataset):
#     def __init__(self, misclassified_dir, transform=None):
#         self.misclassified_dir = misclassified_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(misclassified_dir)]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.misclassified_dir, self.image_files[idx])
#         img = Image.open(img_path).convert('RGB')

#         # Extract class name from the filename (before the '_') and map it to the corresponding label
#         class_name = self.image_files[idx].split('_')[0]
#         label = class_name_to_label.get(class_name, -1)  # Default to -1 if the class is not found

#         if self.transform:
#             img = self.transform(img)

#         return img, label

# class Command(BaseCommand):
#     help = 'Retrain the model with misclassified images'

#     def handle(self, *args, **kwargs):
#         self.stdout.write("Retraining the model with misclassified images...")

#         # Set up device and model
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = Cifar10CnnModel().to(device)
#         model.load_state_dict(torch.load(trained_model_path, map_location=device))
#         model.train()

#         # Load misclassified data
#         misclassified_dataset = MisclassifiedDataset(MISCLASSIFIED_DIR, transform=transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ]))
#         train_loader = DataLoader(misclassified_dataset, batch_size=32, shuffle=True)

#         # Set up optimizer and loss function
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         criterion = torch.nn.CrossEntropyLoss()

#         # Train for a few epochs (you can choose more)
#         for epoch in range(3):  # retrain for 3 epochs, adjust as needed
#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)

#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#         # Save the retrained model
#         torch.save(model.state_dict(), retrained_model_path)

#         self.stdout.write(self.style.SUCCESS('Model retrained successfully and saved to retrained_model.pth'))







# your_app/management/commands/retrain_model.py
from django.core.management.base import BaseCommand
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os
from django.conf import settings
from predictions.models import Cifar10CnnModel  # Import your CNN model

# Paths for model files and misclassified images
MISCLASSIFIED_DIR = os.path.join(settings.MEDIA_ROOT, 'misclassified')
TRAINED_MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictions', 'trained_model.pth')
RETRAINED_MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictions', 'retrained_model.pth')

# Class name to label mapping
CLASS_NAME_TO_LABEL = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}

class MisclassifiedDataset(Dataset):
    """ Dataset to load misclassified images for retraining """
    def __init__(self, misclassified_dir, transform=None):
        self.misclassified_dir = misclassified_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(misclassified_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'webp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.misclassified_dir, self.image_files[idx])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  # Skip corrupted images

        # Extract class name from the filename (assumes format "classname_xxx.jpg")
        class_name = self.image_files[idx].split('_')[0]
        label = CLASS_NAME_TO_LABEL.get(class_name, -1)  # Default to -1 if not found

        if self.transform:
            img = self.transform(img)

        return img, label

class Command(BaseCommand):
    """ Django management command to retrain the CNN model """
    help = 'Retrain the model with misclassified images'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting model retraining...")

        # Set up device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        model = Cifar10CnnModel().to(device)

        # Load existing retrained model (if available) or fall back to trained model
        if os.path.exists(RETRAINED_MODEL_PATH):
            self.stdout.write("Loading previous retrained model...")
            model.load_state_dict(torch.load(RETRAINED_MODEL_PATH, map_location=device))
        else:
            self.stdout.write("Loading original trained model...")
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))

        model.train()  # Set model to training mode

        # Load misclassified data
        misclassified_dataset = MisclassifiedDataset(
            MISCLASSIFIED_DIR,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        )

        # Ensure there are misclassified images to train on
        if len(misclassified_dataset) == 0:
            self.stdout.write(self.style.WARNING("No misclassified images found. Skipping retraining."))
            return

        train_loader = DataLoader(misclassified_dataset, batch_size=32, shuffle=True)

        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for fine-tuning
        criterion = torch.nn.CrossEntropyLoss()

        # Train for a few epochs (adjust as needed)
        EPOCHS = 3
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for images, labels in train_loader:
                if images is None or labels is None:
                    continue  # Skip corrupted data
                
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.stdout.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Save the retrained model
        torch.save(model.state_dict(), RETRAINED_MODEL_PATH)

        self.stdout.write(self.style.SUCCESS('Model retrained successfully and saved to retrained_model.pth'))
