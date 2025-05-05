import torch
import torchvision.transforms as transforms
from PIL import Image
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .models import Cifar10CnnModel, UploadsImage, MisclassifiedImage  # Import models
from django.core.files import File
import os

from .models import Cifar10CnnModel  # Import your model class

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), 'retrained_model.pth')

model = Cifar10CnnModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# List of classes   
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Paths for saving misclassified images
MISCLASSIFIED_DIR = os.path.join(settings.MEDIA_ROOT, 'misclassified')
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PredictImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=400)

        # Read and save uploaded image
        image = request.FILES['image']
        img = Image.open(image).convert('RGB')

        image_name = f"uploaded_{image.name}"
        image_path = os.path.join(UPLOAD_DIR, image_name)
        img.save(image_path)

        # ✅ Save to UploadsImage model
        with open(image_path, 'rb') as f:
            uploads_instance = UploadsImage.objects.create(image=File(f, name=image_name))

        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)

        max_prob = max_prob.item()
        predicted_label = CLASSES[predicted.item()] if max_prob >= 0.5 else "not matched"

        correct_label = request.data.get('correct_label')

        # If incorrect, save misclassified image
        if correct_label and predicted_label != correct_label and predicted_label != "not matched":
            misclassified_name = f"{correct_label}_{predicted_label}_{image.name}"
            misclassified_path = os.path.join(MISCLASSIFIED_DIR, misclassified_name)
            img.save(misclassified_path)

            # ✅ Save to MisclassifiedImage model
            with open(misclassified_path, 'rb') as f:
                misclassified_instance = MisclassifiedImage.objects.create(image=File(f, name=misclassified_name))

        # Generate image URL
        image_url = request.build_absolute_uri(settings.MEDIA_URL + f"uploads/{image_name}")

        return Response({
            "predicted_label": predicted_label,
            "confidence": round(max_prob * 100, 2),
            "image_url": image_url
        }, status=200)
