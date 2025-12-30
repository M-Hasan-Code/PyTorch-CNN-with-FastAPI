from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
from contextlib import asynccontextmanager

class CNN(nn.Module):
    def __init__(self, input_features=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_features, 128, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),          
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 55 * 55, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model_path = "my_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

device = torch.device("cpu")
model = CNN(input_features=1).to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

torch.set_num_threads(os.cpu_count() or 4) 

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    
    transforms.Resize((224, 224)),                  
    transforms.ToTensor(),                          
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Model loaded successfully on CPU. Server starting...")
    yield
    print("Server shutting down...")

app = FastAPI(
    title="Grayscale CNN Image Classifier",
    description="Upload a grayscale-compatible image for classification (4 classes)",
    version="1.0",
    lifespan=lifespan
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")

    try:
        image = Image.open(io.BytesIO(contents))

        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {f"class_{i}": round(float(p), 4) for i, p in enumerate(probabilities[0])}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


