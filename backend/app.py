from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
import torch, io
from PIL import Image
import numpy as np
import cv2
from models import CNN, UNet  # Make sure these are defined properly

app = FastAPI()

# Serve frontend files (optional if frontend hosted elsewhere)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Load models
cnn = CNN(num_classes=30)
cnn.load_state_dict(torch.load("tree_classifier.pth", map_location="cpu"))
cnn.eval()

unet = UNet()
unet.load_state_dict(torch.load("backend/unet.pth", map_location="cpu"))
unet.eval()

class_names = ['amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa', 'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad']

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Segmentation with U-Net
    with torch.no_grad():
        mask = torch.sigmoid(unet(input_tensor))[0][0].numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, image.size)

    # Get bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        crop = image.crop((x, y, x+w, y+h))
    else:
        crop = image
        x, y, w, h = 0, 0, image.width, image.height

    # Classify
    crop_tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        pred = cnn(crop_tensor)
    label = class_names[int(pred.argmax())]

    return {
        "label": label,
        "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)} if contours else None
    }
