from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch, io, base64
from PIL import Image
import numpy as np
import cv2
from .models import CNN, UNet  # Your custom CNN + U-Net

app = FastAPI()

# Mount frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Load models
cnn = CNN()
cnn.load_state_dict(torch.load("backend/tree_classifier.pth", map_location="cpu"))
cnn.eval()

unet = UNet()  # Pre-trained U-Net from SMP
unet.eval()

class_names = ['amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa', 'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor', 'sugarcane', 'vad']

# Preprocessing transform
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Run segmentation with U-Net
    with torch.no_grad():
        mask = torch.sigmoid(unet(input_tensor))[0][0].numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, image.size)

    # Get bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np = np.array(image)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_np, "Tree", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        crop = image.crop((x, y, x+w, y+h))
    else:
        crop = image

    crop_tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        pred = cnn(crop_tensor)
    label = class_names[int(pred.argmax())]

    # Draw label on output image
    if contours:
        cv2.putText(image_np, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    _, im_encoded = cv2.imencode('.jpg', image_np)
    im_b64 = base64.b64encode(im_encoded).decode("utf-8")

    return {"label": label, "image_base64": im_b64}
