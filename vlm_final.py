import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# ---------------- PATHS ----------------
MODEL_PATH = r"C:\Users\ACER\OneDrive\Desktop\DEEPFAKE-DETECTION-USING-VLM\models\deepfake_classifier.pth"
VIDEO_PATH = r"C:\Users\ACER\OneDrive\Desktop\DEEPFAKE-DETECTION-USING-VLM\newsf.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LABELS ----------------
LABELS = {0: "REAL", 1: "FAKE"}

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- LOAD CNN MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

try:
    model.load_state_dict(checkpoint)
except:
    model.load_state_dict(checkpoint['model_state_dict'])

model.to(DEVICE)
model.eval()

print("✅ CNN Model loaded")

# ---------------- LOAD VLM (CLIP) ----------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("✅ CLIP model loaded")

# ---------------- PROMPTS ----------------
texts = [
    "a real human face with natural skin and lighting",
    "a fake AI generated face with artifacts or unnatural blending"
]

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, step=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        idx += 1

    cap.release()
    return frames

# ---------------- VLM FUNCTION ----------------
def get_vlm_probs(image):
    inputs = clip_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    return probs[0].detach().cpu().numpy()

# ---------------- MAIN PIPELINE ----------------
frames = extract_frames(VIDEO_PATH)

if len(frames) == 0:
    raise RuntimeError("❌ No frames extracted")

votes = {"REAL": 0.0, "FAKE": 0.0}

with torch.no_grad():
    for i, frame in enumerate(frames):

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            # ---- CNN ----
            img = Image.fromarray(face)
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            cnn_output = model(img_tensor)
            cnn_probs = torch.softmax(cnn_output, dim=1).cpu().numpy()[0]

            # ---- VLM ----
            vlm_probs = get_vlm_probs(img)

            # ---- FUSION ----
            final_probs = 0.7 * cnn_probs + 0.3 * vlm_probs

            pred_idx = final_probs.argmax()
            confidence = final_probs.max()
            label = LABELS[pred_idx]

            votes[label] += confidence

            print(f"Frame {i}: {label} ({confidence:.2f})")

# ---------------- FINAL RESULT ----------------
final_prediction = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"

print("\n🎥 FINAL VIDEO PREDICTION:", final_prediction)
print(f"🧮 Votes → REAL: {votes['REAL']:.2f}, FAKE: {votes['FAKE']:.2f}")