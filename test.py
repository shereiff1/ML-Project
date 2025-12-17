import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from glob import glob

def predict(dataFilePath, bestModelPath):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    with open(bestModelPath, 'rb') as f:
        model_info = pickle.load(f)
    model = model_info['model']
    scaler = model_info['scaler']
    feature_type = model_info['feature_type']    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = None
    preprocess = None
    if feature_type == "CNN":
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet = resnet.to(device)
        resnet.eval()
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        try:
             import feature_extraction
        except ImportError:
            print("Error: feature_extraction module not found.")
            return []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(dataFilePath, ext)))
    image_files.sort()
    predictions = []
    for img_path in image_files:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            predictions.append("Error")
            continue
        feature_vec = None
        if feature_type == "CNN":
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(img_rgb)
            img_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = resnet(img_batch)
            feature_vec = output.cpu().numpy().flatten().reshape(1, -1)
            
        else:
            feature_vec = feature_extraction.extract_all_features(img_cv).reshape(1, -1)
            
        if feature_vec is not None:
            feature_scaled = scaler.transform(feature_vec)
            pred_id = 6
            try:
                probs = model.predict_proba(feature_scaled)[0]
                max_prob = np.max(probs)
                if max_prob >= 0.5:
                    model_idx = np.argmax(probs)
                    mapping = {0: 2, 1: 0, 2: 4, 3: 1, 4: 3, 5: 5}
                    pred_id = mapping.get(model_idx, 6)
            except:
                pred = model.predict(feature_scaled)[0]
                mapping = {0: 2, 1: 0, 2: 4, 3: 1, 4: 3, 5: 5}
                pred_id = mapping.get(pred, 6)
            predictions.append(pred_id)
    return predictions

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Here you could put the relative path from the current directory for the saved best model accuracy 
    model_path = os.path.join(current_dir, 'best_model.pkl')
    # Here you could put the relative path from the current directory for the test data
    data_path = os.path.join(current_dir, 'tests') 
    if os.path.exists(model_path) and os.path.exists(data_path):
        preds = predict(data_path, model_path)
        print(preds)
    else:
        print("Error: best_model.pkl or tests folder not found.")