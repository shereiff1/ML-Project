import cv2, torch, os, pickle, time, numpy as np
import torch.nn as nn, torchvision.models as m, torchvision.transforms as t
from collections import deque, Counter
import feature_extraction

def enhance(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.filter2D(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB), -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

def run(d, cam_url="http://192.168.1.6:8080/video"):
    md_p = os.path.join(d,'best_model.pkl')
    if not os.path.exists(md_p): return print("Err: bundled model not found.")
    m_info = pickle.load(open(md_p,'rb'))
    scaler, model, ft = m_info['scaler'], m_info['model'], m_info['feature_type']
    print(f"Loaded: {ft}, {type(model).__name__}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp = t.Compose([t.ToPILImage(), t.Resize(256), t.CenterCrop(224), t.ToTensor(), t.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    rn = m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V1) if ft=="CNN" else None
    if rn: rn = nn.Sequential(*list(rn.children())[:-1]).to(dev).eval()

    cls, buf, cap = ['cardboard','glass','metal','paper','plastic','trash'], deque(maxlen=5), cv2.VideoCapture(cam_url)
    if not cap.isOpened(): return print(f"Err: Cannot open camera stream {cam_url}")

    lbl, col, conf, last_t = "Waiting...", (255,255,255), 0.0, 0
    
    while True:
        ret, fr = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue
        
        h, w = fr.shape[:2]
        roi = fr
        if cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).std() < 20:
            lbl, col, conf, buf = "No Object", (100,100,100), 0.0, deque(maxlen=5)
        elif time.time() - last_t >= 1.0:
            f_vec = None
            if ft == "CNN":
                with torch.no_grad(): 
                    f_vec = rn(pp(enhance(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)).cpu().numpy().flatten().reshape(1,-1)
            else: 
                f_vec = feature_extraction.extract_all_features(roi).reshape(1,-1)
            
            p_prob = 1.0
            try: 
                probs = model.predict_proba(scaler.transform(f_vec))[0]
                idx, p_prob = np.argmax(probs), probs[np.argmax(probs)]
            except: 
                idx = model.predict(scaler.transform(f_vec))[0]
            
            buf.append("Unknown" if p_prob < 0.7 else cls[idx])
            if buf:
                lbl = Counter(buf).most_common(1)[0][0]
                col = (0,0,255) if lbl=="Unknown" else (0,255,0)
                conf = p_prob
            last_t = time.time()
        cv2.putText(fr, "Place object in view", (10, 70), 0, 0.7, (255,255,255), 2)
        cv2.putText(fr, f"{lbl} ({conf:.2f})", (10, 30), 0, 1, col, 2)
        cv2.imshow('Class', fr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": 
    run(os.path.dirname(os.path.abspath(__file__)))