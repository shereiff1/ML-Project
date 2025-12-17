import numpy as np
import os
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from knn_classifier import train_knn
from svm_classifier import train_svm
import feature_extraction
import cnn_feature_extraction

def load_or_generate_features(data_dir):
    hand_feats_X = os.path.join(data_dir, "features_X.npy")
    hand_feats_y = os.path.join(data_dir, "labels_y.npy")
    cnn_feats_X = os.path.join(data_dir, "features_cnn_resnet50_X.npy")
    cnn_feats_y = os.path.join(data_dir, "labels_cnn_resnet50_y.npy")
    source_dir = os.path.join(data_dir, "dataset_augmented")
    
    if not os.path.exists(hand_feats_X) or not os.path.exists(hand_feats_y):
        print("Hand-crafted features missing. Generating them...")
        if os.path.exists(source_dir):
            feature_extraction.process_dataset(source_dir, hand_feats_X, hand_feats_y)
        else:
            raise FileNotFoundError(f"Source directory {source_dir} not found.")
            
    if not os.path.exists(cnn_feats_X) or not os.path.exists(cnn_feats_y):
        print("CNN features (ResNet50) missing. Generating them...")
        if os.path.exists(source_dir):
            cnn_feature_extraction.extract_cnn_features(source_dir, cnn_feats_X, cnn_feats_y)
        else:
            raise FileNotFoundError(f"Source directory {source_dir} not found.")

    return {
        "Hand-crafted": (hand_feats_X, hand_feats_y),
        "CNN": (cnn_feats_X, cnn_feats_y)
    }

def train_and_evaluate(X, y, name, output_dir):
    print(f"\nAnalyzing {name} features...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {
        "name": name,
        "feature_shape": X.shape,
        "scaler": scaler
    }
    
    print(f"[{name}] Training SVM...")
    start_time = time.time()
    svm_model, svm_val_acc = train_svm(X_train_scaled, y_train)
    svm_train_time = time.time() - start_time
    svm_preds = svm_model.predict(X_test_scaled)
    svm_test_acc = accuracy_score(y_test, svm_preds)
    
    results["svm"] = {
        "model": svm_model,
        "val_acc": svm_val_acc,
        "test_acc": svm_test_acc,
        "time": svm_train_time
    }
    
    print(f"[{name}] Training KNN...")
    start_time = time.time()
    knn_model, knn_val_acc = train_knn(X_train_scaled, y_train)
    knn_train_time = time.time() - start_time
    knn_preds = knn_model.predict(X_test_scaled)
    knn_test_acc = accuracy_score(y_test, knn_preds)
    
    results["knn"] = {
        "model": knn_model,
        "val_acc": knn_val_acc,
        "test_acc": knn_test_acc,
        "time": knn_train_time
    }
    
    return results

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        feature_files = load_or_generate_features(current_dir)
        best_model_info = {"acc": -1, "model": None, "type": "", "feature_type": "", "scaler": None}
        
        for name, (x_path, y_path) in feature_files.items():
            X = np.load(x_path)
            y = np.load(y_path)
            res = train_and_evaluate(X, y, name, current_dir)
            if res["svm"]["test_acc"] > best_model_info["acc"]:
                best_model_info = {
                    "acc": res["svm"]["test_acc"],
                    "model": res["svm"]["model"],
                    "type": "svm",
                    "feature_type": name,
                    "scaler": res["scaler"]
                }
            if res["knn"]["test_acc"] > best_model_info["acc"]:
                best_model_info = {
                    "acc": res["knn"]["test_acc"],
                    "model": res["knn"]["model"],
                    "type": "knn",
                    "feature_type": name,
                    "scaler": res["scaler"]
                }   
        print(f"\nBest Model: {best_model_info['type'].upper()} with {best_model_info['feature_type']} features (Accuracy: {best_model_info['acc']:.4f})")
        print(f"\nSaving bundled model to {os.path.join(current_dir, 'best_model.pkl')}...")
        with open(os.path.join(current_dir, 'best_model.pkl'), 'wb') as f:
            pickle.dump(best_model_info, f)   
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
