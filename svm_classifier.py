from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def train_svm(X, y):
    print("Training SVM (Fixed Params: C=10, kernel='rbf')...")
    model = SVC(C=10, kernel="rbf", gamma="scale", probability=True)
    scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    val_acc = scores.mean()
    model.fit(X, y)
    print(f"SVM Cross-Val Score: {val_acc:.4f}")
    return model, val_acc
