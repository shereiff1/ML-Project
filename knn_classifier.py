from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def train_knn(X, y):
    print(
        "Training KNN (Fixed Params: n_neighbors=5, weights='distance', metric='cosine')..."
    )

    model = KNeighborsClassifier(
        n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1
    )

    # 5-fold CV
    scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    val_acc = scores.mean()

    model.fit(X, y)
    print(f"KNN Cross-Val Score: {val_acc:.4f}")
    return model, val_acc
