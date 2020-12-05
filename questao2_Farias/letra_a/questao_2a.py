import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

df = pd.read_csv("wine.csv")

X = np.array(df.drop('target', axis = 1))
target_df = np.array(df['target'])
y = target_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

knn = KNN(3)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)



acc = np.sum((pred == y_test) / len(y_test))
print("A acurácia é de: " + str(acc*100)+"%")
