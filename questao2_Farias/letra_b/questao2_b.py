import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import make_blobs

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans():

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]

        self.centroids = [[14.75,1.73,2.39,11.4,91,3.1,3.69,.43,2.81,5.4,1.25,2.73,1150], [12.7,3.87,2.4,23,101,2.83,2.55,.43,1.95,2.57,1.19,3.13,463],\
                          [13.73,4.36,2.26,22.5,88,1.28,.47,.52,1.15,6.62,.78,1.75,520]]

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape


        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            if self._is_converged(centroids_old, self.centroids):
                break

        
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
'''           
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)
    
clusters = len(np.unique(y))
print(clusters)
k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
'''
'''
df = pd.read_csv("iris_new.csv")

X = np.array(df)
k = KMeans(K=3, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
'''
df = pd.read_csv("wine.csv")

X = np.array(df.drop('target', axis = 1))
clusters = 3
k = KMeans(K=clusters, max_iters=50, plot_steps=True)
print("Os centroids são inicializados em: "+str(k.centroids))
y_pred = k.predict(X)
sb.pairplot(df, hue = "target")

df_new = df.drop('target', axis = 1)

y_pred_list = y_pred.tolist()

one_error = 0
two_error = 0
three_error = 0

one_total = 0
two_total = 0
three_total = 0
target_df = df['target'].tolist()

for idx,element in enumerate(y_pred_list):
    if (y_pred_list[idx] == 1):
        y_pred_list[idx] = '2'
        
    elif(y_pred_list[idx] == 2):
        y_pred_list[idx] = '3'
    else:
        y_pred_list[idx] = '1'
    if(target_df[idx] == 1):
        one_total = one_total + 1
        if(y_pred_list[idx] != '1'):
            one_error = one_error + 1
    
    if(target_df[idx] == 3):
        three_total = three_total + 1
        if(y_pred_list[idx] != '3'):
            three_error = three_error + 1
    
    if(target_df[idx] == 2):
        two_total = two_total + 1
        if(y_pred_list[idx] != '2'):
            two_error = two_error + 1


df_new['target'] = y_pred_list

sb.pairplot(df_new, hue = "target", hue_order= ['1','2', '3'])

print("O acerto percentual de 1 é: ", 100*(1-one_error/one_total),'%\n')

print("O acerto percentual de 2 é: ",\
      100*(1-two_error/two_total), '%\n')

print("O acerto percentual de 3 é: ",\
      100*(1-three_error/three_total), '%\n')

print("Os centroids convergidos são: "+str(k.centroids))



plt.show()
