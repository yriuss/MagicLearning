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

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        #random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [[4.9,3.0,1.4,0.2], [5.1,3.5,1.4,0.2],\
                          [4.7,3.2,1.3,0.2]]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

        # Classify samples as the index of their clusters
        
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
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
df = pd.read_csv("iris.csv")

X = np.array(df.drop('target', axis = 1))
clusters = 3
k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
sb.pairplot(df, hue = "target")

df_new = df.drop('target', axis = 1)

y_pred_list = y_pred.tolist()

setosa_error = 0
versicolor_error = 0
virginica_error = 0

setosa_total = 0
versicolor_total = 0
virginica_total = 0
target_df = df['target'].tolist()
for idx,element in enumerate(y_pred_list):
    if (y_pred_list[idx] == 0):
        y_pred_list[idx] = 'versicolor'
        
    elif(y_pred_list[idx] == 1):
        y_pred_list[idx] = 'virginica'
    else:
        y_pred_list[idx] = 'setosa'
        
    if(target_df[idx] == 'setosa'):
        setosa_total = setosa_total + 1
        if(y_pred_list[idx] != 'setosa'):
            setosa_error = setosa_error + 1
    
    if(target_df[idx] == 'virginica'):
        virginica_total = virginica_total + 1
        if(y_pred_list[idx] != 'virginica'):
            virginica_error = virginica_error + 1
    
    if(target_df[idx] == 'versicolor'):
        versicolor_total = versicolor_total + 1
        if(y_pred_list[idx] != 'versicolor'):
            versicolor_error = versicolor_error + 1


df_new['target'] = y_pred_list

sb.pairplot(df_new, hue = "target", hue_order= ['setosa','versicolor', 'virginica'])

print("O acerto percentual da setosa é: ", 100*(1-setosa_error/setosa_total),'%\n')

print("O acerto percentual da virginica é: ",\
      100*(1-virginica_error/virginica_total), '%\n')

print("O acerto percentual da versicolor é: ",\
      100*(1-versicolor_error/versicolor_total), '%\n')

#plt.show()
