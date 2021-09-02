### Stuff this article aims to cover

* **GMM** vs **KMeans**
* The **EM Algorithm** in the context of **GMM**
* How to generate new data using **GMM**


# Introduction

The **GMM** algorithm is a **clustering** algorithm. It can be viewed as an extension of the ideas behind **KMeans** (another clustering algorithm). It is also a tool that can be used for tasks other than simple clustering.

# GMM vs KMeans

Before diving deeper into the differences between these 2 clustering algorithms, let's generate some sample data and plot it.

```py
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(
    n_samples=500, centers=5,
    cluster_std=0.9, random_state=17
)

kmeans= KMeans(5, random_state=420)
labels = kmeans.fit(X).predict(X)
plt.figure(dpi=175)
plt.scatter(X[ : , 0], X[ : , 1], c=labels, s=7, cmap='viridis')
```

![image1](./images/image1.png)

We generated our sample data and we applied the **KMeans** algorithm. After looking at the way each point has been assigned to its cluster we notice that it seems to be a slight overlap between the 2 top right clusters. And this observation leads us to expect that the clustering assignments for some points is more certain than clustering assignments to over points. Well, **KMeans** has no measure of probability or uncertainty for clustering assignments (and if you probably guessed it **GMM** does).

We can visualize the way **KMeans** assigns clusters by placing a circle (in higher dimensions, a hyper-sphere) at the center of each cluster, with a radius defined by the most distant point in the cluster. In the training set, every point outside the circle is not considered a member of the cluster. 

```py
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    plt.figure(dpi=175)
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=7, cmap='viridis', zorder=2)

    # plot the representation of the k-means model
    centers = kmeans.cluster_centers_
    radii = [
        cdist(X[labels == i], [center]).max()
        for i, center in enumerate(centers)
    ]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(
            c, r, fc='#DDDDDD', lw=3, alpha=0.5, zorder=1
        ))


kmeans = KMeans(n_clusters=5, random_state=420)
plot_kmeans(kmeans, X)
```

![image2](./images/image2.png)


