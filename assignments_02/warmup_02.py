print("# --- scikit-learn API ---")

print("\n# The scikit-learn Q1")
import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
slope = model.coef_[0]
intercept = model.intercept_
pred_4_years = model.predict([[4]])[0]
pred_8_years = model.predict([[8]])[0]
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"Predicted salary for 4 years of experience: {pred_4_years}")
print(f"Predicted salary for 8 years of experience: {pred_8_years}")

print("\n# The scikit-learn Q2")
x = np.array([10, 20, 30, 40, 50])
print(f"Shape of x: {x.shape}")
x = x.reshape(-1, 1)
print(f"Shape of x after reshape: {x.shape}")
# scikit-learn needs X to be 2D because each row represents a sample and each column represents a feature.
# Even if there is only one feature, it still needs to be in a column format.

print("\n# The scikit-learn Q3")
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"\nNumber of points in each cluster: {np.bincount(labels)}")

plt.figure()
plt.scatter(
    X_clusters[:, 0],
    X_clusters[:, 1],
    c=labels,
    cmap="viridis",
    s=50,
    edgecolor="k",
    label="Data Points",
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Cluster Centers",
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/kmeans_clusters.png")
plt.close()
