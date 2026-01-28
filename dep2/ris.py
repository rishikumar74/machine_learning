import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.write("K-Means Clustering with Elbow Method")

# Load data
df = pd.read_csv("Mall_Customers.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Encode Genre
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# Select features
X = df.iloc[:, [3, 4]].values   # Annual Income & Spending Score

# ---------- Elbow Method ----------
st.subheader("Elbow Method")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss)
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

# ---------- KMeans Clustering ----------
st.subheader("K-Means Clustering Result")

k = st.slider("Select number of clusters (K)", 2, 10, 5)

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

fig2, ax2 = plt.subplots()

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown']

for i in range(k):
    ax2.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        c=colors[i],
        label=f'Cluster {i+1}'
    )

# Centroids
ax2.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    label='Centroids'
)

ax2.set_title("Clusters of Customers")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1–100)")
ax2.legend()

st.pyplot(fig2)
