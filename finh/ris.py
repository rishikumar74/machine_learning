import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.write("K-Means Clustering with Elbow Method")

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "Upload Mall_Customers.csv",
    type="csv"
)

if uploaded_file is None:
    st.info("Please upload the Mall_Customers.csv file to continue.")
    st.stop()

# ---------- Load Dataset ----------
df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------- Encode Gender ----------
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# ---------- Feature Selection ----------
# Annual Income (k$) & Spending Score (1–100)
X = df.iloc[:, [3, 4]].values

# ---------- Elbow Method ----------
st.subheader("Elbow Method (Optimal K)")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss)
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

# ---------- K-Means Clustering ----------
st.subheader("K-Means Clustering Result")

k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

fig2, ax2 = plt.subplots()

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']

for i in range(k):
    ax2.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        c=colors[i],
        label=f'Cluster {i+1}'
    )

# ---------- Plot Centroids ----------
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

# ---------- Show Clustered Data ----------
df['Cluster'] = y_kmeans
st.subheader("Clustered Dataset")
st.dataframe(df)

# ---------- Download Result ----------
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Clustered Data",
    data=csv,
    file_name="clustered_customers.csv",
    mime="text/csv"
)
