from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route("/")
def index():
    # Load data
    df = pd.read_csv("Mall_Customers.csv")
    X = df.drop(columns=["CustomerID", "Gender"])

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Calculate statistics
    total_customers = len(df)
    avg_age = df["Age"].mean()
    avg_income = df["Annual Income (k$)"].mean()
    
    # Cluster information
    cluster_info = {}
    for cluster_num in range(n_clusters):
        cluster_data = df[df["Cluster"] == cluster_num]
        cluster_info[cluster_num] = {
            "count": len(cluster_data),
            "avg_age": cluster_data["Age"].mean(),
            "avg_income": cluster_data["Annual Income (k$)"].mean(),
            "avg_spending": cluster_data["Spending Score (1-100)"].mean()
        }
    
    # Prepare table for display
    display_count = 20
    df_display = df.head(display_count).copy()
    
    # Create HTML table
    table_html = df_display.to_html(
        classes='table table-striped table-hover',
        index=False,
        escape=False
    )

    return render_template(
        "index.html",
        tables=[table_html],
        total_customers=total_customers,
        n_clusters=n_clusters,
        avg_age=avg_age,
        avg_income=avg_income,
        cluster_info=cluster_info,
        display_count=display_count
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
