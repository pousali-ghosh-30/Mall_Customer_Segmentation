# Mall Customer Segmentation

Group mall customers into natural segments using **K-Means clustering**, then explore the results in an interactive **Streamlit** app.

The project has two parts:

1. A Kaggle notebook that explores the data, finds the best number of clusters with the Elbow method, and trains the model.
2. A Streamlit app that shows the colour-coded clusters and predicts the segment of a new customer.

## Dataset

- **Name:** Customer Segmentation Tutorial in Python (`Mall_Customers.csv`)
- **Source:** [Kaggle, by Vijay Choudhary](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Size:** 200 rows, 5 columns, no missing values

| Column | Description |
|---|---|
| CustomerID | Unique ID for each customer |
| Gender | Male or Female (may appear as `Genre` in the CSV) |
| Age | Age in years |
| Annual Income (k$) | Yearly income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall based on spending behaviour |

Only **Annual Income** and **Spending Score** are used for clustering.

## Features

- Scatter plot of income vs spending score
- Elbow method and silhouette score to choose the number of clusters
- K-Means with k = 5, with colour-coded clusters
- Streamlit app with a cluster chart, a segment summary table and a filterable data table
- Sidebar tool to find the segment of a new customer

## Tech stack

Python, pandas, matplotlib, seaborn, scikit-learn, joblib, Streamlit

## Project structure

```
Mall_segmentation/
    app.py                          # Streamlit app
    Mall_Customers.csv              # original dataset
    customers_with_clusters.csv     # dataset with the cluster column added
    kmeans_model.pkl                # trained K-Means model
    scaler.pkl                      # fitted StandardScaler
    requirements.txt                # Python dependencies
    README.md
```

## How it works

1. Load the CSV with pandas and select Annual Income and Spending Score.
2. Scale both features with `StandardScaler` so neither dominates the distance calculation.
3. Run K-Means for k = 1 to 10 and plot the inertia (Elbow method). The bend appears at **k = 5**.
4. Train the final model, assign each customer a cluster, and plot the result.
5. Name each segment from its average income and spending score.
6. Save the model and scaler with joblib so the app can reuse them.

## Customer segments

| Segment | Income | Spending |
|---|---|---|
| Premium customers | High | High |
| Careful spenders | High | Low |
| Standard customers | Average | Average |
| Enthusiastic shoppers | Low | High |
| Budget customers | Low | Low |

Cluster numbers can change between runs, so match the names in `SEGMENT_NAMES` (inside `app.py`) to your own cluster averages.

## Getting started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Mall_segmentation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## Re-training the model

To train the model again, run the notebook on Kaggle (add the dataset linked above), then download `kmeans_model.pkl`, `scaler.pkl` and `customers_with_clusters.csv` into this folder.

> **Note:** use the same scikit-learn version for training and for running the app. A mismatch can cause warnings or errors when loading the `.pkl` files.

## Deployment

You can deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this folder to a GitHub repository (keep the `.pkl` files and `Mall_Customers.csv`).
2. Sign in to Streamlit Community Cloud and choose **New app**.
3. Select the repository, the `main` branch and `app.py`, then deploy.

## Possible improvements

- Add Age as a third feature and view the clusters in 3D
- Compare with hierarchical clustering or DBSCAN
- Add a downloadable CSV of each segment
- Show the elbow plot inside the app

## Credits

Dataset by Vijay Choudhary on Kaggle. Please check the dataset page for its current license and usage terms.
