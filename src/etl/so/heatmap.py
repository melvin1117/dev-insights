import json
import ujson
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# Load JSON data using ujson
with open("./asset/so/survey_results_public.json", "r") as f:
    data = ujson.load(f)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

df = df.dropna(subset=["Employment", "Country"])

# Drop rows with non-numeric values in "YearsCode"
df = df[pd.to_numeric(df["YearsCode"], errors='coerce').notna()]

# Convert "YearsCode" to numeric
df["YearsCode"] = pd.to_numeric(df["YearsCode"])

# Handle missing values
df = df.dropna(subset=["Employment", "Country", "YearsCode"])

# Convert "YearsCode" to numeric
df["YearsCode"] = pd.to_numeric(df["YearsCode"])

# Filter data based on employment status (full-time)
filtered_df = df[df["Employment"].str.contains("full-time")]

# Extract relevant columns
relevant_columns = ["Country", "YearsCode"]
filtered_df = filtered_df[relevant_columns].copy()

# Mini-batch K-means clustering
mbk = MiniBatchKMeans(n_clusters=3, batch_size=10000)
mbk.fit(filtered_df["YearsCode"].values.reshape(-1, 1))

# Add cluster labels to the DataFrame
filtered_df["Cluster"] = mbk.labels_

# Prepare data for heatmap
data_for_heatmap = filtered_df.groupby("Country")["YearsCode"].mean().to_frame()
data_for_heatmap["Cluster"] = filtered_df.set_index("Country")["Cluster"]

# Prepare data for geographical plot (if desired)
# ...
# Use geospatial libraries like GeoPandas to plot heatmap on a map

# Create the heatmap
plt.figure(figsize=(10, 6))
ax = plt.pcolor(data_for_heatmap, cmap="YlOrRd")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=data_for_heatmap["YearsCode"].min(), vmax=data_for_heatmap["YearsCode"].max()))
sm.set_array([])
plt.colorbar(sm, label="Average Years of Code")

# Add labels and title
plt.xlabel("Country")
plt.ylabel("Average Years of Code")
plt.title("Average Years of Code by Country with Mini-batch K-means Clusters")

# Show the plot
plt.show()