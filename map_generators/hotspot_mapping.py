# ------------------------------------------
# Mapping & Visualization
# ------------------------------------------
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Load data (with model predictions)
df = pd.read_csv("MLPROJECTFINALDATA.csv")

# Safety check
print("Rows:", len(df))
print(df.columns)

# Use coordinates + labels
lat_col, lon_col = "latitude", "longitude"
label_col = "hotspot_label"

# Create base map (center of Tamil Nadu)
m = folium.Map(location=[10.5, 78.5], zoom_start=7, tiles="CartoDB positron")

# Color by label
colors = {0:"green", 1:"orange", 2:"red"}
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=4,
        color=colors.get(row[label_col], "gray"),
        fill=True,
        fill_color=colors.get(row[label_col], "gray"),
        fill_opacity=0.8
    ).add_to(m)

# Optional heat layer (for hotspots)
HeatMap(df[[lat_col, lon_col]].values, radius=10, blur=15).add_to(m)

m.save("hotspot_map_tamilnadu.html")
print("✅ hotspot_map_tamilnadu.html saved")
