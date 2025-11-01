# ------------------------------------------
#  District-wise Hotspot Mapping
# ------------------------------------------

import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip

# Load dataset
df = pd.read_csv("MLPROJECTFINALDATA.csv")

# Ensure 'district' column exists
if "district" not in df.columns:
    print("⚠️ No 'district' column found. Add one manually or via reverse geocoding.")
    df["district"] = "Unknown"

# Convert district to string for consistency
df["district"] = df["district"].astype(str)

# Aggregate hotspot counts per district
summary = (
    df.groupby("district")["hotspot_label"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Compute weighted risk score
summary["Risk_Score"] = (
    summary.get(1, 0) * 1 + summary.get(2, 0) * 2
) / (
    summary.get(0, 0) + summary.get(1, 0) + summary.get(2, 0)
)

# Load Tamil Nadu district GeoJSON
tamil_gdf = gpd.read_file("TN_DISTRICTS.geojson")

# Normalize column names
tamil_gdf.columns = tamil_gdf.columns.str.lower()

# Ensure 'district' column exists and matches
if "district" not in tamil_gdf.columns:
    dist_col = [c for c in tamil_gdf.columns if "dist" in c][0]
    tamil_gdf.rename(columns={dist_col: "district"}, inplace=True)

# Convert district to string for merge compatibility
tamil_gdf["district"] = tamil_gdf["district"].astype(str)
summary["district"] = summary["district"].astype(str)

# Merge GeoJSON with summary data
merged = tamil_gdf.merge(summary, on="district", how="left")

# Create Folium map
m = folium.Map(location=[10.8, 78.7], zoom_start=7, tiles="CartoDB positron")

# Add choropleth layer
folium.Choropleth(
    geo_data=merged,
    name="Risk Level",
    data=merged,
    columns=["district", "Risk_Score"],
    key_on="feature.properties.district",
    fill_color="YlOrRd",
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Average Contamination Risk (0–2)"
).add_to(m)

# Add tooltip for interactivity
tooltip = GeoJsonTooltip(
    fields=["district", "Risk_Score"],
    aliases=["District:", "Avg Risk Score:"],
    localize=True
)

folium.GeoJson(
    merged,
    name="Districts",
    style_function=lambda x: {"fillOpacity": 0, "color": "black", "weight": 0.5},
    tooltip=tooltip
).add_to(m)

# Save map
m.save("district_hotspot_map.html")
print("✅ Saved district_hotspot_map.html")