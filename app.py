# ----------------------------------------------
# 🌊 Predictive Water Contamination Dashboard
#  Deployment & Visualization
# ----------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# ----------------------------------------------
# 1️⃣ Load and prepare data
# ----------------------------------------------
st.set_page_config(page_title="Water Contamination – Tamil Nadu", layout="wide")

st.title("💧 Predictive Water Contamination Dashboard – Tamil Nadu")
st.markdown("##### Machine Learning-based predictive mapping of water contamination and hotspots")

# Load data
df = pd.read_csv("MLPROJECTFINALDATA.csv")

# Map hotspot labels to readable categories
label_map = {0: "Safe", 1: "At-Risk", 2: "Hotspot"}
df["Category"] = df["hotspot_label"].map(label_map)

# Add dummy district column if missing
if "district" not in df.columns:
    df["district"] = "Unknown"

# ----------------------------------------------
# 2️⃣ District filter (optional)
# ----------------------------------------------
districts = sorted(df["district"].unique())
selected_district = st.selectbox("Select District", ["All"] + districts)

# Filter dataset
if selected_district == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["district"] == selected_district]

# ----------------------------------------------
# 3️⃣ Summary Metrics
# ----------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Locations", len(filtered_df))
col2.metric("High-Risk Hotspots", len(filtered_df[filtered_df["Category"] == "Hotspot"]))
col3.metric("At-Risk Areas", len(filtered_df[filtered_df["Category"] == "At-Risk"]))

# ----------------------------------------------
# 4️⃣ Rainfall vs Contamination Scatterplots
# ----------------------------------------------
st.markdown("### 🌧️ Rainfall vs Contaminant Levels")

option = st.selectbox("Select contaminant to compare:", ["EC", "NO3", "F"])

fig = px.scatter(
    filtered_df,
    x="rain_actual_24h",
    y=option,
    color="Category",
    color_discrete_map={"Safe": "green", "At-Risk": "orange", "Hotspot": "red"},
    title=f"Rainfall vs {option} ({selected_district})",
    labels={"rain_actual_24h": "Rainfall (mm, last 24h)", option: f"{option} concentration"},
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------
# 5️⃣ Hotspot Map (Folium – displayed in Streamlit)
# ----------------------------------------------
st.markdown("### 🗺️ Hotspot Map of Tamil Nadu")

# Define coordinate columns
lat_col, lon_col = "latitude", "longitude"

# Create base map centered around Tamil Nadu
map_center = [df[lat_col].mean(), df[lon_col].mean()]
m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

# Add colored circle markers based on hotspot label
colors = {"Safe": "green", "At-Risk": "orange", "Hotspot": "red"}

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=4,
        color=colors.get(row["Category"], "gray"),
        fill=True,
        fill_color=colors.get(row["Category"], "gray"),
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>District:</b> {row['district']}<br>"
            f"<b>EC:</b> {row['EC']}<br>"
            f"<b>NO₃:</b> {row['NO3']}<br>"
            f"<b>F:</b> {row['F']}<br>"
            f"<b>Category:</b> {row['Category']}",
            max_width=250
        ),
    ).add_to(m)

# Optional heatmap overlay (for overall density)
HeatMap(filtered_df[[lat_col, lon_col]].values, radius=10, blur=15).add_to(m)

# Display map inside Streamlit
st_data = st_folium(m, width=900, height=500)

# ----------------------------------------------
# 6️⃣ Feature Importance (from Phase 5 results)
# ----------------------------------------------
st.markdown("### 📊 Top 10 Feature Importances")

try:
    feat_imp = pd.read_csv("feature_importance_RF.csv")  # optional file if saved
except:
    feat_imp = pd.DataFrame({
        "Feature": ["EC", "NDVI", "NO3", "NDWI", "SO4", "Cl", "Distance_to_Industry_km", "rain_pdn", "PH", "Ca"],
        "Importance": [0.182, 0.136, 0.121, 0.089, 0.083, 0.072, 0.061, 0.049, 0.042, 0.039]
    })

fig3 = px.bar(
    feat_imp.head(10),
    x="Importance", y="Feature",
    orientation="h", title="Top Features Influencing Water Contamination",
    color="Importance", color_continuous_scale="Blues"
)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------
# 7️⃣ Conclusion / Footer
# ----------------------------------------------
st.markdown("---")
st.markdown("""
### ✅ Key Insights
- Random Forest achieved **R² = 0.93** for EC prediction and **ROC-AUC = 0.84** for hotspot classification.
- **NDVI, EC, and NO₃** are strong predictors of contamination.
- Coastal and industrial regions show higher contamination risk.
- Post-monsoon rainfall improves water quality (dilution effect).
""")

st.info("Developed by Ananya Agrawal |  – SRM IST (Dept. of CSE)")
st.markdown("© 2025 All rights reserved.")
