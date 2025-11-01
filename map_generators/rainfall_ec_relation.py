# ------------------------------------------
# Phase 6 – Rainfall vs EC Relationship
# ------------------------------------------
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("MLPROJECTFINALDATA.csv")

# Map hotspot labels to readable categories
label_map = {0: "Safe", 1: "At-Risk", 2: "Hotspot"}
df["Risk_Label"] = df["hotspot_label"].map(label_map)

# Quick sanity check
print(df[["rain_actual_24h", "EC", "Risk_Label"]].head())

# Create scatter plot
fig = px.scatter(
    df,
    x="rain_actual_24h",
    y="EC",
    color="Risk_Label",
    color_discrete_map={"Safe": "green", "At-Risk": "orange", "Hotspot": "red"},
    size_max=8,
    title="Relationship Between Rainfall and Electrical Conductivity (EC)",
    labels={
        "rain_actual_24h": "Rainfall (mm, last 24h)",
        "EC": "Electrical Conductivity (µS/cm)",
        "Risk_Label": "Contamination Category"
    },
    template="plotly_white"
)

# Add trend lines for context
fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="black")))
fig.update_layout(
    legend=dict(title="Hotspot Level", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(size=12)
)

# Save and show
fig.write_html("rainfall_ec_relation.html")
print("✅ Saved interactive plot as rainfall_ec_relation.html")

fig.show()
