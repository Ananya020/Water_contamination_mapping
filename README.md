# Predictive Water Contamination Mapping – Tamil Nadu 🌊

### Overview
This project uses Machine Learning to predict groundwater and surface water contamination across Tamil Nadu.  
By integrating rainfall, satellite (NDVI/NDWI), and chemical parameters, we classify locations into *Safe*, *At-Risk*, and *Hotspot* zones.

---

### Objectives
- Predict contaminant concentrations (EC, NO₃, F) using regression models.
- Classify contamination hotspots using ensemble ML models.
- Visualize results spatially via interactive maps (Folium/QGIS).

---

### Methodology
1. **Data Collection:** CGWB, TNPCB, IMD, WRIS, Satellite indices.
2. **Preprocessing:** Missing value imputation, scaling, geospatial merging.
3. **Feature Engineering:** NDVI, rainfall anomalies, distance to industry.
4. **Model Development:** Random Forest, SVM, Logistic Regression, XGBoost.
5. **Validation:** k-Fold CV, ROC-AUC, RMSE metrics.
6. **Visualization:** Hotspot maps, rainfall–contamination trends, district-wise analysis.

---

### Results
| Model | Accuracy | ROC-AUC | RMSE (EC) | R² (EC) |
|--------|-----------|----------|------------|----------|
| Random Forest | 0.66 | 0.84 | 380.6 | 0.93 |
| Logistic Regression | 0.67 | 0.83 | – | – |
| SVM | 0.61 | 0.79 | – | – |

> **NDVI, EC, and NO₃ were key predictors.**
> High-risk clusters found in coastal and industrial districts (Tuticorin, Nagapattinam, Salem).

---

### Repository Structure
### Repository Structure
data/ – Cleaned and raw datasets
scripts/ – Model training and mapping scripts
results/ – Visual outputs (plots, maps, regression fits)
docs/ – Final report and supporting files

yaml
Copy code

---

### Run Locally
```bash
pip install -r requirements.txt
python scripts/phase5_complete_model.py
python scripts/phase6_mapping.py
```
### Visualization
Hotspot Map: results/hotspot_map_tamilnadu.html
Rainfall–EC Plot: results/rainfall_ec_relation.html
Regression Outputs: results/EC_RandomForest_regression.png

### Future Scope

Automate monthly data ingestion (CGWB + IMD APIs).

Deploy online dashboard using Streamlit.

Integrate rainfall forecasts for proactive contamination alerts.