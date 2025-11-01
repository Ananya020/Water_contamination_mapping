# 🌊 Predictive Water Contamination Mapping – Tamil Nadu

## 🔍 Abstract
This project aims to **predict and map water contamination** across Tamil Nadu using **Machine Learning and geospatial analytics**.  
By integrating **groundwater and surface water data** (pH, EC, NO₃, F, etc.) with **rainfall records, satellite indices (NDVI, NDWI)**, and **industrial proximity**, the system classifies regions as:
- 🟢 *Safe*  
- 🟠 *At-Risk*  
- 🔴 *Hotspot*  

The project demonstrates how **data-driven environmental intelligence** can support **water resource management and public health** initiatives through predictive modelling and spatial visualization.

---

## 🎯 Objectives
- Predict **key water quality parameters** such as EC, Nitrate (NO₃), and Fluoride (F).  
- Classify **contamination hotspots** using supervised learning.  
- Visualize spatial trends through **interactive maps and dashboards**.  
- Evaluate model performance using **RMSE, R², ROC-AUC, and F1-score**.

---

## 🧠 Methodology

### 1️⃣ Data Collection
| Source | Data Type | Description |
|--------|------------|-------------|
| **CGWB / TNPCB Reports** | Groundwater quality | EC, NO₃, F, pH, TDS, hardness |
| **IMD** | Rainfall | Daily & cumulative rainfall data |
| **WRIS / Sentinel-2** | NDVI, NDWI | Vegetation & water spread indices |
| **Industrial Maps (CPCB)** | Distance to Industry | For risk correlation |

### 2️⃣ Preprocessing
- Null value imputation (median strategy).  
- Feature scaling using **StandardScaler**.  
- Coordinate-based geospatial mapping (latitude, longitude).  
- Outlier handling using quantile clipping.  
- Label generation (`hotspot_label` = 0, 1, 2).  

### 3️⃣ Feature Engineering
| Feature | Description |
|----------|-------------|
| NDVI | Normalized Difference Vegetation Index |
| NDWI | Normalized Difference Water Index |
| Distance_to_Industry_km | Estimated industrial proximity |
| rain_actual_24h | Rainfall in the last 24 hours |
| rain_pdn | Rainfall anomaly (actual vs normal) |
| EC, NO₃, F | Key chemical contaminants (targets) |

---

## 🔢 Mathematical Modelling of the Algorithm

### Inputs and Outputs
Let  
- \( X = [x_1, x_2, ..., x_n] \) be input feature vectors (rainfall, NDVI, NDWI, EC, etc.)  
- \( y \) be target contaminant concentration or category label.  

Two ML tasks:
1. **Regression:** Predict contaminant level (EC, NO₃, F).  
   \[
   \hat{y} = f(X) + \epsilon
   \]
2. **Classification:** Predict contamination category (0, 1, 2).  
   \[
   \hat{y} = \text{argmax}(P(y|X))
   \]

### Model Functions
**Random Forest Regressor**
\[
\hat{y} = \frac{1}{T} \sum_{t=1}^T f_t(X)
\]

**Logistic Regression Classifier**
\[
P(y=1|X) = \frac{1}{1 + e^{-(wX + b)}}
\]

### Loss / Objective Functions
- **Regression Loss:**  
  Mean Squared Error (MSE)  
  \[
  L_{MSE} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2
  \]
- **Classification Loss:**  
  Cross-Entropy Loss  
  \[
  L_{CE} = -\frac{1}{n} \sum_i y_i \log(\hat{y}_i)
  \]
  
### Optimization
Models were trained using:
- Gradient-based optimization (for LR/SVM).
- Greedy ensemble averaging (for RandomForest/XGBoost).
- 5-Fold Cross-validation for robustness.

---

## ⚙️ Implementation Details

### Tools & Libraries
| Category | Tools Used |
|-----------|-------------|
| Programming | Python 3.11 |
| Libraries | pandas, numpy, scikit-learn, xgboost, seaborn, plotly, folium, streamlit |
| Visualization | Plotly, Folium, QGIS |
| ML Models | Random Forest, Logistic Regression, SVM, XGBoost |

### Dataset Description
| Column | Meaning |
|---------|----------|
| latitude, longitude | Spatial coordinates |
| EC | Electrical Conductivity (µS/cm) |
| NO3 | Nitrate (mg/L) |
| F | Fluoride (mg/L) |
| NDVI, NDWI | Satellite indices |
| rain_actual_24h, rain_pdn | Rainfall data |
| hotspot_label | Target category (0, 1, 2) |

---

## 📊 Results & Discussion

### 🔹 Classification Results
| Model | Accuracy | F1-Score | ROC-AUC |
|--------|-----------|----------|----------|
| Random Forest | 0.662 | 0.656 | 0.841 |
| Logistic Regression | 0.671 | 0.655 | 0.828 |
| SVM | 0.608 | 0.550 | 0.796 |

### 🔹 Regression Results
| Target | Model | RMSE | MAE | R² |
|---------|--------|------|------|----|
| EC | RandomForest | 380.6 | 192.1 | 0.93 |
| EC | XGBoost | 399.5 | 193.3 | 0.92 |
| NO3 | RandomForest | 50.1 | 34.0 | 0.27 |
| F | RandomForest | 0.43 | 0.33 | 0.21 |

---

## 🗺️ Visualization & Dashboard

**Streamlit Dashboard Features**
- Interactive rainfall vs contamination plots.  
- Folium-based Tamil Nadu hotspot map.  
- Dynamic filters by district/category.  
- Feature importance visualization.  

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Key Insights
- **Random Forest** was the most reliable model for both regression and classification.
- **NDVI, EC, and NO₃** strongly influence contamination predictions.
- Coastal and industrial districts (Tuticorin, Nagapattinam, Salem) show persistent contamination risks.
- Post-monsoon data showed improved water quality due to dilution effects.

---

## 🔮 Future Scope
- Integrate **real-time rainfall and pollution APIs** for live monitoring.
- Add **temporal forecasting models** (LSTM, Prophet).
- Deploy cloud-based dashboard for **public & policy access**.
- Extend model to cover **South Indian river basins**.

---

## 🧾 Repository Structure

```
📁 predictive-water-contamination/
│
├── data/
│   ├── MLPROJECTFINALDATA.csv
│
├── scripts/
│   ├── phase5_model_development.py
│   ├── mapping_visualization.py
│
├── results/
│   ├── hotspot_map_tamilnadu.html
│   ├── rainfall_ec_relation.png
│   ├── EC_RandomForest_regression.png
│
├── app.py
├── README.md
├── requirements.txt
```

---

## 👩‍💻 Author
**Ananya Agrawal**  
