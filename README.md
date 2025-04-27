
# **California Housing Dataset Implementation**  
**End-to-End Machine Learning Pipeline**  

## **Project Overview**  
This project implements a complete **machine learning pipeline** on the California Housing dataset, covering:  
- Data exploration & visualization  
- Stratified sampling  
- Custom transformers & feature engineering  
- Hyperparameter tuning with `GridSearchCV`  
- Model evaluation (Linear Regression, Decision Trees, Random Forests)  

**Key Skills Demonstrated**:  
✔ Data preprocessing with `ColumnTransformer`  
✔ Handling categorical/numerical features  
✔ Cross-validation & performance metrics (RMSE)  

---

## **Dataset**  
**Source**: [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | [Original Book Repository](https://github.com/ageron/handson-ml2)  
**Features**:  
- `median_house_value` (target)  
- `median_income`, `housing_median_age`, `population`, etc.  
- `ocean_proximity` (categorical)  

---

## **Code Structure**  
```bash
california_housing/
├── california_housing.ipynb  # Main Colab notebook
├── datasets/
│   └── housing/              # Auto-downloaded data
│       └── housing.csv
└── README.md
```

---


## **Key Code Snippets**  
### **1. Data Pipeline**  
```python
# Custom transformer to add rooms/household
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        return np.c_[X, rooms_per_household]

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), ["ocean_proximity"]),
])
```

### **2. Hyperparameter Tuning**  
```python
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
]
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=5, scoring='neg_mean_squared_error'
)
grid_search.fit(housing_prepared, housing_labels)
```

---

## **Results**  
| Model               | RMSE (Train) | RMSE (Test) |
|---------------------|-------------|-------------|
| Linear Regression   | 68,628      | 68,134      |
| Random Forest       | 18,662      | 47,209      |
| **Tuned Random Forest** | **17,982** | **46,850**  |


---


