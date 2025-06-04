# Preprocessing_Practice

This project contains a Jupyter Notebook that demonstrates essential data preprocessing techniques on a loan dataset.

## Main Steps Covered

- **Loading Data:** Importing the dataset using pandas.
- **Exploring Data:** Checking the shape, info, and previewing the data.
- **Handling Missing Values:** 
  - Identifying missing values.
  - Filling missing categorical values with mode.
  - Filling missing numerical values with mean (using pandas and scikit-learn).
- **Dropping Columns:** Removing unnecessary columns such as `Credit_History`.
- **Data Visualization:** Using seaborn heatmaps to visualize missing data.
- **Encoding Categorical Variables:** 
  - Applying one-hot encoding to categorical columns.
  - Renaming encoded columns for clarity.
  - Dropping original categorical columns and merging encoded columns back.
- **Label Encoding:** 
  - Using scikit-learn's `LabelEncoder` to convert categorical columns (like `Property_Area`) into numeric codes.
  - Adding the encoded values as new columns without dropping the originals.
  - Useful for both ordinal and nominal data.
- **Feature Scaling:**  
  - **Standardization (Z-score Scaling):**  
    Standardization rescales features so that they have the properties of a standard normal distribution (mean = 0 and standard deviation = 1). This is useful when your data has outliers or is not on the same scale.  
    In the notebook, `StandardScaler` from scikit-learn is used to standardize numerical columns such as `LoanAmount`.
  - **Normalization (Min-Max Scaling):**  
    Normalization rescales the values into a range of [0, 1]. This is helpful when you want all features to have the same scale, especially for algorithms that are sensitive to the scale of data.  
    In the notebook, `MinMaxScaler` from scikit-learn is used to normalize columns like `CoapplicantIncome`.
- **Final Dataset:** Ensuring the dataset is clean and ready for machine learning tasks.

## Example Code Snippets

**One-Hot Encoding:**
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(dataset[['Gender', 'Married', 'Education','Self_Employed','Loan_Status']]).toarray()
encoded_features = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
# Rename columns if needed
encoded_features.rename(columns={
    'Gender_1.0': 'Gender',
    'Married_1.0': 'Married',
    'Education_1.0': 'Education',
    'Self_Employed_1.0': 'Self_Employed',
    'Loan_Status_1.0': 'Loan_Status'
}, inplace=True)
# Drop original columns and merge
columns_to_drop = [col for col in ['Gender', 'Married', 'Education','Self_Employed','Loan_Status'] if col in dataset.columns]
dataset = dataset.drop(columns=columns_to_drop, axis=1)
dataset = pd.concat([dataset, encoded_features], axis=1)
```

**Label Encoding:**
```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelencoder.fit(dataset['Property_Area'])
dataset['Property_Area_Encoded'] = labelencoder.transform(dataset['Property_Area'])
```

**Standardization (Z-score Scaling):**
- Scales features to have mean 0 and standard deviation 1.
- Useful for algorithms that assume data is normally distributed.

**Normalization (Min-Max Scaling):**
- Scales features to a fixed range, usually [0, 1].
- Useful when you want all features to have the same scale.

## Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- Jupyter Notebook

## Usage

1. Open the notebook in Jupyter.
2. Run each cell step by step to see the preprocessing workflow.
3. Modify the code as needed for your own datasets.

---

This notebook is a practical guide for beginners to understand and apply data preprocessing (missing value handling, encoding, scaling, visualization) in real-world projects.