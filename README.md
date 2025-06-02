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
- **Final Dataset:** Ensuring the dataset is clean and ready for machine learning tasks.

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

This notebook is a practical guide for beginners to understand and apply data preprocessing in real-world projects.