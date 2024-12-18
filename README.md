# Titanic Kaggle Competition

This repository contains my solution for the Kaggle Titanic Machine Learning competition. The notebook explores the dataset, performs exploratory data analysis (EDA), and builds a machine learning model to predict passenger survival based on various features.

## Repository Structure

```
.
├── titanic-eda-and-prediction.ipynb  # Jupyter notebook containing the solution
└── README.md                          # Project documentation
```

## Objective
The goal of this challenge is to build a predictive model that answers the question: "What sorts of people were more likely to survive?" using passenger data (e.g., name, age, gender, socio-economic class).

## Dataset

The dataset for this challenge can be found on the [Kaggle Titanic competition page](https://www.kaggle.com/competitions/titanic/overview). It includes:

- **Train.csv**: Contains the training data to build the model, including passenger details and whether they survived.
- **Test.csv**: Contains passenger details without the survival information, used for predictions.

## Steps Implemented

### 1. Importing Libraries
Key libraries used in the project include:
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for visualization
- `sklearn` for machine learning

### 2. Exploratory Data Analysis (EDA)
- Data cleaning and handling missing values
- Visualizing relationships between survival and key features like:
  - Age
  - Gender
  - Passenger class (Pclass)
  - Fare
- Correlation heatmaps and feature analysis

### 3. Feature Engineering
- Imputation of missing values (e.g., Age and Embarked)
- Creating new features:
  - Title extraction from passenger names
  - Family size
  - IsAlone (whether a passenger is traveling alone)
- Encoding categorical features (e.g., Gender and Embarked)

### 4. Model Building and Evaluation
- Splitting the training data into training and validation sets.
- Training models including:
  - Logistic Regression
  - SGDClassifier
  - MLP ( Multi layer perceptron)
- Evaluation metrics:
  - Accuracy

### 5. Prediction
- Applying the trained model to the test dataset.
- Generating and saving submission files for Kaggle.

## Key Insights
- Gender (female) and class (1st class passengers) had significant positive impacts on survival rates.
- Age and family relationships (e.g., traveling with family) also influenced survival likelihood.
- Feature engineering improved model accuracy significantly.

## Results
The final model achieved score 0.74641 and provided insights into key survival factors.

## Getting Started
To explore the project or run the notebook:

1. Clone the repository:
   ```bash
   git clone https://github.com/raniakh/titanic-kaggle.git
   ```
2. Install required Python libraries:

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook titanic-eda-and-prediction.ipynb
   ```

## Dependencies
- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Contributing
Feel free to fork the repository and submit pull requests for improvements or new ideas!

## License
This project is licensed under the MIT License.

