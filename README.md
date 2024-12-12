# Titanic Survival Prediction with AutoML Optimization

This project predicts Titanic survival using PyCaret for classification, Optuna for hyperparameter tuning, and LightGBM as the primary model. The workflow integrates feature engineering, model selection, and advanced optimization techniques to achieve high prediction accuracy.

## Prompt Summary

The project involves:
1. Feature engineering: Dropping unnecessary columns and handling missing values in the Titanic dataset.
2. Model selection: Utilizing PyCaret to automatically compare models and select the best-performing one.
3. Hyperparameter optimization: Using Optuna to fine-tune the selected model's hyperparameters with a custom search space.
4. Visualization: Plotting confusion matrix and feature importance for evaluation.
5. Prediction: Applying the trained model to the test dataset and saving the predictions.

---

## Project Details

### Dataset
- **Training data**: Provided as `train.csv`, contains passenger information with survival labels.
- **Testing data**: Provided as `test.csv`, contains passenger information without survival labels.

### Requirements
- Python 3.10+
- Required libraries:
  - `pandas`
  - `pycaret`
  - `optuna`
  - `lightgbm`
  - `scikit-learn`

### Key Features
1. **Automated Model Selection**: PyCaret identifies and compares multiple models.
2. **Advanced Optimization**: Optuna optimizes hyperparameters for the selected model, enabling fine-grained control.
3. **End-to-End Pipeline**: From preprocessing to prediction, the workflow is fully automated.

---

## Code Structure

- **Data Preprocessing**:
  - Remove irrelevant features like `PassengerId`, `Name`, `Ticket`, and `Cabin`.
  - Fill missing values for `Age`, `Fare`, and `Embarked`.
- **Model Training**:
  - Use PyCaret to compare and tune models.
  - Perform hyperparameter tuning with Optuna's `study.optimize`.
- **Evaluation**:
  - Generate confusion matrix and feature importance plots.
- **Prediction**:
  - Predict survival on the test dataset and save results to `submission.csv`.

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction
   
1. **Install dependencies**:
    pip install -r requirements.txt
2. **Prepare the datasets**:
Place train.csv and test.csv in the root directory.
3. **Run the script**:
python main.py
4. **Check the output**:
Predictions will be saved in submission.csv.

##Visualization
Confusion Matrix
A graphical representation of the model's classification performance.

##Feature Importance
A chart showing the importance of each feature in making predictions.

##Sample Results
PassengerId	Survived
892	0
893	1
894	0

##Future Enhancements
Implement cross-validation for more robust evaluation.
Add more feature engineering techniques (e.g., interaction terms).
Experiment with additional AutoML frameworks like H2O or MLFlow.

##License
This project is licensed under the MIT License. See the LICENSE file for details.

