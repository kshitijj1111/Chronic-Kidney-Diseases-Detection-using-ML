# Chronic Kidney Disease (CKD) Detection using Machine Learning

This project focuses on developing a machine learning model to detect Chronic Kidney Disease (CKD) based on various patient parameters. The project utilizes several classification algorithms, including Random Forest, Naive Bayes, Decision Tree, and XGBoost, and then combines them into a robust Voting Classifier for improved prediction accuracy.

## Table of Contents

- [Chronic Kidney Disease (CKD) Detection using Machine Learning](#chronic-kidney-disease-ckd-detection-using-machine-learning)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Models Used](#models-used)
  - [Ensemble Model (Voting Classifier)](#ensemble-model-voting-classifier)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Project Overview

Chronic Kidney Disease (CKD) is a significant global health problem. Early detection is crucial for effective management and preventing progression. This project aims to provide a reliable and accurate prediction system for CKD using supervised machine learning techniques.

The workflow involves:
1.  **Data Preprocessing:** Cleaning, handling missing values, and transforming raw data.
2.  **Model Training:** Training individual classification models.
3.  **Ensemble Learning:** Combining the strengths of multiple models using a Voting Classifier.
4.  **Evaluation:** Assessing the performance of the models using various metrics.

## Dataset

The dataset used for this project is `kidney_disease(1).csv`, which is preprocessed into `cleaned_ckd_data.xlsx`. This dataset contains various attributes related to patient health, including:

* `age`
* `bp` (blood pressure)
* `sg` (specific gravity)
* `al` (albumin)
* `su` (sugar)
* `rbc` (red blood cells)
* `pc` (pus cell)
* `pcc` (pus cell clumps)
* `ba` (bacteria)
* `bgr` (blood glucose random)
* `bu` (blood urea)
* `sc` (serum creatinine)
* `sod` (sodium)
* `pot` (potassium)
* `hemo` (hemoglobin)
* `pcv` (packed cell volume)
* `wc` (white blood cell count)
* `rc` (red blood cell count)
* `htn` (hypertension)
* `dm` (diabetes mellitus)
* `cad` (coronary artery disease)
* `appet` (appetite)
* `pe` (pedal edema)
* `ane` (anemia)
* `classification` (target variable: `ckd` or `notckd`)

## Models Used

Individual machine learning models were trained and evaluated:

* **Random Forest Classifier:** (`test1.py`) An ensemble learning method for classification that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
* **XGBoost Classifier:** (`test2.py`) An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.
* **Naive Bayes Classifier (GaussianNB):** (`test5.py`) A probabilistic machine learning model based on the Bayes' theorem with a "naive" independence assumption between the features.
* **Decision Tree Classifier:** (`test6.py`) A non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Ensemble Model (Voting Classifier)

The `ensembling.py` script combines the predictions of the individual models using a `VotingClassifier` with `soft` voting. This approach leverages the strengths of each model, aiming for higher overall accuracy and robustness.

## Project Structure


.
├── cleaned_ckd_data.xlsx - Sheet1.csv
├── kidney_disease(1).csv
├── processing.py
├── test1.py
├── test2.py
├── test5.py
├── test6.py
├── ensembling.py
├── models/
│   ├── random_forest_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── decision_tree_model.pkl
│   ├── xgboost_model.pkl
│   └── voting_classifier_model.pkl
└── README.md


## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ckd-detection-ml.git](https://github.com/your-username/ckd-detection-ml.git)
    cd ckd-detection-ml
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn xgboost openpyxl
    ```

## Usage

Follow these steps to preprocess the data, train the individual models, and then train and evaluate the ensemble model:

1.  **Data Preprocessing:**
    Run the `processing.py` script to clean and preprocess the raw dataset. This will generate `cleaned_ckd_data.xlsx`.
    ```bash
    python processing.py
    ```

2.  **Train Individual Models:**
    Run each of the individual model training scripts. These scripts will save the trained models as `.pkl` files in the `models/` directory.
    ```bash
    python test1.py # Trains Random Forest and saves random_forest_model.pkl
    python test2.py # Trains XGBoost and saves xgboost_model.pkl
    python test5.py # Trains Naive Bayes and saves naive_bayes_model.pkl
    python test6.py # Trains Decision Tree and saves decision_tree_model.pkl
    ```
    *Note: Ensure you have a `models` directory created in your project root before running these scripts, or the scripts will create it.*

3.  **Train and Evaluate Ensemble Model:**
    Run the `ensembling.py` script. This script loads the individual models, trains the Voting Classifier, evaluates its performance, and saves the ensemble model as `voting_classifier_model.pkl`.
    ```bash
    python ensembling.py
    ```

## Results

The output of the individual model scripts (`test1.py`, `test2.py`, `test5.py`, `test6.py`) and the ensemble script (`ensembling.py`) will display the accuracy, confusion matrix, and classification report for each model. The Voting Classifier is expected to show improved performance due to its ensemble nature.

Example output from `ensembling.py`:


✅ Voting Classifier Accuracy: 98.75%

Confusion Matrix:
[[50  0]
[ 1 29]]

Classification Report:
precision    recall  f1-score   support

       0       0.98      1.00      0.99        50
       1       1.00      0.97      0.98        30

accuracy                           0.99        80

macro avg       0.99      0.98      0.99        80
weighted avg       0.99      0.99      0.99        80


*(Note: The exact accuracy and metrics may vary slightly based on the random state and data splits.)*

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [Your Name/Email/LinkedIn Profile].
