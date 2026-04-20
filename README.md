# 🚢 Titanic Survival Prediction

## 👥 Team Member Details
BASONE AYUSH BHANUPRATAP	          : 2503031240138 [TEAM LEADER]
SINGOTHU DHANA	                    : 2503031240110
SOUMYA PRAKASH MUDASI	              : 2503031240113
THAKARE GAURAV SANJAY	              : 2503031240117
MAKIREDDY DINESH REDDY	            : 2503031240128
CHIDANAND SHANKAR PARASHETTI	      : 2503031240139
HIMARAPURA KUMARAPPA GARI MANIKANTA :	2503031240142

---

## 📌 Problem Statement
The objective of this project is to predict whether a passenger survived the Titanic disaster based on individual attributes such as age, gender, ticket class, fare, and family details.  
The goal is to identify key factors affecting survival and build a machine learning model to predict outcomes accurately.

---

## 📊 Dataset Description
The dataset used in this project contains the following features:

| Feature     | Description |
|------------|------------|
| PassengerId | Unique ID of passenger |
| Pclass      | Ticket class (1st, 2nd, 3rd) |
| Name        | Passenger name |
| Sex         | Gender (male/female) |
| Age         | Age of passenger |
| SibSp       | Number of siblings/spouses aboard |
| Parch       | Number of parents/children aboard |
| Ticket      | Ticket number |
| Fare        | Ticket fare |
| Cabin       | Cabin number |
| Embarked    | Port of embarkation |
| Survived    | Survival (0 = No, 1 = Yes) |

The dataset may contain missing values which are handled during preprocessing.

---

## 🧹 Data Preprocessing Steps
- Checked dataset for missing/null values  
- Filled missing values:
  - Age → mean/median  
  - Embarked → mode  
- Dropped irrelevant columns (Name, Ticket, Cabin if needed)  
- Converted categorical variables:
  - Sex → 0/1  
  - Embarked → encoding  
- Split dataset into training and testing sets  
- Applied feature scaling (if required)  

---

## 🤖 Model Used and Training Details
- **Model Used:** Logistic Regression  
- **Library:** Scikit-learn  
- Dataset split:
  - Training set (80%)  
  - Testing set (20%)  
- Model trained on training data and predictions made on test data  

---

## 📈 Model Evaluation Results
- The model performance was evaluated using:
  - Accuracy Score  
  - Confusion Matrix  
  - Precision & Recall  

- Key Observations:
  - Females had higher survival rates  
  - Passengers in higher classes survived more  
  - Fare and age influenced survival probability  

---

## 🔗 GitHub Collaboration Summary
- Repository created and managed on GitHub  
- Code and dataset uploaded  
- Version control maintained using commits  
- README file added for documentation  
- Project structured for easy understanding  

---

## 📌 Conclusion
This project demonstrates how machine learning can be used to predict survival outcomes.  
Logistic Regression provides a strong baseline model.  
Further improvements can be achieved using advanced models like Random Forest or XGBoost.
