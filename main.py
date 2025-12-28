
# HEART DISEASE PREDICTION USING LOGISTIC REGRESSION


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset

df = pd.read_csv("heart_disease_data (1).csv")

# Viewing basic information
print("Shape of the data:", df.shape)
print(df.head())
print(df.info())


# 2. Check for missing values

print("\nMissing values in each column:")
print(df.isnull().sum())

# (If your dataset had missing values, you would fill them.
# This dataset usually has no missing values.)


# 3. Split the data into features and target

X = df.drop(columns="target", axis=1)
Y = df["target"]


# 4. Train–test split (80% train, 20% test)
# Using stratify to maintain target distribution

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

print("\nTraining and Testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# 5. Create the Logistic Regression model

model = LogisticRegression(max_iter=1000)   # Increasing max_iter to avoid convergence warning


# 6. Training the model

model.fit(X_train, Y_train)


# 7. Predictions and Evaluation


# Training accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(Y_train, train_pred)
print("\nTraining Accuracy:", train_acc)

# Testing accuracy
test_pred = model.predict(X_test)
test_acc = accuracy_score(Y_test, test_pred)
print("Testing Accuracy:", test_acc)

# Detailed performance
print("\nClassification Report:")
print(classification_report(Y_test, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, test_pred))


# 8. Predicting for a new sample input

input_data = (51,1,3,125,213,0,0,125,1,1.4,2,1,2)

# Convert to array
input_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_array)

if prediction[0] == 0:
    print("\nPrediction: Good news — no heart disease.")
else:
    print("\nPrediction: High risk — patient should consult a doctor.")
