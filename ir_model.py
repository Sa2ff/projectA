import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels (species)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data=np.c_[X, y], columns=iris.feature_names + ['species'])

# Prepare data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Visualize feature importance
importance = model.feature_importances_
plt.barh(iris.feature_names, importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Iris Dataset')
plt.show()