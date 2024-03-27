import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the dataset
data = pd.read_csv("dataset.csv")

# Handle missing values
data.dropna(inplace=True)

# Preprocess the data
encoder = LabelEncoder()
data["risk_level"] = encoder.fit_transform(data["risk level"])
data["disease"] = encoder.fit_transform(data["disease"])
data["symptoms"] = encoder.fit_transform(data["symptoms"])  # Use the same encoder for symptoms
data["cures"] = encoder.fit_transform(data["cures"])
data["doctor"] = encoder.fit_transform(data["doctor"])

# Split data into features and target
x = data[['disease', 'cures']]
y = data["symptoms"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize and train the decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = tree_clf.predict(x_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {'categorical_column1': ['A', 'B', 'C', 'A', 'D'],
        'categorical_column2': ['X', 'Y', 'Z', 'Z', 'X']}
df = pd.DataFrame(data)

X = df[['categorical_column1', 'categorical_column2']]

encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(X)

# Converting the sparse matrix to an array and then to a DataFrame
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

print(encoded_df)


