https://colab.research.google.com/drive/1ijmq_mTEY92hXkrMWlkfyNrAenFSPPoD?authuser=1#scrollTo=54pgoBbehBwA

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Training data
X = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change"]
]
y = ["Yes", "Yes", "No", "Yes"]

# Convert string features to numbers using Label Encoding
from sklearn.preprocessing import LabelEncoder

encoders = []
X_encoded = []
for col in zip(*X):
    X_encoded.append(LabelEncoder().fit_transform(col))

X_encoded = list(zip(*X_encoded))

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Visualize the tree
tree.plot_tree(clf, feature_names=["Sky",



def consistent(hypo, instance):
    return all(h == '?' or h == val for h, val in zip(hypo, instance))

def candidate_elimination(data):
    num_features = len(data[0]) - 1

    S = data[0][:-1]  # Same as Find-S start
    G = [["?"] * num_features]  # Most general hypothesis

    for row in data:
        features, label = row[:-1], row[-1]

        if label == "Yes":
            # Update S like Find-S
            for i in range(num_features):
                if S[i] != features[i]:
                    S[i] = '?'

            # Prune G to keep only consistent hypotheses
            G = [g for g in G if consistent(g, features)]

        else:  # Negative example
            G_new = []
            for g in G:
                if consistent(g, features):
                    for i in range(num_features):
                        if g[i] == '?':
                            if S[i] != '?':
                                new_g = g.copy()
                                new_g[i] = S[i]
                                if not consistent(new_g, features):
                                    G_new.append(new_g)
                        elif g[i] != features[i]:
                            new_g = g.copy()
                            new_g[i] = '?'
                            if not consistent(new_g, features):
                                G_new.append(new_g)
                else:
                    G_new.append(g)
            G = G_new

    return S, G

# Example use
S_final, G_final = candidate_elimination(data)
print("Final Specific Hypothesis (S):", S_final)
print("Final General Hypotheses (G):", 



def find_s(data):
    S = data[0][:-1]  # Start with first positive example

    for row in data:
        features, label = row[:-1], row[-1]
        if label == "Yes":
            for i in range(len(S)):
                if S[i] != features[i]:
                    S[i] = "?"

    return S
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load your CSV file
data = pd.read_csv('your_data.csv')  # replace with your actual file name

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode string features in X
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Encode y if it contains strings
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV
data = pd.read_csv('your_data.csv')  # replace with your actual file name

# Separate features and target
X = data.iloc[:, :-1]   # all columns except last one
y = data.iloc[:, -1]    # last column is target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Example use
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change", "Yes"]
]

print("Find-S Hypothesis:", find_s(data))
