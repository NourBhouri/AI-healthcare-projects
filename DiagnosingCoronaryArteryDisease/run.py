# Data Science Tools 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Machine Learning Tools
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Deep Learning Tools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam 

# Import the heart disease dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The names will be the names of each column in our pandas DataFrame
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "class"]

# Read the CSV data into a pandas DataFrame
cleveland_data = pd.read_csv(url, names=names)

# Data Cleaning
# Replace missing data ("?") with NaN and then drop rows with missing values
data = cleveland_data.replace('?', np.nan)
data = data.dropna()

# Convert the necessary columns to numeric (specifically "ca" and "thal")
data["ca"] = pd.to_numeric(data["ca"])
data["thal"] = pd.to_numeric(data["thal"])

# Data Normalization (Standardize features)
X = data.drop(["class"], axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ensure target variable is integer (classification)
y = data["class"].values.astype(int)

# Handle classification into 5 categories (as expected for this dataset)
Y = to_categorical(y, num_classes=5)

# Split data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

# Build a Deep Learning Model

# Define a function to create the Keras model
def create_model():
    # Create the model
    model = Sequential()
    
    # Input layer with 13 input dimensions (the number of features) and first hidden layer
    model.add(Dense(8, input_dim=13, kernel_initializer="normal", activation="relu"))
    
    # Second hidden layer
    model.add(Dense(4, kernel_initializer="normal", activation="relu"))
    
    # Output layer with 5 neurons for 5 classes and softmax activation
    model.add(Dense(5, activation="softmax"))

    # Compile the model
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

# Create the model
model = create_model()

# Uncomment to check the model summary
print(model.summary())

# Fit the model to the training data
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
print("Model fitted") 

# generate classification report using predictions for categorical model
categorical_pred = np.argmax(model.predict(X_test), axis=1)

# Convert y_test back to its original form (from one-hot encoding)
y_test_labels = np.argmax(y_test, axis=1)

print('Results for Categorical Model')
print("Accuracy Score:", accuracy_score(y_test_labels, categorical_pred))
print(classification_report(y_test_labels, categorical_pred))
