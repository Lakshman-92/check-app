import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
diabetes_dataset = pd.read_csv("D:\\Downloads\\diabetes.csv")

# Separate the features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create the SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Fit the classifier on the training data
classifier.fit(X_train, Y_train)  # Ensure this line runs without errors

# Evaluate the classifier on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data: ', training_data_accuracy)

# Evaluate the classifier on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data: ', test_data_accuracy)

# Save the trained model to a file
filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Function to make predictions
def predict_diabetes(input_data, model):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Example prediction
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
prediction = predict_diabetes(input_data, loaded_model)

if prediction == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
