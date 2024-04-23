from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Load the breast cancer dataset from scikit-learn
cancer_data = datasets.load_breast_cancer()

# Display the features of the fifth sample
print(cancer_data.data[4])

# Display the shape of the data
print(cancer_data.data.shape)

# Display the target values
print(cancer_data.target)

# Display information about the dataset
print(cancer_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4, random_state=109)

# Create an SVM classifier with a linear kernel
cls = svm.SVC(kernel='linear')

# Train the model
cls.fit(X_train, y_train)

# Make predictions on the test set
pred = cls.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred=pred))
print("Precision:", metrics.precision_score(y_test, y_pred=pred))
print("Recall:", metrics.recall_score(y_test, y_pred=pred))
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred=pred))
