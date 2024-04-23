from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression before PCA
logreg_before_pca = LogisticRegression()
logreg_before_pca.fit(X_train_scaled, y_train)

# Evaluate accuracy before PCA
y_pred_before_pca = logreg_before_pca.predict(X_test_scaled)
accuracy_before_pca = accuracy_score(y_test, y_pred_before_pca)
print("Accuracy before PCA:", accuracy_before_pca)

# Perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit logistic regression after PCA
logreg_after_pca = LogisticRegression()
logreg_after_pca.fit(X_train_pca, y_train)

# Evaluate accuracy after PCA
y_pred_after_pca = logreg_after_pca.predict(X_test_pca)
accuracy_after_pca = accuracy_score(y_test, y_pred_after_pca)
print("Accuracy after PCA:", accuracy_after_pca)