
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

iris_data = pd.read_csv('iris.csv')


print(iris_data.head())


X = iris_data.drop('species', axis=1) 
y = iris_data['species']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_classifier = SVC(kernel='linear', random_state=42)

svm_classifier.fit(X_train, y_train)


predictions = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")


print(classification_report(y_test, predictions))
