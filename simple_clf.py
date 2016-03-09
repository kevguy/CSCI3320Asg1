import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 5000

centers = [(-2, -2), (2, 2)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
log_reg = linear_model.LogisticRegression()


# Train the model using the training sets
log_reg.fit(X_train, y_train)

# Prediction
pred = log_reg.predict(X_test)
print 'prediction', pred

### draw the scatterplot, with color-coded training and testing points
class_0_color = "b"
class_1_color = "r"
import matplotlib.pyplot as plt
for test, prediction in zip(X_test, pred):
	if (prediction == 0):
		plt.scatter( test[0], test[1], color=class_0_color ) 
	else:
		plt.scatter( test[0], test[1], color=class_1_color ) 

plt.title('Logistic Regression Prediction')
plt.show()

correct = 0
count = 0
for y_test, prediction in zip(y_test, pred):
	if (prediction == y_test):
		correct = correct + 1
	count = count + 1

print 'Number of sammples in testing set: ', count
print 'Number of correct predictions: ', correct
print 'Number of wrong predictions: ', count - correct