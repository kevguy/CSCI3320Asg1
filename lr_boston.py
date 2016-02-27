import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn; seaborn.set()

# Load the diabetes dataset
boston = datasets.load_boston()
# structure of boston: data, feature_names, DESCR and target
print 'Number of features in the Boston dataset is: ', len(boston['feature_names'])
print 'Number of samples in the Boston dataset is: ', len(boston['data'])
### The following line gives the same answer
### print 'Number of samples in the Boston dataset is: ', len(boston['target'])

best_score = -1000
for i_feature in range(0, len(boston['feature_names'])):
	# Get the feature name
	feature_name = boston.feature_names[i_feature]
	print 'Feature name is', feature_name

	# Use only one feature
	diabetes_X = boston.data[:, np.newaxis, i_feature]

	# Split the data into training/testing sets
	boston_X_train = diabetes_X[:-20]
	boston_X_test = diabetes_X[-20:]

	# Split the targets into training/testing sets
	boston_y_train = boston.target[:-20]
	boston_y_test = boston.target[-20:]

	# Create linear regression object
	model = linear_model.LinearRegression()

	# Train the model using the training sets
	model.fit(boston_X_train, boston_y_train)

	# Explained variance score: score=1 is perfect prediction
	model_score = model.score(boston_X_test, boston_y_test)
	print 'The score is ', model_score, '\n'

	if model_score > best_score:
		best_score = model_score
		best_feature = feature_name
		best_idx = i_feature

print 'Best fitted feature name is: ', best_feature
print 'Best fitted model score is: ', best_score

### Train using the best feature
# Use only one feature
diabetes_X = boston.data[:, np.newaxis, best_idx]

# Split the data into training/testing sets
boston_X_train = diabetes_X[:-20]
boston_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(boston_X_train, boston_y_train)

# Prediction
pred = model.predict(boston_X_test)

# Calculate Loss Function
sub = pred - boston_y_test
sub_square = np.square(sub)
sub_square_sum = np.sum(sub_square)
loss = sub_square_sum / len(sub)
print 'Value of the loss function', loss