import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()

df = pd.read_csv('imports-85.data',
                  header=None,
                  names=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
                         'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
                         'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
                         'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'],
                      na_values='?')

# Drop every piece of data that contains NaN
df = df.dropna()

# Split 80% of data for training set
split_idx = int((len(df) * 0.8))
train = df[:split_idx]
test = df[split_idx:]

feature_train = train['engine-size']
target_train = train['price']
feature_test = test['engine-size']
target_test = test['price']
 
reg = linear_model.LinearRegression()
reg.fit(feature_train.reshape(-1,1), target_train.reshape(-1,1))

score = reg.score(feature_test.reshape(-1, 1), target_test.reshape(-1,1))
print 'The score is ', score

### draw the scatterplot, with color-coded training and testing points
train_color = "b"
test_color = "r"
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test, target_test, color=test_color, label="test")
plt.scatter(feature_train, target_train, color=train_color, label="train")


### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test.reshape(-1,1)) )
except NameError:
    pass

plt.title('Linear regression on clean data')
plt.xlabel('Engine-size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Predict result for 175
pred_result = reg.predict(175)
print 'Price prediction for engine size equals to 175 is: ', pred_result[0][0]