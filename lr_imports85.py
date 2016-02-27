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

print df
df = df.dropna()
print df