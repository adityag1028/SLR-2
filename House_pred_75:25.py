#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:13:07 2024

@author: ayjeeg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.nan)

df = pd.read_csv(r'/Users/ayjeeg/Downloads/All ML assignments/Project-3.House Price prediction/House_data.csv')
df
space = df['sqft_living']
price = df['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# lets split the training and test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# here we have give 75% to training and 25% to test


# fitting SLR into training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the price

pred = regressor.predict(x_test)

# visualizing the training test results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'black')
plt.title('Visuals for training set')
plt.xlabel('Space')
plt.ylabel('price')
plt.show()

# visualizing the test results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'black')
plt.title('Visuals for training set')
plt.xlabel('Space')
plt.ylabel('price')
plt.show()

