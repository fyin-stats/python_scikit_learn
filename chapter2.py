#### chapter 2
#### Introduction to regression


#### boston housing data
####
boston = pd.read_csv('boston.csv')

####
print(boston.head())


#### creating feature and target arrays
####

X = boston.drop('MEDV', axis = 1).values # drop the target
y = boston['MEDV'].values # keep only the target

#
X_rooms = X[:,5]

#
type(X_rooms), type(y)

# reshape
y = y.reshape(-1,1) # -1 means keep the shape of the corresponding axis

#
X_rooms = X_rooms.reshape(-1, 1)

#
plt.scatter(X_rooms, y)
plt.ylabel('Value of house / 1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

# fitting a regression model
import numpy as np
from sklearn.linear_model import LinearRegression

#
reg = LinearRegression()
reg.fit(X_rooms, y)

#
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

#
plt.scatter(X_rooms, y, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space), color = 'black', linewidth = 3)
plt.show()

# importing data for supervised learning
#

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv("gapminder.csv")

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))


# Exploring the Gapminder data
# constructing a heatmap
# df
# df.head(), df.info(), df.describe()

# df.corr()

# seaborn's heatmap
#
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# Regression mechanics
# y = ax + b
# how do we choose a and b
# define an error functions for any given line
# the loss function
# when you call fit, it does minimizing squared loss under the hood
#

# Linear regression on all features
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

#
reg_all = LinearRegression()

#
reg_all.fit(X_train, y_train)

#
y_pred = reg_all.predict(X_test)

#
reg_all.score(X_test, y_test)

# can add regularization

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt( mean_squared_error(y_test, y_pred)  )
print("Root Mean Squared Error: {}".format(rmse))


# Cross validation
# Model performance is dependent on way the data is split
# Not representative of the model's ability to generakuze
# Solution: Cross-validation

# 5-fold cross validation
# more folds = more computationally expensive
# Cross validation in scikit-learn

#
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv = 5)

#
print(cv_results)

#
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv = 5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# compare 10-fold CV against 3-fold CV
# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv = 3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv = 10)
print(np.mean(cvscores_10))


# Regularized regression
# Why regularize
# Linear regression minimizes a loss function
# It chooses a coefficient for each feature variable
# large coefficients can lead to overfitting
# Ridge regression: add a L2 penalty

# Alpha: parameter we need to choose
# Picking alpha here is similar to picking k in k-NN
# Hyperparameter tuning
# Alpha controls model complexity
# Alpha = 0, we get back OLS
# Very high alpha: can lead to underfitting

#
from sklearn.linear_model import Ridge
X_trai, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# LASSO regression
# Loss function
from sklearn.linear_model import Lasso
# call function Lasso, alpha, normalize
# lasso.fit
# lasso_pred
# lasso.score


# Lasso regression for feature selection
# shrinks the coefficients of less important features to exactly zero
#
from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis = 1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

# plotting
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation = 60)
_ = plt.ylabel("Coefficients")
plt.show()

#
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha = 0.4, normalize = True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

#
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


#
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)