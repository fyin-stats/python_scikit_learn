## Preprocessing data
## real world data is not always nice

## Dealing with categorical features
## scikit-learn will not accept categorical features by default
## need to encde categorical features numerically
## convert to dummy variables

## dummy variables
## cresate binary features for each of the origins
## US, Europe, Asia
## no need to use three dummy variables
##

## Dealing with categorical features in Python
## scikit-learn: OneHotEncoder()
## pandas: get_dummies()

## mpg: target variable
## origin: predictors?

##
import pandas as pd

df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies()

#
df_origin = df_origin.drop('origin_Asia', axis=1)

#
print(df_origin.head())

#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#
ridge.score(X_test, y_test)

#
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Drop 'Region_America' from df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

# Regression with Categorical features
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha = 0.5, normalize = True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv = 5)

# Print the cross-validated scores
print(ridge_cv)

# Handling missing data
# no value in a certain feature in a column
# missing values can be coded in a number of ways
# print(df.head())
#

# df.insulin.replace(0, np.nan, inplace = True)
# df.triceps.replace(0, np.nan, inplace = True)
# df.bmi.replace(0, np.nan, inplace = True)
# df.info()

# df = df.dropna()
# df.shape

# impute missing data
# making an educated guess about the missing values
# Example:
# 1, using the mean of the non-missing entries
# 2,
from sklearn.preprocessing import Imputer
imp = Imputer(missing_value = 'NaN', strategy = 'mean', axis = 0)
# 0 means columns, 1 means row (axis)
#
imp.fit(X)
X = imp.transform(X)

# Imputing with a pipeline
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_value = 'NaN', strategy = 'mean', axis = 0)
logreg = LogisticRegression()
steps = [('imputation', imp), ('logistic_regression', logreg)]

#
pipeline = Pipeline(steps)

#
pipeline.fit(X_train, y_train)

# pipeline.predict(X_test)
# pipeline.score()

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

#
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

#
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

# centering and scaling
# Why scale your data?
# range varies widely!

# many models use some form of distance to inform them
# features on larger scales can unduly influence on the model
# Example: kNN uses distance explicitly when making predictions
# we want features to be on a similar scale
# normalizing (or scaling and centering)

# ways to normalize your data
# standardization: subtract the mean and divide by variance
# all features are centered around zero and have variance one
# minimum zero and maximum one
#
from sklearn.preprocessing import scale
X_scaled = scale(X)
np.mean(X), np.std(X)
np.mean(X_scaled), np.std(X_scaled)

# scaling in a pipeline
#
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()), ('knn', KNeighborCalssifier())]

pipeline = Pipeline(steps)
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# CV and scaling in a pipeline
#
# parameters = # as a dictionary
# GridSearchCV
#
# # Import scale
# from sklearn.preprocessing import scale
#
# # Scale the features: X_scaled
# X_scaled = scale(X)
#
# # Print the mean and standard deviation of the unscaled features
# print("Mean of Unscaled Features: {}".format(np.mean(X)))
# print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))
#
# # Print the mean and standard deviation of the scaled features
# print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
# print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

#########################################################################
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
#############################################################################


# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

######################################################
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
