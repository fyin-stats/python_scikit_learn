## How good is your model?
## Accuracy is not always ideal
## For example, Spam classification
## 99% of emails are real, 1% of emails are spam
## Could build a classifier that predicts all emails are real
## 99% accurate
## but horrible at actually classifying spam
## fails at its original purpose
##

## need more nuanced metrics
## Diagnosing classification predictions
## Confusion matrix
## Actual vs predictions

## accuracy = (tp + tn) / (tp + tn + fp + fn)
## Metrics from the confusion matrix
## precision (tp)/(tp+fp)
## recall (tp)/(tp + fn)
## F1 score = 2*precision*recall / (precision + recall)

## true positive rate, also known as sensitivity = tp/(tp+fn)
## true negative rate, also known as specificity = tn/(tn+fp)

## High precision
## not many real emails predicted as spam

## High recall
## predicted most spam emails correctly

##
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# instantiate the classifier
# split the data
# confusion_matrix
# classification_report
# note true labels should be fed into the function as the first argument
#

#
# Import necessary modules
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




# Logistic regression and the ROC curve
# in classification problems
# linear decision boundary
#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

#
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Probability thresholds
# Not specfic to

from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # roc curve based on the predicted probabiites
# and the true labels

#
logreg.predict_proba(X_test)[:,1]

#
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#

# Import
# roc_curve
# from sklearn.metrics.
#
# Using
# the
# logreg
# classifier, which
# has
# been
# fit
# to
# the
# training
# data, compute
# the
# predicted
# probabilities
# of
# the
# labels
# of
# the
# test
# set
# X_test.Save
# the
# result as y_pred_prob.
# Use
# the
# roc_curve()
# function
# with y_test and y_pred_prob and unpack the result into the variables fpr, tpr, and thresholds.
# Plot
# the
# ROC
# curve
# with fpr on the x-axis and tpr on the y-axis.

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#
# This is a correct statement. Notice how a high precision corresponds to a low recall:
# The classifier has a high threshold to ensure the positive predictions it makes are correct,
# which means it may miss some positive labels that have lower probabilities.


# Area under the ROC curve
# extract a metric of interest
# Larger area under the ROC curve = better model
#
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()

# pass the true label and predicted probability to roc_auc_score function
# cross_val_score
# cv_scores

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv =5, scoring = 'roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# Hyperparameter tuning
# Linear regression: Choosing parameters
# Ridge/lasso regression: Choosing alpha
# k-nearest neighbors: Choosing n_neighbors
# parameters like alpha and k: hyperparameters
# Hyperparameters cannot be learned by fitting the model

# Choosing the correct hyperparameter
# Grid search cross validation
# class gridsearchcv

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,50)} # need to be stored in a dictionary

#
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)

#
knn_cv.fit(X, y)

#
knn_cv.best_params_

# randomized search CV
# jump around grid

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# Hold-out set for final evaluation
# Hold-out set reasoning
# How well can the model perform on a dataset that has never been exposed to the model?
# How it performs on never before seen data?

#
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
