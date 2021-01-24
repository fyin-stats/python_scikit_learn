################
# https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn
# supervised learning with scikit learn
# machine learning: the art and science of giving computers the ability to learn to make decisions from data
# without being explicitly programmed

# learning to predict spam emails
# clustering wikipedia entries into different categories

# Unsupervised learning: uncovering hidden patterns from unlabeled data

# Reinforcement learning: software agents interact with an environment
# learn how to optimize their behavior
# given a system of rewards and punishments
# draws inspiration from behavioral psychology
#

# Applications:
# economics
# genetics
# game playing

# Supervised learning
# predictor variables/features and a target variable
# Aim: Predict the target variable, given the predictor variables
# naming conventions:
# features = predictor variables = independent variables
# target variable = response variable

# automate time-consuming or expensive manual tasks
# make predictions about the future
# need labeled data -- historical data with labels, experiments to get labeled data, crowd sourcing labeled data
#


# scikit-learn / sklearn
# Integrates well with the SciPy stack



# first dataset, the iris dataset
# The iris dataset in scikit-learn
#
#

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()

type(iris)
#
print(iris.keys())

#
type(iris.data)
# type(iris.keys())

#
type(iris.keys())
#
type(iris.data.shape)

# samples are in rows, features are in columns
#
iris.target_names



# EDA
X = iris.data
y = iris.target

# convert numpy array to pandas dataframe
df = pd.DataFrame(X, columns=iris.feature_names)

# viewing the head of the dataframe
print(df.head())

# visual EDA
_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8,8], s = 150, marker = 'D')
# c stands for color

# Numerical EDA
#

# Numerical EDA
#
# In this chapter, you'll be working with a dataset obtained from the UCI Machine Learning Repository consisting of votes made by US House of Representatives Congressmen. Your goal will be to predict their party affiliation ('Democrat' or 'Republican') based on how they voted on certain key issues. Here, it's worth noting that we have preprocessed this dataset to deal with missing values. This is so that your focus can be directed towards understanding how to train and evaluate supervised learning models. Once you have mastered these fundamentals, you will be introduced to preprocessing techniques in Chapter 4 and have the chance to apply them there yourself - including on this very same dataset!
#
# Before thinking about what supervised learning models you can apply to this, however, you need to perform Exploratory data analysis (EDA) in order to understand the structure of the data. For a refresher on the importance of EDA, check out the first two chapters of Statistical Thinking in Python (Part 1).
#
# Get started with your EDA now by exploring this voting records dataset numerically. It has been pre-loaded for you into a DataFrame called df. Use pandas' .head(), .info(), and .describe() methods in the IPython Shell to explore the DataFrame, and select the statement below that is not true.

# describe, info, shape

# Visual EDA
#
# plt.figure()
# sns.countplot(x='education', hue='party', data=df, palette='RdBu')
# plt.xticks([0,1], ['No', 'Yes'])
# plt.show()

# countplot
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# The classification challenge
# K-nearest neighbors
# Basic idea: Predict the label of a data point by
# looking at the k closest labeled data points
# majority vote

# kNN gives decision boundaries
# kNN intuition
# Scikie-learn fit and predict
# Training a model on the data = "fitting" a model to the data
# .fit() method
# To predict the labels of new data, .predict() method


# Using scikit learn to fit a classifier
#
from sklearn.neighbors import KNeighborsClassifier

#
knn = KNeighborsClassifier(n_neighbors=6)

#
knn.fit(iris['data'], iris['target'])

# KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None,
# n_jobs = 1, n_neighbors=6, p=2, weights = 'uniform')

# dealing with categorical features
# and missing data, later

# iris['data'].shape (150, 4)
# iris['target'].shape (150,)

# Prediciting on unlabeled data
X_new = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])

#
prediction = knn.predict(X_new) # features in columns, observations in rows
# should return a three by one array

#

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis = 1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


# Measuring model performance
# in classification, accuracy is a commonly used metric
# Accuracy = Fraction of correct predictions

# Measuring model performance
# could compute accuracy on data used to fit classifier
# not indicative of ability to generalize
# split data into training and test set
# fit train the classifier on the training set
# make predictions on test set

# compare predictions with known labels
# Train/test split

#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=21, stratify=y)
# train test split returns 4 arrays

#
knn.score(X_test, y_test)

# Model complexity and over/under fitting

#

# The digits recognition dataset
#
# Up until now, you have been performing binary classification, since the target variable had two possible outcomes. Hugo, however, got to perform multi-class classification in the videos, where the target variable could take on three possible outcomes. Why does he get to have all the fun?! In the following exercises, you'll be working with the MNIST digits recognition dataset, which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is one of scikit-learn's included datasets, and that is the one we will use in this exercise.
#
# Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black. Recall that scikit-learn's built-in datasets are of type Bunch, which are dictionary-like objects. Helpfully for the MNIST dataset, scikit-learn provides an 'images' key in addition to the 'data' and 'target' keys that you have seen with the Iris data. Because it is a 2D array of the images corresponding to each sample, this 'images' key is useful for visualizing the images, as you'll see in this exercise (for more on plotting 2D arrays, see Chapter 2 of DataCamp's course on Data Visualization with Python). On the other hand, the 'data' key contains the feature array - that is, the images as a flattened array of 64 pixels.
#
# Notice that you can access the keys of these Bunch objects in two different ways: By using the . notation, as in digits.images, or the [] notation, as in digits['images'].
#
# For more on the MNIST data, check out this exercise in Part 1 of DataCamp's Importing Data in Python course. There, the full version of the MNIST dataset is used, in which the images are 28x28. It is a famous dataset in machine learning and computer vision, and frequently used as a benchmark to evaluate the performance of a new model.


#
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#####################################################################
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))


######################################################################

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()