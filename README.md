# Sonar_Rock-Vs-Mine-Prediction
This project is a binary classification task that predicts whether a sonar signal bounces off a rock or a mine. We use the classic Sonar dataset, which consists of 208 samples of sonar signals. The goal is to train a machine learning model that can accurately classify whether the object is a rock or a mine based on the sonar signal.

## Getting Started

### Prerequisites

To run this project, you will need Python 3 and the following libraries:

* NumPy
* Pandas
* Scikit-learn
* Matplotlib

### Installing

To install the required libraries, run the following command:

```
pip install numpy pandas scikit-learn matplotlib 
```

## Methodology

We start by loading the dataset using Pandas and separating the features and labels. We then split the dataset into training and testing sets using scikit-learn's train_test_split function.

Next, we create an instance of the LinearRegression class from scikit-learn and fit it to the training data. We use the fitted model to make predictions on the testing data.

Finally, we evaluate the performance of the model using the accuracy score and confusion matrix. We save the model to a file using the joblib library so that we can reuse it later.

## Results
We achieved an accuracy score of 0.81 on the testing data, which indicates that our model is reasonably accurate. The confusion matrix shows that the model is better at classifying rocks than mines, with a higher precision and recall for the rock class.

## Conclusion
In this project, we demonstrated how to use linear regression for binary classification tasks using the Sonar dataset. We showed how to load and preprocess the data, train a machine learning model, and evaluate its performance. This project can serve as a starting point for more complex classification tasks and provides an example of how to use scikit-learn for machine learning tasks.

## Acknowledgments
* The Sonar dataset was obtained from the UCI Machine Learning Repository.
* The code for this project was adapted from the scikit-learn documentation.



