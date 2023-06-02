import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = None
y = None


def fetchdata_from_file():
    global X, y
    studyData = pd.read_csv('study_score.csv')

    X = studyData['Hours'].values
    X = X.reshape(-1, 1)
    y = studyData['Scores'].values


def plot_scatter_and_regression_line():
    global model
    plt.scatter(X, y)

    model = LinearRegression()
    model.fit(X, y)

    line = model.coef_ * X + model.intercept_
    plt.plot(X, line, 'black')
    plt.scatter(X, y)
    plt.title("students' scores")
    plt.xlabel("Hour")
    plt.ylabel("Score")


def user_predict_value():
    predict_value = model.predict([[9]])
    print(predict_value[0])


if __name__ == '__main__':
    fetchdata_from_file()
    plot_scatter_and_regression_line()
    user_predict_value()
