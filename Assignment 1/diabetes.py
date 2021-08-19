import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def sign(n):
    if n > 0:
        return 1
    else:
        return -1


def perceptron(x, y, eta):
    w = np.zeros((1, np.size(x, 1)))
    # print(validation(x, w, y))
    for T in range(1000):
        yx = np.zeros((1, np.size(x, 1)))
        for t in range(len(x)):
            k = np.dot(w, x[t, :]) * y[t]
            if k <= 0:
                yx += (y[t] * x[t, :])

        w += eta * yx

    return w


def perceptronAlt(x, y, eta):
    w = np.zeros((1, np.size(x, 1)))
    T = 0
    while test(w, x, y)[0] > 0.31 and T < 10000:
        T += 1
        yx = np.zeros((1, np.size(x, 1)))
        for t in range(len(x)):
            k = np.dot(w, x[t, :]) * y[t]
            if k <= 0:
                yx += (y[t] * x[t, :])

        w += eta * yx

    return w, T


def test(w, x, y):
    error = 0
    count = 0
    for t in range(len(x)):
        ystar = sign(np.dot(w, x[t, :]))
        if ystar < 0:
            count += 1
        if ystar != y[t]:
            error += 1

    return error / len(x), count


diabetesData = pd.read_csv("diabetes.csv")
# Get y values
diabetesOutcome = diabetesData["Outcome"]
diabetesOutcome[diabetesOutcome == 0] = -1
# Get x values
diabetesData = diabetesData.drop("Outcome", axis=1)
diabetesData = diabetesData.to_numpy()
diabetesOutcome = diabetesOutcome.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(diabetesData, diabetesOutcome, random_state=1)

etas = [1, 0.01, 0.001, 0.0001]
for eta in etas:
    w, T = perceptronAlt(X_train, y_train, eta)
    print(test(w, X_test, y_test), T)
