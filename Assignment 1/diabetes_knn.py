import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


diabetesData = pd.read_csv("diabetes.csv")
# Get y values
diabetesOutcome = diabetesData["Outcome"]
diabetesOutcome[diabetesOutcome == 0] = -1
# Get x values
diabetesData = diabetesData.drop("Outcome", axis=1)
diabetesData = diabetesData.to_numpy()
diabetesOutcome = diabetesOutcome.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(diabetesData, diabetesOutcome, random_state=1)
neighbours = np.arange(1, 100)
test_accuracy = np.empty(len(neighbours))

for i, k in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_accuracy[i] = 1 - knn.score(X_test, y_test)

plt.plot(neighbours, test_accuracy)
plt.title("$k$ nearest neighbours with diabetes data.\nMinimum error $=$" 
    + str(round(min(test_accuracy), 3)))
plt.xlabel("$k$")
plt.ylabel("Error")
plt.savefig('knn', dpi=600)