from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)

print(KNN.score(X_train, y_train))