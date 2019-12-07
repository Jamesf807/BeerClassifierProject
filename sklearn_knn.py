import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

input_file = 'beers.csv'

data = pd.read_csv(input_file)

occurences = data['label'].value_counts()[:10]
occurences = occurences.index.tolist()
df2 = data[data.label.isin(occurences)]

df_x = df2[['abv', 'ibu']]
df_y = df2['label']

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, )

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)

print(KNN)

y_expect = y_test
y_pred = KNN.predict(X_test)

print(metrics.classification_report(y_expect, y_pred))