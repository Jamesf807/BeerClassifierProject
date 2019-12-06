import sklearn

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

input_file = 'beers.csv'

data = pd.read_csv(input_file)

occurences = data['label'].value_counts()[:20]
occurences = occurences.index.tolist()
df2 = data[data.label.isin(occurences)]

df_x = df2[['abv', 'ibu']]
df_y = df2['label']
""
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_x, df_y, test_size=0.25)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))