from buildTree import *

# Read in Beer Dataset
df = pd.read_csv("beers.csv")

# Dataset of only top 5 most frequent labels
occurences = df['label'].value_counts()[:5]
occurences = occurences.index.tolist()
df2 = df[df.label.isin(occurences)]

train_df, test_df = train_test_split(df2, 0.25)
train_data = train_df.values
tree = build_tree(train_data, train_df)

df3 = pd.read_csv("demoBeers.csv")

def demoRun(beer):
    print(classify_single(df3.iloc[beer], tree))

demoRun(4);