import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_file = pd.read_csv('train.csv')
test_file  = pd.read_csv('test.csv')

# ? Using these particular set of features has the benefit that
# ? they don't contain null values in train/test set.
feature_list = ['Pclass', 'Sex', 'SibSp', 'Parch']

X = train_file[feature_list]
X_test = test_file[feature_list]

# ? Since the data set contains strings as well, get_dummies will encode these into
# ? numeric data types.
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

y = train_file.Survived

# ? First identify appropriate parameters using split,
# ? Then train on the entire training set. 
# train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

model = RandomForestClassifier(max_depth=5, random_state=1)

model.fit(X, y)

output = pd.DataFrame({'PassengerId': test_file.PassengerId, 'Survived': model.predict(X_test)})

output.to_csv('titanicsubmission.csv', index=False)


