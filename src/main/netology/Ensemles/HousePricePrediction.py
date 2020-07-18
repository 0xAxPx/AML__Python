import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from util.logger import initLogger

log = initLogger(__name__)


# util func
def X_better(mae1, mae2):
    if mae1_test > mae2_test:
        print("Better Option 2, MAE2_test = {} when MAE1_test = {}, label_X_train".format(mae2_test, mae1_test))
        return label_X_train
    else:
        print("Better Option 1, MAE1_test = {} when MAE2_test = {}, X_train_drop ".format(mae1_test, mae2_test))
        return X_train_drop


X = pd.read_csv("house_params.csv")

# Get rid of NaN for target column"
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Get rid of columns with missing values, in other case of dealing with missing values we can impute missing value with mean values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data"
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, test_size=0.2,
                                                    random_state=0)

rndmForest = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_leaf=20, max_features=0.5, n_jobs=-1)

# Option 1 : Get rid of categorical columns (dtype = 'object')
X_train_drop = X_train.select_dtypes(exclude=['object'])
X_test_drop = X_test.select_dtypes(exclude=['object'])

# Fit model on base data
rndmForest.fit(X_train_drop, y_train)

# Importance of features
imp = pd.Series(rndmForest.feature_importances_)
imp.sort_values(ascending=False)

# Do predictions and get MAE
y_predicts_test = rndmForest.predict_proba(X_test_drop)[:, 1]
y_predicts_train = rndmForest.predict_proba(X_train_drop)[:, 1]
mae1_test = mean_absolute_error(y_test, y_predicts_test)
mae1_train = mean_absolute_error(y_train, y_predicts_train)
log.info("Dropped MAE1_test = {}".format(mae1_test))
log.info("Dropped MAE1_train = {}".format(mae1_train))

# Option 2  - Label categorical data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if
                   set(X_train[col]) == set(X_test[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_test = X_test.drop(bad_label_cols, axis=1)

# label encoder
label = LabelEncoder()
for col in set(good_label_cols):
    label_X_train[col] = label.fit_transform(X_train[col])
    label_X_test[col] = label.transform(X_test[col])

# Fit model
rndmForest.fit(label_X_train, y_train)

# Importance of features
imp = pd.Series(rndmForest.feature_importances_)
imp.sort_values(ascending=False)

# Option 2 : Predict
y_predicts_test = rndmForest.predict_proba(label_X_test)[:, 1]
y_predicts_train = rndmForest.predict_proba(label_X_train)[:, 1]
mae2_test = mean_absolute_error(y_test, y_predicts_test)
mae2_train = mean_absolute_error(y_train, y_predicts_train)
log.info("Labeled MAE2_test = {}".format(mae2_test))
log.info("Labeled MAE2_train = {}".format(mae2_train))

classifier = StackingClassifier(
    [
        ('logr', LogisticRegression()),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier())
    ],
    LogisticRegression())

X_better_train = X_better(mae1_test, mae2_test)

classifier.fit(X_better_train, y_train)

classifier.named_estimators_['logr']

y_pred_proba = classifier.predict_proba(X_better_train)[:,1]

classifier.final_estimator_

pd.Series(classifier.final_estimator_.coef_.flatten()[:len(classifier.named_estimators_.keys())], index=classifier.named_estimators_.keys()).plot(kind='barh')

