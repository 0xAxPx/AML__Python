import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import pylab as pl
import numpy as np
from sklearn.tree import export_graphviz
from util.logger import initLogger

log = initLogger(__name__)

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
td = pd.concat([data_train, data_test], ignore_index=True, sort=False)

log.info("1. Data set columns: \n", td.columns)

log.info("2. Get rid of NaN for some features")
log.info(td.Cabin.isnull().sum())
if td.Embarked.isnull().sum() > 0: td.Embarked.fillna(td.Embarked.mode()[0], inplace=True)
td.Cabin = td.Cabin.fillna('NA')

log.info("3. Survived ordered by Pclass of cabin:\n",
         td[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                           ascending=False)
         )
log.info("4. Survived ordered by sex:\n",
         td[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
         )

log.info("5. Visualize data of age of survived:")
g = sns.FacetGrid(td, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

log.info("6. Drop data we will not use in our model")
td.drop(['Name', 'Fare', 'Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

log.info("7. Transform male/female into binaries for Sex and Embarked column")
label = LabelEncoder()
dicts = {}
label.fit(td.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
td.Sex = label.transform(td.Sex)
log.info(Counter(td.Sex))

dicts1 = {}
label.fit(td.Embarked.drop_duplicates())
dicts1['Embarked'] = list(label.classes_)
td.Embarked = label.transform(td.Embarked)
log.info("EMBARKED - ", Counter(td.Embarked))

log.info("8. Ad hoc data analysis is done \n", td.head())

log.info("----------- 9. Fit model and Predict  ---------------- ")
clf_dtc = DecisionTreeClassifier(max_depth=6)
clf_rfc = RandomForestClassifier(n_estimators=70)

train_data = td
train_data = train_data.dropna()

# target  object
y = train_data['Survived']

# exclude Survived from data set
X = train_data.drop(['Survived'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf_dtc.fit(x_train, np.ravel(y_train))

log.info("NB Accuracy: {} %".format(repr(round(clf_dtc.score(x_test, y_test) * 100, 2))))
result_rf = cross_val_score(clf_dtc, x_train, y_train, cv=10, scoring='accuracy')
log.info('The cross validated score for DecisionTree is: {}%'.format(round(result_rf.mean() * 100, 2)))
y_pred = cross_val_predict(clf_dtc, x_train, y_train, cv=10)
sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='3.0f', cmap="summer")
# plt.title('Confusion_matrix for NB', y=1.05, size=15)

# Add prediction to dataframe
probas = clf_dtc.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
roc_auc = auc(fpr, tpr)
# pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('DecisionTreeClassifier',roc_auc))
log.info("Area under the ROC curve : %f" % roc_auc)

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################

i = np.arange(len(tpr))  # index for df
roc = pd.DataFrame(
    {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
     'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
roc.loc[(roc.tf - 0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color='red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
plt.show()

clf_rfc.fit(x_train, np.ravel(y_train))
log.info("NB Accuracy: {} %".format(repr(round(clf_rfc.score(x_test, y_test) * 100, 2))))

result_rf = cross_val_score(clf_rfc, x_train, y_train, cv=10, scoring='accuracy')
log.info('The cross validated score for RandomForestClassifier is: {}%'.format(round(result_rf.mean() * 100, 2)))

y_pred = cross_val_predict(clf_dtc, x_train, y_train, cv=10)
sns.heatmap(confusion_matrix(y_train, y_pred), annot=True, fmt='3.0f', cmap="autumn")
# plt.title('Confusion_matrix for NB', y=1.05, size=15)


log.info(
    export_graphviz(clf_dtc, out_file=None, feature_names=x_train.columns.tolist(), class_names=['Died', 'Survived'],
                    rounded=True, proportion=False, precision=0, filled=True))
