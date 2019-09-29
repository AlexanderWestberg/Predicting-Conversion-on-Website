import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('online_shoppers_intention.csv')

corr = pd.get_dummies(df, drop_first=True).corr()

X = df.iloc[:,:17]
y = df.iloc[:,-1]

X_dummies = pd.get_dummies(X, drop_first=True)
y_dummies = pd.get_dummies(y, drop_first=True)

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#EnsembleModels
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
import catboost as cb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier


#Metrics and HPT 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint



#Training and Testing
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y_dummies, test_size= 0.2, random_state=1)

def ML_scores(y_pred, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
    import pandas as pd
    prec = [precision_score(y_pred, y_test)]
    recall = [recall_score(y_pred, y_test)]
    acc = [accuracy_score(y_pred, y_test)]
    f1_score = [f1_score(y_pred, y_test)]
    model_score = pd.concat([pd.DataFrame(prec), pd.DataFrame(recall), pd.DataFrame(acc), pd.DataFrame(f1_score)], axis=1)
    model_score.columns = ["Precision","Recall", "Accuaracy", "f1_score"]
    print(model_score)
    return model_score

def vfi(model, X):

    importances = cbc.feature_importances_

    # Sort importances
    sorted_index = np.argsort(importances)
    # Create labels
    labels = X.columns[sorted_index]
    sns.set()

    plt.barh(range(X.shape[1]), importances[sorted_index], tick_label=labels)
    plt.show()
    

#All the models


#LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_score = ML_scores(lr_pred, y_test)


#KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_score = ML_scores(knn_pred, y_test)


#DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt_param = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

dt_cv = RandomizedSearchCV(dt,dt_param, cv=5)
dt_cv.fit(X_train, y_train)
dt_pred = dt_cv.predict(X_test)
dt_score = ML_scores(dt_pred, y_test)

#GaussianNB
nb = GaussianNB()

nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_score = ML_scores(nb_pred, y_test)

#RandomForestClassifier
rf = RandomForestClassifier()

rf_param = {"max_depth": [3, None],
              "max_features": np.arange(1, 10),
              "min_samples_leaf": np.arange(1, 10),
              "criterion": ["gini", "entropy"]}

rf_cv = GridSearchCV(rf,rf_param, cv=5)

rf_cv.fit(X_train, y_train)
rf_pred = rf_cv.predict(X_test)
rf_score = ML_scores(rf_pred, y_test)


#AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, random_state=500)


ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_score = ML_scores(ada_pred, y_test)


#GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=500)


gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_score = ML_scores(gb_pred, y_test)

#BaggingClassifier
bag = BaggingClassifier(base_estimator=rf, n_estimators=21,  random_state=500)

bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)
bag_score = ML_scores(bag_pred, y_test)

#Catboost
cbc = cb.CatBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

cbc.fit(X_train, y_train)
cbc_pred = cbc.predict(X_test)
cbc_score = ML_scores(cbc_pred, y_test)


"""
#Light Gradient Boosting
lgbc = lgb.LGBMRegressor(max_depth = 3, learning_rate = 0.1, n_estimators = 100, seed=500)
lgbc.fit(X_train, y_train)

lgbc_pred = lgbc.predict(X_test) 
lgbc_score = ML_scores(lgbc_pred, y_test)
"""
#VotingClassifier
vtc = VotingClassifier(estimators=[("ada", ada), ("gb", gb), ("bag", bag), ("nb",nb)], weights=[2,1,1,2] )


vtc.fit(X_train, y_train)
vtc_pred = vtc.predict(X_test)
vtc_score = ML_scores(vtc_pred, y_test)

#Averaging

avg = VotingClassifier(estimators=[("ada", ada), ("gb", gb), ("bag", bag),("cbc", cbc), ("nb",nb)], weights=[3,1,1,2,2], voting='soft' )


avg.fit(X_train, y_train)
avg_pred = avg.predict(X_test)
avg_score = ML_scores(avg_pred, y_test)

#Stacking

stk = StackingClassifier(classifiers=[gb, cbc, rf], meta_classifier=rf, use_features_in_secondary = False)

stk.fit(X_train, y_train)
stk_pred = stk.predict(X_test)
stk_score = ML_scores(stk_pred, y_test)

#Deep Learning
import keras
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu', input_dim = 26))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


ann_pred = classifier.predict(X_test)
ann_pred = (ann_pred > 0.5)
ann_score = ML_scores(ann_pred, y_test)



#Evaluating

all_model_names = ["LogisticRegression", "KNeighborsClassifier","DecisionTreeClassifier", "GaussianNB", "RandomForestClassifier",  "AdaBoostClassifier", "GradientBoostingClassifier", "BaggingClassifier", "CatBoostClassifier", "VotingClassifier","AveragingVotingClassifier","Stacking Classifier" ,"Artificel Neural Networks" ]

all_model_score = pd.concat([lr_score, knn_score, dt_score, nb_score, rf_score,  ada_score, gb_score, bag_score, cbc_score, vtc_score, avg_score,stk_score, ann_score])
all_model_score.index = all_model_names
all_model_score = all_model_score.sort_values(by=["f1_score"], ascending=False)
