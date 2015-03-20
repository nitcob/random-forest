# Creates a partial random forest model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# read feature names
feature_names = open("./features_clean.txt").read().splitlines()
col_names = [f.split(" ")[1] for f in feature_names]

# Create data frames from combined data
df_X = pd.read_csv("./X_all.txt", sep=" *", engine="python", header=None, names=col_names)
df_y = pd.read_csv("./y_all.txt", header=None, names=["activity"])
df_sub = pd.read_csv("./subject_all.txt", header=None, names=["subject"])

# Choose training, testing and validation data Scikit-learn doesnt natively
# support pandas. So Convert training and testing sets into matrices  for
# feeding to classifier algorithms

training_set = df_sub["subject"] >= 27
test_set = df_sub["subject"] <= 6
cv_set = (df_sub["subject"] >= 21) & (df_sub["subject"] < 27)

X_train = df_X[training_set].as_matrix()
y_train = df_y[training_set].as_matrix().squeeze()

X_test = df_X[test_set].as_matrix()
y_test = df_y[test_set].as_matrix().squeeze()

X_cv = df_X[cv_set].as_matrix()
y_cv = df_y[cv_set].as_matrix().squeeze()


clf = RandomForestClassifier(n_estimators=500, oob_score=True)
clf.fit(X_train, y_train)
print clf.oob_score_


