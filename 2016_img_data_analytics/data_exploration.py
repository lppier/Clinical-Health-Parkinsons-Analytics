import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ctrl_path = "/home/pier/Machine_Learning/Clinical_Health_Parkinsons_Analytics/2016_img_data_analytics/data/control/"
park_path = "/home/pier/Machine_Learning/Clinical_Health_Parkinsons_Analytics/2016_img_data_analytics/data/parkinsons/"

df_ctrl = pd.read_csv(ctrl_path + 'control_all_with_header.csv')
df_park = pd.read_csv(park_path + 'park_all_with_header.csv')
df = pd.concat([df_ctrl, df_park])

# Compute the correlation matrix
df_corr = df.drop(['Subject', 'PWP'], axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(df_corr.corr())
plt.xticks(range(len(df_corr.columns)), df_corr.columns)
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
plt.show()

# remove z
df = df.drop(['Z'], axis=1)
print(df.head())

# sns.pairplot(df)
# plt.show()
#
# sns.distplot(df['Test_ID'])
# plt.show()
# Segregate data into 3 tests
df_0 = df[df['Test_ID'] == 0]
df_1 = df[df['Test_ID'] == 1]
df_2 = df[df['Test_ID'] == 2]

# For each of the tests, take out one PWP and one control for test data
df_0_test_pwp = df_0[df_0['Subject'] == 77]
df_0_test_ctrl = df_0[df_0['Subject'] == 1]
df_0_test_pwp = df_0_test_pwp.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_0_test_ctrl = df_0_test_ctrl.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_combined_pwp_ctrl = pd.concat([df_0_test_ctrl, df_0_test_pwp])
df_combined_pwp_ctrl.to_csv('test_0_pwp_comb_ctrl.csv', index=False)
df_0_test_pwp.to_csv('test_0_pwp_subject.csv', index=False)
df_0_test_ctrl.to_csv('test_0_ctrl_subject.csv', index=False)
df_0_train = df_0[df_0['Subject'] != 77]
df_0_train = df_0_train[df_0_train['Subject'] != 1]
df_0_train = df_0_train.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_0_train.to_csv('train_0.csv', index=False)

df_1_test_pwp = df_1[df_1['Subject'] == 77]
df_1_test_ctrl = df_1[df_1['Subject'] == 1]
df_1_test_pwp = df_1_test_pwp.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_1_test_ctrl = df_1_test_ctrl.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df1_combined_pwp_ctrl = pd.concat([df_1_test_ctrl, df_1_test_pwp])
df1_combined_pwp_ctrl.to_csv('test_1_pwp_comb_ctrl.csv', index=False)
df_1_test_pwp.to_csv('test_1_pwp_subject.csv', index=False)
df_1_test_ctrl.to_csv('test_1_ctrl_subject.csv', index=False)
df_1_train = df_1[df_1['Subject'] != 77]
df_1_train = df_1_train[df_1_train['Subject'] != 1]
df_1_train = df_1_train.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_1_train.to_csv('train_1.csv', index=False)

df_2_test_pwp = df_2[df_2['Subject'] == 77]
df_2_test_ctrl = df_2[df_2['Subject'] == 1]
df_2_test_pwp = df_2_test_pwp.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_2_test_ctrl = df_2_test_ctrl.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df2_combined_pwp_ctrl = pd.concat([df_2_test_ctrl, df_2_test_pwp])
df2_combined_pwp_ctrl.to_csv('test_2_pwp_comb_ctrl.csv', index=False)
df_2_test_pwp.to_csv('test_2_pwp_subject.csv', index=False)
df_2_test_ctrl.to_csv('test_2_ctrl_subject.csv', index=False)
df_2_train = df_2[df_2['Subject'] != 77]
df_2_train = df_2_train[df_2_train['Subject'] != 1]
df_2_train = df_2_train.drop(['Subject', 'Test_ID', 'Timestamp'], axis=1)
df_2_train.to_csv('train_2.csv', index=False)


# from sklearn.model_selection import train_test_split
#
# df_0 = df_0.drop(['Subject','Test_ID','Timestamp', 'Z'], axis=1)
# train_0, test_0 = train_test_split(df_0, test_size=0.25)
# train_0.to_csv('train_0.csv', index=False)
# test_0.to_csv('test_0.csv', index=False)
#
# df_1 = df_1.drop(['Subject', 'Test_ID','Timestamp', 'Z'], axis=1)
# train_1, test_1 = train_test_split(df_1, test_size=0.25)
# train_1.to_csv('train_1.csv', index=False)
# test_1.to_csv('test_1.csv', index=False)
#
# df_2 = df_2.drop(['Subject', 'Test_ID','Timestamp', 'Z'], axis=1)
# train_2, test_2 = train_test_split(df_2, test_size=0.25)
# train_2.to_csv('train_2.csv', index=False)
# test_2.to_csv('test_2.csv', index=False)

import sklearn
import numpy as np

X = df_0.drop(['Subject', 'PWP', 'Test_ID', 'Timestamp', 'Z'], axis=1)
y = df_0['PWP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
# sgd_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=20, scoring="accuracy")
