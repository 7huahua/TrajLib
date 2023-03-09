import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import os

# Concatenate all csv files into one dataframe
# csv_folder = "/content/drive/MyDrive/colab/TrajLib/databases/geolife_transmode/seg_feature_csv"
csv_folder = "/Users/hanqiuhan/github/TrajLib/databases/geolife/"
df_list = []
for file_name in os.listdir(csv_folder):
    if file_name.endswith("segment_features_geolife.csv"):
        file_path = os.path.join(csv_folder, file_name)
        df = pd.read_csv(file_path, parse_dates=['starTime','endTime'])
        df_list.append(df)
geolife_data = pd.concat(df_list, ignore_index=True)
geolife_data.drop(['Unnamed: 0'], axis=1, inplace=True)
print(geolife_data.head)

# Drop unnecessary columns
drop_list2 = ['starTime', 'endTime','userId']
geolife_data = geolife_data.drop(drop_list2, axis=1)

# Split data into features and target
y = geolife_data['target']
X = geolife_data.drop('target', axis=1)

# Create classifiers
rf = RandomForestClassifier(n_estimators=50)
dt = DecisionTreeClassifier(max_depth=5)
nn = MLPClassifier()
nb = GaussianNB()
# qda = QuadraticDiscriminantAnalysis()

# Perform 10-fold cross validation and compute accuracy and F1 scores
classifiers = {'Random Forest': rf, 'Decision Tree': dt, 'Neural Network': nn, 'Naive Bayes': nb}
accuracy_scores = {}
f1_scores = {}
for name, clf in classifiers.items():
    accuracy_scores[name] = np.mean(cross_val_score(clf, X, y, cv=10, scoring='accuracy'))
    f1_scores[name] = np.mean(cross_val_score(clf, X, y, cv=10, scoring='f1_macro'))
    print(name,accuracy_scores[name],f1_scores[name])

# Output scores to csv file
scores_df = pd.DataFrame({'Accuracy': accuracy_scores, 'F1 Score': f1_scores})
scores_df.to_csv('classification_scores.csv')

# Plot comparison figure of accuracy and F1 scores
barWidth = 0.4
r1 = np.arange(len(accuracy_scores))
r2 = [x + barWidth for x in r1]


plt.figure(figsize=(10, 6))
plt.bar(r1, list(accuracy_scores.values()), align='center', width=barWidth,color='b', alpha=0.5, label='Accuracy')
plt.bar(r2, list(f1_scores.values()), align='center', width=barWidth, color='g', alpha=0.5, label='F1 Score')
# plt.xticks(range(len(classifiers)), list(classifiers.keys()))
plt.xticks([r + barWidth/2 for r in range(len(classifiers))], list(classifiers.keys()))
plt.xlabel('Classifier')
plt.ylabel('Score')
plt.legend(loc='best')
plt.title('Comparison of Classification Scores')
plt.tight_layout()
plt.show()

import joblib
rf = RandomForestClassifier(n_estimators=50)
dt = DecisionTreeClassifier(max_depth=5)
nn = MLPClassifier()
nb = GaussianNB()

# Save the model
joblib.dump(rf, 'random_forest_model.joblib')
joblib.dump(dt, 'decision_tree_model.joblib')
joblib.dump(nn, 'neural_network_model.joblib')
joblib.dump(nb, 'gaussian_nb_model.joblib')
