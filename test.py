import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

geolife_data = pd.read_csv('databases/geolife/segment_features_geolife.csv',parse_dates=['starTime','endTime'])
print(geolife_data.head())


drop_list = ['isInValid', 'isPure', 'stopRate', 'starTime', 'endTime', 'isWeekDay', 'dayOfWeek', 'durationInSeconds',
    'distanceTravelled', 'startToEndDistance', 'startLat', 'starLon', 'endLat', 'endLon', 'selfIntersect',
    'modayDistance', 'tuesdayDistance', 'wednesdayDay', 'thursdayDistance', 'fridayDistance', 'saturdayDistance',
    'sundayDistance', 'stopTotal', 'stopTotalOverDuration', 'userId']

drop_list2 = ['starTime', 'endTime']

geolife_data = geolife_data.drop(drop_list2, axis=1)
geolife_data = geolife_data.drop(geolife_data[geolife_data['target']=='car'].index)


# 获取特征和标签

y = geolife_data['target']
X = geolife_data.drop('target', axis=1)

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=50)

# 进行10折交叉验证，计算准确性和f1分数
scores = cross_val_score(rf, X, y, cv=10)
accuracy = np.mean(scores)
f1 = cross_val_score(rf, X, y, cv=10, scoring='f1_macro').mean()

print('Accuracy:', accuracy)
print('F1 Score:', f1)
