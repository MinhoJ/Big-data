import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

csv = pd.read_csv('diamonds.csv')

# 필요한 열 추출하기
csv_data = csv[["carat","depth","table","price"]]
csv_label = csv["cut"]

# 학습 전용 데이터와 테스트 전용 데이터로 나누기
data_train, data_test, label_train, label_test = train_test_split(csv_data, csv_label)

# 데이터 학습시키기
clf = RandomForestClassifier()
clf.fit(data_train, label_train)

# 데이터 예측하기
predict = clf.predict(data_test)

# 결과 테스트하기
result = pd.DataFrame({"label": label_test, "pre": predict})
print(result[0:10])

ac_score = metrics.accuracy_score(label_test, predict)
print("정답률 =", ac_score)