import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("Egypt_map.csv")

print(data.shape)
print(data.dtypes)
print(data.isnull().sum())

data = data.drop(["capital","iso","admin","country"], axis=1)
print(data.isnull().sum().sum())

data = data.dropna()
print(data.isnull().sum().sum())
print(data.shape)

La = LabelEncoder()
data["city"] = La.fit_transform(data["city"])
print(data["city"].dtypes)
print(data.dtypes)

print(data.head())

print(data.groupby("city")["pop_people"].sum().sort_values(ascending=True))

data["category"] = 1
data.loc[(data["pop_people"] > 500000) & (data["pop_people"] <= 1000000), "category"] = 2
data.loc[(data["pop_people"] > 1000000) & (data["pop_people"] <= 2000000), "category"] = 3
data.loc[data["pop_people"] > 2000000, "category"] = 4
print(data["category"].value_counts())

# print(data.head())
print(data.head().sort_values(by=["category"] ,ascending=False))

x = data.drop("category", axis=1)
y = data["category"]
print(x.shape)
print(y.shape)
ss = StandardScaler()
x = ss.fit_transform(x)


Lo = LogisticRegression()
Lo.fit(x, y)
print(Lo.score(x, y))

print("_" * 100)

Dt = DecisionTreeClassifier()
Dt.fit(x, y)
print(Dt.score(x, y))

p1 = [0.55,0.77,0.55,0.132,0.9]
p2 = [0.51,0.78,0.91,0.16489,0.15000]
p3 = [0.9,0.5,0.05,0.66,0.22]

print(Dt.predict([p1, p2, p3]))