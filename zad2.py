import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def procent(data):
    objects_no = len(data.index)
    count = data.count()
    per_missing = 0
    for i in count:
        if i != objects_no:
            per_missing += (objects_no - i)
    return (per_missing / objects_no) * 100

file = open('Admission_Predict2.csv')

raw_data = pd.read_csv(file, delimiter=';')
print(raw_data.head(5))
data = raw_data.drop(['Serial No.'], axis=1)

print('Ilosc danych w kolumnach:')
print(raw_data.count())


nan_procent = procent(data)
print('Procent:')
print(nan_procent)

## Krzywa regresji dla danych bez braków:
data_without_nans = data.dropna()
print(data_without_nans)

X = data_without_nans[['TOEFL_Score']]
Y = data_without_nans['CGPA']

print(X)

print(Y)


model = LinearRegression().fit(X, Y)
r_sq = model.score(X, Y)
print('R^2:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


x = np.linspace(0,50,100)
y = model.coef_*x+model.intercept_
plt.plot(x, y, '-r')
plt.title('Graph of linear regression')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
