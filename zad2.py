import statistics
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


def create_graph(a, b, X, Y):
    x = np.linspace(90,130,100)
    y = a*x+b
    plt.plot(x, y, '-r')
    plt.plot(X, Y, 'ro')

    plt.title('Graph of linear regression')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.grid()
    plt.show()


file = open('Admission_Predict2.csv')

raw_data = pd.read_csv(file, delimiter=';')
print(raw_data.head(5))
data = raw_data.drop(['Serial No.'], axis=1)

print('Ilosc danych w kolumnach:')
print(raw_data.count())


nan_procent = procent(data)
print('Procent danych z brakami: ', nan_procent)

## Krzywa regresji dla danych bez braków:
data_without_nans = data.dropna()

x_column = 'TOEFL_Score'
y_column = 'CGPA'

X = data_without_nans[[x_column]]
Y = data_without_nans[y_column]


model = LinearRegression().fit(X, Y)
r_sq = model.score(X, Y)
print('R^2:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

create_graph(model.coef_, model.intercept_, X, Y)


##dane przed uzupełnieniem:
mean = data[x_column].mean()
st_dev = statistics.stdev(data_without_nans[x_column])
quantile = data.TOEFL_Score.quantile([0.25, 0.5, 0.75])

## Uzupełnienie danych metoda imputation;
filled_data = data.copy()
filled_data[x_column] = data[x_column].fillna(mean)


##dane po uzupełnieniu:
mean_after_fill = filled_data[x_column].mean()
st_dev_after_fill = statistics.stdev(filled_data[x_column])
quantile_after_fill = filled_data.TOEFL_Score.quantile([0.25, 0.5, 0.75])

print()
print('Srednia przed: ', mean)
print('Srednia po: ', mean_after_fill)
print('Roznica: ', abs(mean_after_fill - mean))
print()

print('Odchylenie standardowe przed: ', st_dev)
print('Odchylenie standardowe po: ', st_dev_after_fill)
print('Roznica: ', abs(st_dev_after_fill - st_dev))
print()

print('Kwartyle przed: ')
print(quantile)
print('Kwartyle po: ')
print(quantile_after_fill)
print('Roznica: ')
print(abs(quantile_after_fill - quantile))
print()


#regracja po imputancji
X_2 = filled_data[[x_column]]
Y_2 = filled_data[y_column]
model_2 = LinearRegression().fit(X_2, Y_2)
r_sq_2 = model.score(X_2, Y_2)
print('R^2 po imputancji:', r_sq_2)
print('intercept po imputancji:', model_2.intercept_)
print('slope po imputancji:', model_2.coef_)
print()
print('Różnica R^2:', abs(r_sq - r_sq_2))
print('Różnica intercept:', abs(model.intercept_ - model_2.intercept_))
print('Różnica slope:', abs(model.coef_ - model_2.coef_))
print()


create_graph(model_2.coef_, model_2.intercept_, X_2, Y_2)
