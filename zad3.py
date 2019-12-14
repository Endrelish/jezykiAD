# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

iterations_count = 250
test_values_fraction = 0.3


def classify(iterations, train_data, train_result, test_data, test_actual_result):
    scores = []
    iter = []
    for i in range(iterations - 1):
        clf.fit(train_data, train_result)
        prediction = clf.predict(test_data)
        score = accuracy_score(prediction, test_actual_result)
        iter.append(clf.n_iter_)
        scores.append(score)
    return scores, iter

def draw_graph(no_of_graph, data1, data2, color):
    f = plt.figure(no_of_graph)
    plt.plot(data1, data2, color=color, linewidth=1)
    f.show()


data_file = open('pima-indians-diabetes.data.txt', 'r')
data = pd.read_csv(data_file, delimiter=',')
print(data)

data_diabetics = data.Diabetic
main_data_no_diabetic = data.drop(['Diabetic'], axis=1)

train_data, test_data, train_result, test_result = train_test_split(
    main_data_no_diabetic, data_diabetics, test_size=test_values_fraction)

clf = MLPClassifier(solver='adam', alpha=0.000001, random_state=1, max_iter=1, warm_start=True)

scores, iterations = classify(iterations_count, train_data, train_result, test_data, test_result)

print('scores:')
print(scores)
print()
print('iterations:')
print(iterations)
print()

draw_graph(1, iterations, scores, 'blue')





print("After PCA:")

pca = PCA(n_components=2)
pca.fit(main_data_no_diabetic)
result_pca = pca.transform(main_data_no_diabetic)
print(result_pca)

pca_train_data, pca_test_data, pca_train_result, pca_test_result = train_test_split(
    result_pca, data_diabetics, test_size=test_values_fraction)

clf = MLPClassifier(solver='adam', alpha=0.000001, random_state=1, max_iter=1, warm_start=True)

scores, iterations = classify(iterations_count, pca_train_data, pca_train_result, pca_test_data, pca_test_result)

print('scores:')
print(scores)
print()
print('iterations:')
print(iterations)

draw_graph(1, iterations, scores, 'red')

plt.show()