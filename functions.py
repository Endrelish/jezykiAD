import collections
import statistics
import numpy
import classes
import csv

def read_data():
    iris = [];

    with open('iris.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            iris.append(classes.Iris(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]))

    return iris


def multiple_mode(values):
    distinct = list(collections.OrderedDict.fromkeys(values))
    count = list(map(lambda d: (d, count_values(d, values)), distinct))
    maxVal = max(map(lambda c: c[1], count))
    return list(filter(lambda v: v[1] == maxVal, count))


def count_values(value, values):
    return len(list(filter(lambda v: v == value, values)))


def calculate_float_results(values):
    result = [('Name:', 'Median:', 'Minimum:', 'Maximum'),
              calculate_set_results(map(lambda v: v.sepalLength, values), 'Sepal length:'),
              calculate_set_results(map(lambda v: v.sepalWidth, values), 'Sepal width:'),
              calculate_set_results(map(lambda v: v.petalLength, values), 'Petal length:'),
              calculate_set_results(map(lambda v: v.petalWidth, values), 'Petal width:')]

    return result


def calculate_set_results(values, name):
    vals = list(values)
    median = statistics.median(vals)
    minimum = min(vals)
    maximum = max(vals)
    return name, median, minimum, maximum


def correlation(iris):
    data = [list(map(lambda i: i.sepalWidth, iris)), list(map(lambda i: i.sepalLength, iris)),
            list(map(lambda i: i.petalWidth, iris)), list(map(lambda i: i.petalLength, iris))]
    result = [
        ['Name:', 'Sepal width', 'Sepal length', 'Petal width:', 'Petal length'],
        ['Sepal width', 0, 0, 0, 0],
        ['Sepal length', 0, 0, 0, 0],
        ['Petal width', 0, 0, 0, 0],
        ['Petal length', 0, 0, 0, 0]
    ]

    for i in range(4):
        for j in range(4):
            result[i + 1][j + 1] = round(numpy.corrcoef(data[i], data[j])[0][1], 4)

    return result;


def histogram(corr, iris):
    data = [list(map(lambda i: i.sepalWidth, iris)), list(map(lambda i: i.sepalLength, iris)),
            list(map(lambda i: i.petalWidth, iris)), list(map(lambda i: i.petalLength, iris))]
    maximum = (1, 2)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            if abs(corr[i][j]) > abs(corr[maximum[0]][maximum[1]]):
                maximum = (i, j)

    his_x = numpy.histogram(data[maximum[0] - 1])
    his_y = numpy.histogram(data[maximum[1] - 1])

    return [(corr[maximum[0]][0], his_x), (corr[0][maximum[1]], his_y)]

