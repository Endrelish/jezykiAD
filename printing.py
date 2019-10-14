import matplotlib.pyplot as plt

def print_matrix_results(results):
    row_format = "{:>15}" * len(results[0])
    for row in results:
        print(row_format.format(*row))
    print('\n')


def print_mode(modes):
    if len(modes) > 1:
        print('Cannot determine mode, there are multiple values with the same number of occurrences:')
    for mode in modes:
        print(f'{mode[0]} - {mode[1]}')
    print('\n')


def print_histogram(histogram, title):
    plt.bar(list(map(lambda h: str(round(h, 2)), histogram[1][:-1])), histogram[0])
    plt.suptitle(title)
    plt.show()
