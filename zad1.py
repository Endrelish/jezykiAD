import functions
import printing

iris = functions.read_data()

printing.print_matrix_results(functions.calculate_float_results(iris))
printing.print_mode(functions.multiple_mode(list(map(lambda i: i.flowerClass, iris))))
correlation = functions.correlation(iris)
printing.print_matrix_results(correlation)
histogram = functions.histogram(correlation, iris)
printing.print_histogram(histogram[0][1], histogram[0][0])
printing.print_histogram(histogram[1][1], histogram[1][0])

