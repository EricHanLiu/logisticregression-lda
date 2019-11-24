import numpy as np
import time
from scipy import stats
from matplotlib import pyplot
import itertools


class Matrices:
    def __init__(self):
        self.wine_matrix = None
        self.cancer_matrix = None
        self.wine_stats = {}
        self.cancer_stats = {}
        self.cancer_status = {}
        self.feature_means = []
        self.feature_stds = []

    @staticmethod
    def parse_file_into_array(filename, separator):
        """
        Takes a csv-type file and parses it into a two-dimensional python array
        :param filename: name of the file to open
        :param separator: what we're splitting each row by (, or ;)
        :return: two dimensional array of float values
        """
        arr = []
        with open(filename) as file:
            for row in file.read().splitlines():
                try:
                    row_arr = [float(cell) for cell in row.split(separator)]
                    if 'winequality' in filename:
                        row_arr[-1] = 1 if row_arr[-1] > 5 else 0  # convert to binary classification
                    elif 'breast-cancer' in filename:
                        row_arr[-1] = 1 if row_arr[-1] == 4 else 0  # convert to binary classification
                except ValueError:
                    continue
                arr.append(row_arr)
        return arr

    def load_matrices(self):
        """
        Initializes the matrices as numpy array objects
        """
        self.wine_matrix = np.array(self.parse_file_into_array('winequality-red.csv', ';'))
        self.cancer_matrix = np.array(self.parse_file_into_array('breast-cancer-wisconsin.data', ','))

    def compute_statistics(self):
        """
        Computes some statistics on our datasets
        """
        for i in range(len(self.wine_matrix[0, :])):
            feature = self.wine_matrix[:, i]
            self.wine_stats['feature ' + str(i)] = {}
            if i == 11:  # results column
                self.wine_stats['feature ' + str(i)]['positive_class_ratio'] = (feature == 1).sum() / len(feature)
            null, self.wine_stats['feature ' + str(i)]['pvalue'] = stats.normaltest(feature)

            # plot
            # pyplot.hist(feature, bins=50)
            # pyplot.show()

        for i in range(len(self.cancer_matrix[0, :])):
            feature = self.cancer_matrix[:, i]
            self.cancer_stats['feature ' + str(i)] = {}
            if i == 10:  # results column
                self.cancer_stats['feature ' + str(i)]['positive_class_ratio'] = (feature == 1).sum() / len(feature)
            null, self.cancer_stats['feature ' + str(i)]['pvalue'] = stats.normaltest(feature)

            # plot
            # pyplot.hist(feature, bins=50)
            # pyplot.show()

    def remove_outliers(self, matrix):
        """
        Removes rows that contain features that are outside of 3 standard deviations of the mean of that feature
        """
        input = matrix[:, :-1]
        row_incides_to_delete = []
        for j, column in enumerate(input.transpose()):
            self.feature_means.append(np.mean(column))
            self.feature_stds.append(np.std(column))

            for i, row in enumerate(input):
                cell = input[i, j]
                if cell > self.feature_means[j] + 3 * self.feature_stds[j] or cell < self.feature_means[j] - 3 * \
                        self.feature_stds[j]:
                    row_incides_to_delete.append(i)
        matrix = np.delete(matrix, row_incides_to_delete, 0)
        return matrix, len(list(set(row_incides_to_delete)))


class LogisticRegression:
    def __init__(self, input, output, learning_rate, descents):
        """
        :param input: training data (X)
        :param output: training data desired output (y)
        :param learning_rate: how fast the model learns
        :param descents: the number of gradient descent iterations
        """
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.descents = descents
        self.num_features = len(input[0, :])
        self.num_samples = len(input[:, 0])
        self.w0 = np.array([0] * self.num_features)  # initial weight vector w0
        self.final_weights = None

    def fit(self, input, output):
        """
        Trains the model by modifying the model parameters using the inputs
        :param input: X
        :param output: y
        """
        last = self.w0  # equivalent to wk in the loop
        for iteration in range(self.descents):
            sum_over_i = [0.0] * len(last)
            for i in range(len(input)):
                wtx = np.dot(np.transpose(last), input[i, :])
                sum_over_i = np.add(sum_over_i, input[i, :] * (output[i] - self.sigmoid(wtx)))
            last = np.add(last, self.learning_rate * sum_over_i)
        self.final_weights = last

    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    def predict(self, input):
        """
        Outputted probabilities need to be converted to binary, thresholded on 0.5
        :param input: set of input points X
        :return: output predictions (y hat) for this set of input points
        """
        a = np.dot(np.transpose(self.final_weights), input)
        prob = self.sigmoid(a)
        return 1 if prob > 0.5 else 0

    def evaluate_acc(self, input_set, output_set):
        """
        Evaluates the accuracy of the model based on how many correct outputs over the total number of samples
        """
        successes = 0
        num_samples = len(output_set)
        for i in range(num_samples):
            y_hat = self.predict(input_set[i, :])
            y = output_set[i]
            if y == y_hat:
                successes += 1
        return successes / num_samples

    def k_folds_cross_validate(self, k):
        """
        Runs k training iterations of the model, leaving out a set of size 1/k as the validation set each iteration
        """
        start_time = time.time()
        partition_size = int(self.num_samples / k)
        partitions = [
            (i * partition_size, (i + 1) * partition_size) for i in range(k)
        ]
        average_accuracy = 0.0
        for start, end in partitions:
            validation_input_set = self.input[start:end, :]  # subset of input of size k (k samples)
            validation_output_set = self.output[start:end]  # subset of output of size k (k outputs)
            training_input_set = np.delete(self.input, np.s_[start:end], 0)  # subset of input excluding validation set
            training_output_set = np.delete(self.output, np.s_[start:end], 0)  # subset of output excluding validation

            self.fit(training_input_set, training_output_set)
            accuracy = self.evaluate_acc(validation_input_set, validation_output_set)
            # print('Accuracy: ', accuracy)  # accuracy of each fold
            average_accuracy += accuracy
        average_accuracy /= k
        print('Average accuracy: ', average_accuracy)
        print('Runtime: ', time.time() - start_time, 'seconds')


class LDA:
    def __init__(self, data_matrix, input, output):
        self.data_matrix = data_matrix
        self.input = input
        self.output = output
        self.mean_A, self.mean_B, self.cov = [], [], []
        self.p_A, self.p_B = 0, 0
        self.num_samples = len(input[:, 0])

    @staticmethod
    def split(matrix):
        arr = [[], []]
        for row in matrix:
            if row[-1] == 0 or row[-1] == 2:
                arr[0].append(row[:-1])
            else:
                arr[1].append(row[:-1])
            # arr[0].append(row[:-1]) if (row[-1] == 0 or row[-1] == 2) else arr[1].append(row[:-1])
        return arr

    def fit(self, data_set):
        # check percents
        arr = self.split(data_set)
        arr = np.array(arr)
        self.mean_A = np.array(arr[0]).mean(axis=0)
        self.mean_B = np.array(arr[1]).mean(axis=0)

        # should it include the output vector too ??
        cov_a = np.cov(np.transpose(arr[0]))
        cov_b = np.cov(np.transpose(arr[1]))
        self.cov = (cov_a + cov_b)

        len_a = len(arr[0])
        len_b = len(arr[1])
        self.p_A = np.log(len_a / (len_a + len_b))
        self.p_B = np.log(len_b / (len_a + len_b))

    def predict(self, input):
        y = []
        cov_inv = np.linalg.pinv(self.cov)
        transpose_mean_a = np.transpose(self.mean_A)
        transpose_mean_b = np.transpose(self.mean_B)
        for row in input:
            a = np.transpose(row).dot(cov_inv).dot(self.mean_A) - 0.5 * transpose_mean_a.dot(cov_inv).dot(
                self.mean_A) + self.p_A
            b = np.transpose(row).dot(cov_inv).dot(self.mean_B) - 0.5 * transpose_mean_b.dot(cov_inv).dot(
                self.mean_B) + self.p_B
            if a > b:
                y.append(0)
            else:
                y.append(1)
        return y

    @staticmethod
    def predict_accuracy(output, predict):
        successes = 0
        for i in range(len(output)):
            if output[i] == predict[i]:
                successes += 1
        return successes / len(output)

    def k_folds_cross_validate(self, k):
        start_time = time.time()
        partition_size = int(self.num_samples / k)
        partitions = [
            (i * partition_size, (i + 1) * partition_size) for i in range(k)
        ]
        average_accuracy = 0.0
        for start, end in partitions:
            validation_input_set = self.input[start:end, :]  # subset of input of size k (k samples)
            validation_output_set = self.output[start:end]  # subset of output of size k (k outputs)
            training_data_set = np.delete(self.data_matrix, np.s_[start:end], 0)

            self.fit(training_data_set)
            accuracy = self.predict_accuracy(validation_output_set, self.predict(validation_input_set))
            average_accuracy += accuracy
        average_accuracy /= k
        print('Average accuracy: ', average_accuracy)
        print('Runtime: ', time.time() - start_time, 'seconds')


def test_learning_rates(model):
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1, 100]
    for rate in learning_rates:
        print('Learning rate: ', rate)
        model.learning_rate = rate
        model.k_folds_cross_validate(5)
        print()


def main():
    matrices = Matrices()
    matrices.load_matrices()
    matrices.compute_statistics()
    # uncomment this to clean the data of outliers
    # matrices.wine_matrix, removed = matrices.remove_outliers(matrices.wine_matrix)
    matrices.cancer_matrix, removed = matrices.remove_outliers(matrices.cancer_matrix)
    print('Removed training examples: ', removed)
    wine_input = matrices.wine_matrix[:, :-1]
    wine_output = matrices.wine_matrix[:, -1]
    cancer_input = matrices.cancer_matrix[:, :-1]
    cancer_output = matrices.cancer_matrix[:, -1]

    print('________WINE DATASET________')
    print(matrices.wine_stats)
    # wine gets better accuracy without extra quadratic features
    print('___LOGISTIC REGRESSION___')
    wine_lr = LogisticRegression(wine_input, wine_output, 0.1, 100)
    test_learning_rates(wine_lr)

    print('___LDA___')
    wine_lda = LDA(matrices.wine_matrix, wine_input, wine_output)
    wine_lda.k_folds_cross_validate(5)
    # leave one out
    wine_lda.k_folds_cross_validate(wine_lda.num_samples)

    print('________CANCER DATASET________')
    print(matrices.cancer_stats)
    # cancer gets better accuracy with quadratic features
    print('___LOGISTIC REGRESSION___')
    quadratic_input = quadratic_expansion(cancer_input)
    cancer_lr = LogisticRegression(quadratic_input, cancer_output, 0.1, 100)
    test_learning_rates(cancer_lr)

    print('___LDA___')
    cancer_lda = LDA(matrices.cancer_matrix, cancer_input, cancer_output)
    cancer_lda.k_folds_cross_validate(5)
    # leave one out
    cancer_lda.k_folds_cross_validate(cancer_lda.num_samples)


def quadratic_expansion(matrix):
    """
    Returns a matrix with twice as many features, where the new features are quadratic expansions of the originals
    """
    arr = np.copy(matrix)
    arr = np.array([x + x ** 2 for x in arr])
    return np.concatenate((matrix, arr), axis=1)


if __name__ == '__main__':
    main()
