import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(mx, mn, val):
    return (val - mn) / (mx - mn)


class KNN:
    def __init__(self, train, k):
        self.k = k
        self.examples = train['diagnosis']
        self.data = {}
        self.norm_limits = {}
        self.normalize_train(train)

    def normalize_train(self, train):
        features = train.keys().drop(['diagnosis'])
        for f in features:
            mx, mn = max(train[f]), min(train[f])
            self.data[f] = [normalize(mx, mn, v) for v in train[f]]
            self.norm_limits[f] = (mx, mn)

    def distance(self, data, test, example):
        dist = 0
        for feature in self.data:
            dist += (self.data[feature][example] - data[feature][test])**2
        return np.sqrt(dist)

    def normalize_data(self, data):
        d = {}
        for f, lims in self.norm_limits.items():
            mx, mn = lims
            d[f] = [normalize(mx, mn, v) for v in data[f]]
        return d

    def classify(self, data, test):
        # print(test)
        distances = sorted([(self.distance(data, test, i), self.examples[i])
                            for i in range(0, len(self.examples))], key=lambda x: x[0])
        knn = distances[:self.k]
        count = sum(1 for n in knn if n[1])

        return 1 if count >= self.k - count else 0

    def accuracy(self, data):
        tests = range(len(data['diagnosis']))
        d = self.normalize_data(data)
        matches = sum(1 for t in tests if self.classify(d, t) == data['diagnosis'][t])
        return matches / len(tests)


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    train_data = pd.DataFrame(data)
    df_test = pd.read_csv("test.csv")
    test_data = pd.DataFrame(df_test)

    knn = KNN(train_data, 9)
    ks = [1, 3, 9, 27]
    accuracies = []
    for k in ks:
        knn.k = k
        accuracy = knn.accuracy(test_data)
        print("{}NN Accuracy: {:.2%}".format(k, accuracy))
        accuracies.append(accuracy*100)
    print(accuracies)

    fig, ax = plt.subplots()
    ax.scatter(ks, accuracies)
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('k choice')
    ax.scatter(ks, accuracies)
    plt.title("KNN Accuracy for different k values")

    for x, y in zip(ks, accuracies):
        if y < 98:
            ax.annotate("{:.2f}%".format(y), xy=(x, y), textcoords='offset points', xytext=(0, 10), ha='center')
        else:
            ax.annotate("{:.2f}%".format(y), xy=(x, y), textcoords='offset points', xytext=(0, -15), ha='center')

    plt.show()


