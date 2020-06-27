import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(mx, mn, val):
    return (val - mn) / (mx - mn)


def normalize_train(train):
    features = train.keys().drop(['diagnosis'])
    data, norm_limits = {}, {}
    for f in features:
        mx, mn = max(train[f]), min(train[f])
        data[f] = [normalize(mx, mn, v) for v in train[f]]
        norm_limits[f] = (mx, mn)

    return data, norm_limits


def normalize_test(test, limits):
    features = test.keys().drop(['diagnosis'])
    data = {}
    for f in features:
        mx, mn = limits[f]
        data[f] = [normalize(mx, mn, v) for v in test[f]]
    return data


class KNN:
    def __init__(self, examples, c, features, k):
        self.k = k
        self.examples = examples
        self.c = c
        self.data = features

    def distance(self, data, test, example):
        dist = 0
        for feature in self.data:
            dist += (self.data[feature][example] - data[feature][test])**2
        return np.sqrt(dist)

    def classify(self, data, test):
        distances = sorted([(self.distance(data, test, i), self.c[i])
                            for i in self.examples], key=lambda x: x[0])
        knn = distances[:self.k]
        t = sum(1 for n in distances if n[1])
        f = len(distances) - t
        count = sum(1 for n in knn if n[1])

        return 1 if count >= len(knn) - count else 0

    def accuracy(self, c, data):
        tests = range(len(c))
        matches = sum(1 for t in tests if self.classify(data, t) == c[t])
        return matches / len(tests)


class Node:
    def __init__(self, features, examples):
        self.s1 = None
        self.s2 = None
        self.classification = None #should be to majority
        self.f = None #if not a leaf should conatain the feature which devide his sons
        self.mid = None # the value which under it is s1 and above is s2
        self.features = features # the features who left. Important Type = Index
        self.examples = examples # current examples. Important Type = List

    def majority(self, c):
        total = len(self.examples)
        t = sum(1 for e in self.examples if c[e])
        f = total - t
        return (1, t / total) if t >= f else (0, f / total)


def entropy(examples):
    total = len(examples)
    if total == 0:
        return 0
    p_true = sum(examples) / total
    p_false = 1 - p_true

    t = p_true * np.log2(p_true) if p_true else 0
    f = p_false * np.log2(p_false) if p_false else 0

    return -(t+f)


def IG(examples, threshold):
    s1 = [e[1] for e in examples if e[0] <= threshold]
    s2 = [e[1] for e in examples if e[0] > threshold]
    ex = s1 + s2

    s1_ratio, s2_ratio = len(s1) / len(ex), len(s2) / len(ex)

    return entropy(ex) - s1_ratio * entropy(s1) - s2_ratio * entropy(s2)

class DT:
    def __init__(self, c, data, k=0):
        self.features = data.keys()
        self.c = c
        self.train = data
        self.root = Node(self.features, range(len(c)))
        self.k = k
        self.epsilon = {f: 0.1*np.std(self.train[f]) for f in self.features}

        self.ID3(self.root)

    def ID3(self, node):
        if len(node.examples) == 0:
            return

        node.classification, ratio = node.majority(self.c)
        if ratio == 1 or len(node.examples) <= self.k:
            return

        node.f, node.mid = self.select_feature(node.examples)

        s1 = [e for e in node.examples if self.train[node.f][e] <= node.mid]
        s2 = [e for e in node.examples if self.train[node.f][e] > node.mid]

        node.s1, node.s2 = Node(self.features, s1), Node(self.features, s2)
        self.ID3(node.s1)
        self.ID3(node.s2)

    def select_feature(self, examples):
        max_ig, best_f, best_mid = -1, None, None
        for f in self.features:
            vals = sorted([(self.train[f][e], self.c[e]) for e in examples], key=lambda x: x[0])
            thresholds = ([(vals[i][0] + vals[i+1][0]) / 2 for i in range(len(vals)-1)])

            for t in thresholds:
                ig = IG(vals, t)
                if ig > max_ig:
                    max_ig, best_f, best_mid = ig, f, t

        return best_f, best_mid

    def leaves(self, data, test, node):
        if node.s1 is None and node.s2 is None:
            return node.examples

        if abs(data[node.f][test] - node.mid) <= self.epsilon[node.f]:
            l1 = self.leaves(data, test, node.s1)
            l2 = self.leaves(data, test, node.s2)
            return l1 + l2
        elif data[node.f][test] <= node.mid:
            return self.leaves(data, test, node.s1)
        else:
            return self.leaves(data, test, node.s2)

    def epsilon_leaves(self, data, test):
        return self.leaves(data, test, self.root)

    def _print(self, node, space):
        if node is None:
            return
        space += 10
        self._print(node.s2, space)
        print()
        sp = " "*space
        t = sum(1 for e in node.examples if self.c[e])
        f = len(node.examples) - t
        print(sp, "(T = {}, F = {})".format(t, f))

        self._print(node.s1, space)

    def print(self):
        self._print(self.root, 0)



if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    train_data = pd.DataFrame(data)
    df_test = pd.read_csv("test.csv")
    test_data = pd.DataFrame(df_test)

    norm_train, limits = normalize_train(train_data)
    c_train = [c for c in train_data['diagnosis']]

    t9 = DT(c_train, norm_train, 9)
    # t9.print()

    norm_test = normalize_test(df_test, limits)
    c_test = [c for c in test_data['diagnosis']]
    res = [0]*len(c_test)
    # faulty = [140, 143, 160, 161]
    for t in range(len(c_test)):
    # for t in faulty:
        knn = KNN(t9.epsilon_leaves(norm_test, t), t9.c, t9.train, 9)
        res[t] = knn.classify(norm_test, t)

    correct = [t for t in range(len(c_test)) if res[t] == c_test[t]]
    wrong = [t for t in range(len(c_test)) if res[t] != c_test[t]]
    count = len(correct)
    print("Correct: {} which is {:.2%}".format(count, count/len(c_test)))









