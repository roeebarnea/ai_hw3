import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, features, examples):
        self.s1 = None
        self.s2 = None
        self.classification = None #should be to majority
        self.f = None #if not a leaf should conatain the feature which devide his sons
        self.mid = None # the value which under it is s1 and above is s2
        self.features = features # the features who left. Important Type = Index
        self.examples = examples # current examples. Important Type = List


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


# select feature with the best IG, return the function and its mid value
def selectFeature(features, examples, df):
    max_ig, best_f, best_mid = -1, None, None
    for f in features:
        # ig, mid = best_IG_mid(f, examples, df)
        vals = sorted([(df[f][e], df['diagnosis'][e]) for e in examples], key=lambda x: x[0])
        thresholds = [(vals[i][0] + vals[i + 1][0]) / 2 for i in range(len(vals) - 1)]

        for t in thresholds:
            ig = IG(vals, t)
            if ig > max_ig:
                max_ig = ig
                best_f = f
                best_mid = t

    return best_f, best_mid


# return the majority of the classification of examples and the ratio(to know if all one sided class)
def majorityClass(examples, df):
    total = len(examples)
    true = sum(1 for e in examples if df['diagnosis'][e] == 1)

    if true >= total - true:
        return 1, true/total
    else:
        return 0, (total-true)/total


def ID3(cur_node, df, k):
    if len(cur_node.examples) == 0:
        return
    c, ratio = majorityClass(cur_node.examples, df)
    cur_node.classification = c
    if ratio == 1 or len(cur_node.examples) <= k:
        return

    cur_node.f, cur_node.mid = selectFeature(cur_node.features, cur_node.examples, df)

    s1 = [e for e in cur_node.examples if df[cur_node.f][e] <= cur_node.mid]
    s2 = [e for e in cur_node.examples if df[cur_node.f][e] > cur_node.mid]

    cur_node.s1, cur_node.s2 = Node(cur_node.features, s1), Node(cur_node.features, s2)
    ID3(cur_node.s1, df, k)
    ID3(cur_node.s2, df, k)


# build decision tree ID3 with the k thing with the data df
def get_classifier_ID3(df, k=0):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])))
    tree = Node(features, examples)
    ID3(tree, df, k)
    return tree


# check for the example i if is right in the tree
def is_right_answer(tree, df_test, i):
    if tree.s1 is None and tree.s2 is None:
        return df_test['diagnosis'][i] == tree.classification

    if df_test[tree.f][i] <= tree.mid:
        return is_right_answer(tree.s1, df_test, i)
    else:
        return is_right_answer(tree.s2, df_test, i)


# check all examples in the tree and return the ratio of accuracy
def get_classifier_accuracy(tree, df_test):
    examples = list(range(0, len(df_test['diagnosis'])))
    total = len(df_test['diagnosis'])

    count_t = sum(1 for e in examples if is_right_answer(tree, df_test, e))
    return count_t/total


def leaves(tree, epsilon, test, train, e):
    if tree.s1 is None and tree.s2 is None:
        t = sum(1 for ex in tree.examples if train['diagnosis'][ex])
        return t, len(tree.examples) - t

    if abs(test[tree.f][e] - tree.mid) <= epsilon[tree.f]:
        t1, f1 = leaves(tree.s1, epsilon, test, train, e)
        t2, f2 = leaves(tree.s2, epsilon, test, train, e)
        return t1+t2, f1+f2
    elif test[tree.f][e] <= tree.mid:
        return leaves(tree.s1, epsilon, test, train, e)
    else:
        return leaves(tree.s2, epsilon, test, train, e)


def epsilon_classify(tree, epsilon, test, train, e):
    t, f = leaves(tree, epsilon, test, train, e)
    return 1 if t >= f else 0


def epsilon_accuracy(tree, test, train):
    examples = list(range(0, len(test['diagnosis'])))
    features = test.keys().drop(['diagnosis'])
    eps = {f: 0.1 * np.std(train[f]) for f in features}

    matches = sum(1 if epsilon_classify(tree, eps, test, train, e) == test['diagnosis'][e] else 0 for e in examples)
    return matches / len(examples)


if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    df = pd.DataFrame(data)
    tree = get_classifier_ID3(df)

    data_test = pd.read_csv("test.csv")
    df_test = pd.DataFrame(data_test)

    accuracy = get_classifier_accuracy(tree, df_test)
    accuracy_train = get_classifier_accuracy(tree, df)

    print("the accuracy of ID3 tree on train is: {:.2%}".format(accuracy_train))
    print("ID3 (No pruning): {:.2%}".format(accuracy))

    ks = [3, 9, 27]
    accuracies =[]
    for k in ks:
        t = get_classifier_ID3(df, k)
        a = get_classifier_accuracy(t, df_test)
        print("ID3 (k = {}): {:.2%}".format(k, a))
        accuracies.append(a*100)
    print(accuracies)

    fig, ax = plt.subplots()
    ax.scatter(ks, accuracies)
    ax.set_ylabel('Accuracy %')
    ax.set_xlabel('k val')

    plt.xticks(range(0, 28, 3))
    plt.yticks(np.arange(92, 94, 0.5))
    plt.show()

    t9 = get_classifier_ID3(df, 9)
    eps_acc = epsilon_accuracy(t9, df_test, df)
    print("T9 epsilon accuracy: {:.2%}".format(eps_acc))
