import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, features, examples):
        self.s1 = None
        self.s2 = None
        self.classification = None
        self.f = None
        self.mid = None
        self.features = features
        self.examples = examples

def selectFeature(features, examples, df):
    total_e = len(examples)
    max_IG, best_f, best_mid= -1, None, None
    # if (len(features) == 0):
    #     print("ERROR")
    for f in features:
        mid = (df[f].max()-df[f].min())/2
        count_s1, count_s2 = 0, 0
        count_s1_t, count_s1_f, count_s2_t, count_s2_f = 0, 0, 0, 0
        H1, H2 = 0, 0
        for e in examples:
            if df[f][e] < mid:
                count_s1 += 1
                if df["diagnosis"][e] == 1:
                    count_s1_t += 1
                else:
                    count_s1_f += 1
            else:
                count_s2 += 1
                if df["diagnosis"][e] == 1:
                    count_s2_t += 1
                else:
                    count_s2_f += 1
        if count_s1_t == 0 or count_s1_f == 0 or count_s1 == 0:
            H1 =0
        else:
            rat_s1_t = count_s1_t / count_s1
            rat_s1_f = count_s1_f / count_s1
            H1 = -(rat_s1_t * np.log2(rat_s1_t) + rat_s1_f * np.log2(rat_s1_f))
        if count_s2_t == 0 or count_s2_f == 0 or count_s2 == 0:
            H2 =0
        else:
            rat_s2_t = count_s2_t / count_s2
            rat_s2_f = count_s2_f / count_s2
            H2 = -(rat_s2_t * np.log2(rat_s2_t) + rat_s2_f * np.log2(rat_s2_f))

        IG = 1 - ( (count_s1/total_e)*H1 + (count_s2/total_e)*H2 )
        if IG > max_IG:
            max_IG = IG
            best_f = f
            best_mid = mid
    return best_f, best_mid

def majorityClass(examples, df):
    total = len(examples)
    count_t, count_f = 0, 0
    for e in examples:
        if df["diagnosis"][e] == 1:
            count_t += 1
        else:
            count_f += 1
    if count_t >= count_f:
        return 1 , count_t/total
    else:
        return 0 , count_f/total

def new_node_s1(examples, features, f):
    new_examples = []
    # print(f)
    mid = (df[f].max() - df[f].min()) / 2
    for e in examples:
        if df[f][e] < mid:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples)

def new_node_s2(examples, features, f):
    new_examples = []
    mid = (df[f].max() - df[f].min()) / 2
    for e in examples:
        if df[f][e] >= mid:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples)

def ID3(cur_node, df):
    if len(cur_node.examples) == 0:
        return
    c = majorityClass(cur_node.examples, df)
    cur_node.classification = c[0]
    if c[1] == 1 or len(cur_node.features) == 0:
        return
    res_SF = selectFeature(cur_node.features, cur_node.examples, df)
    cur_node.f ,cur_node.mid = res_SF[0], res_SF[1]
    cur_node.s1 = new_node_s1(cur_node.examples, cur_node.features, cur_node.f)
    cur_node.s2 = new_node_s2(cur_node.examples, cur_node.features, cur_node.f)
    ID3(cur_node.s1, df)
    ID3(cur_node.s2, df)

def get_classifier_ID3(df):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])-1))
    tree = Node(features, examples)
    ID3(tree,df)
    return tree

def ID3_k(cur_node, df, k):
    if len(cur_node.examples) == 0:
        return
    c = majorityClass(cur_node.examples, df)
    cur_node.classification = c[0]
    if c[1] == 1 or len(cur_node.features) == 0 or len(cur_node.examples) <= k :
        return
    res_SF = selectFeature(cur_node.features, cur_node.examples, df)
    cur_node.f ,cur_node.mid = res_SF[0], res_SF[1]
    cur_node.s1 = new_node_s1(cur_node.examples, cur_node.features, cur_node.f)
    cur_node.s2 = new_node_s2(cur_node.examples, cur_node.features, cur_node.f)
    ID3_k(cur_node.s1, df, k)
    ID3_k(cur_node.s2, df, k)

def get_classifier_ID3_k(df, k):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])-1))
    tree = Node(features, examples)
    ID3_k(tree, df, k)
    return tree

def is_right_answer(tree, df_test, i):
    if tree.s1 == None and tree.s1 == None:
        return  df_test['diagnosis'][i] == tree.classification
    # print(tree.f)
    # print(i)
    if df_test[tree.f][i] < tree.mid:
        return is_right_answer(tree.s1, df_test, i)
    else:
        return is_right_answer(tree.s2, df_test, i)

def get_classifier_accuracy(tree, df_test):
    examples = list(range(0, len(df_test['diagnosis']) - 1))
    count_t, count_f = 0, 0
    total = len(df_test['diagnosis'])
    for e in examples:
        ans = is_right_answer(tree, df_test, e)
        if (ans):
            count_t +=1
        else:
            count_f +=1
    return count_t/total


if __name__ == "__main__":
    # data = pd.read_csv("test.csv")
    # df = pd.DataFrame(data)
    # keys = df.keys().copy()
    # #keys.delete()
    # l =keys.drop(['diagnosis'])
    # print(l)
    # print(keys)
    # print(type(keys))
    # print(df['area_mean'][0])
    # t_max = df['area_mean'].max()
    # t_min = df['area_mean'].min()
    # print(t_max, " ", t_min)
    # print(len(df['area_mean']))

    data = pd.read_csv("train.csv")
    df = pd.DataFrame(data)
    tree = get_classifier_ID3(df)
    data_test = pd.read_csv("test.csv")
    df_test = pd.DataFrame(data_test)
    accuracy = get_classifier_accuracy(tree, df_test)
    accuracy_train = get_classifier_accuracy(tree, df)

    print("the accuracy of ID3 tree is: ", accuracy)
    print("the accuracy of ID3 tree on train is: ", accuracy_train)

    ks = [3,9,27]
    accuracies =[]
    for k in ks:
        t = get_classifier_ID3_k(df, k)
        a = get_classifier_accuracy(t, df_test)
        accuracies.append(a)
    print(accuracies)
    plt.scatter(ks,accuracies)
    plt.show()