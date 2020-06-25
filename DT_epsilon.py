import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, features, examples, d):
        self.s1 = None
        self.s2 = None
        self.classification = None #should be to majority
        self.f = None #if not a leaf should conatain the feature which devide his sons
        self.std = None # the value which under it is s1 and above is s2, here the std of the feature examples value
        self.features = features # the features who left. Important Type = Index
        self.examples = examples # current examples. Important Type = List
        self.d = d # depth

# select feature with the best IG, return the function and his mid value
def selectFeature(features, examples, df):
    total_e = len(examples)
    max_IG, best_f, best_std= -np.Inf, None, None
    # if (len(features) == 0):
    #     print("ERROR")
    for f in features:
        std = np.std(df[f])
        count_s1, count_s2 = 0, 0
        count_s1_t, count_s1_f, count_s2_t, count_s2_f = 0, 0, 0, 0
        H1, H2 = 0, 0
        for e in examples:
            if df[f][e] < std:
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
            best_std = std
    return best_f, best_std

#return the majority of the classification of examples and the ratio(to know if all one sided class)
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


# build new node to s1, when know the best feature to classify with.
# gets the features of his father so need to cut the best feature of it.
# get the depth of the Node.
# return the new Node
def new_node_s1(examples, features, f, d):
    new_examples = []
    # print(f)
    std = np.std(df[f])
    for e in examples:
        if df[f][e] < std:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples, d)


# build new node to s2, when know the best feature to classify with.
# gets the features of his father so need to cut the best feature of it.
# get the depth of the Node.
# return the new Node
def new_node_s2(examples, features, f, d):
    new_examples = []
    std = np.std(df[f])
    for e in examples:
        if df[f][e] >= std:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples, d)

# pretty much the class TDIDT algorithm but the mid value is std, and it stops in max_depth (become a leaf)
def ID3_maxDepth(cur_node, df, max_depth):
    if len(cur_node.examples) == 0:
        return
    c = majorityClass(cur_node.examples, df)
    cur_node.classification = c[0]
    if c[1] == 1 or len(cur_node.features) == 0 or max_depth == cur_node.d:
        return
    res_SF = selectFeature(cur_node.features, cur_node.examples, df)
    cur_node.f ,cur_node.std = res_SF[0], res_SF[1]
    cur_node.s1 = new_node_s1(cur_node.examples, cur_node.features, cur_node.f, cur_node.d+1)
    cur_node.s2 = new_node_s2(cur_node.examples, cur_node.features, cur_node.f, cur_node.d+1)
    ID3_maxDepth(cur_node.s1, df, max_depth)
    ID3_maxDepth(cur_node.s2, df, max_depth)

# build decision tree ID3 with max depth and mid value as std with the data df
def get_classifier_ID3(df, max_depth):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])))
    tree = Node(features, examples, 0)
    ID3_maxDepth(tree, df, max_depth)
    return tree

# check for the example i if is right in the tree with the new rules
# it sums all the leafs he gets 1 as 1 and 0 as -1
# so if the sum is less then 0 the class is 0, else the class is 1
def right_ratio_eps(tree, df_test, i):
    if tree.s1 == None and tree.s1 == None:
        if tree.classification == 1:
            return 1
        else:
            return -1
    eps = tree.std*0.1
    if abs(df_test[tree.f][i] - tree.std) <= eps:
        return right_ratio_eps(tree.s1, df_test, i) + right_ratio_eps(tree.s2, df_test, i)
    else:
        if df_test[tree.f][i] < tree.std:
            return right_ratio_eps(tree.s1, df_test, i)
        else:
            return right_ratio_eps(tree.s2, df_test, i)

# check all examples in the tree and return the ratio of accuracy
def get_classifier_accuracy_eps(tree, df_test):
    examples = list(range(0, len(df_test['diagnosis'])))
    count_t, count_f = 0, 0
    total = len(df_test['diagnosis'])
    for e in examples:
        rat = right_ratio_eps(tree, df_test, e)
        ans = 1
        if rat < 0:
            ans = 0
        if (ans == df_test['diagnosis'][e]):
            count_t +=1
        else:
            count_f +=1
    return count_t/total


if __name__ == "__main__":


    data = pd.read_csv("train.csv")
    df = pd.DataFrame(data)
    tree = get_classifier_ID3(df, 9)
    data_test = pd.read_csv("test.csv")
    df_test = pd.DataFrame(data_test)
    accuracy = get_classifier_accuracy_eps(tree, df_test)


    print("the accuracy of T9 tree is: ", accuracy)


    # ks = [3,9,27]
    # accuracies =[]
    # for k in ks:
    #     t = get_classifier_ID3_k(df, k)
    #     a = get_classifier_accuracy(t, df_test)
    #     accuracies.append(a)
    # print(accuracies)
    # plt.scatter(ks,accuracies)
    # plt.show()