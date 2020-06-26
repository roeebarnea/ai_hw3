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

def IG_dynamic(examples_val, tj):
    total_e = len(examples_val)
    count_s1, count_s2 = 0, 0
    count_s1_t, count_s1_f, count_s2_t, count_s2_f = 0, 0, 0, 0
    H1, H2 = 0, 0
    for e in examples_val:
        if e[0] < tj:
            count_s1 += 1
            if e[1] == 1:
                count_s1_t += 1
            else:
                count_s1_f += 1
        else:
            count_s2 += 1
            if e[1] == 1:
                count_s2_t += 1
            else:
                count_s2_f += 1
    if count_s1_t == 0 or count_s1_f == 0 or count_s1 == 0:
        H1 = 0
    else:
        rat_s1_t = count_s1_t / count_s1
        rat_s1_f = count_s1_f / count_s1
        H1 = -(rat_s1_t * np.log2(rat_s1_t) + rat_s1_f * np.log2(rat_s1_f))
    if count_s2_t == 0 or count_s2_f == 0 or count_s2 == 0:
        H2 = 0
    else:
        rat_s2_t = count_s2_t / count_s2
        rat_s2_f = count_s2_f / count_s2
        H2 = -(rat_s2_t * np.log2(rat_s2_t) + rat_s2_f * np.log2(rat_s2_f))
    IG = 1 - ((count_s1 / total_e) * H1 + (count_s2 / total_e) * H2)
    return IG

def best_IG_mid(f, examples, df):
    examples_val = []
    best_IG = -1
    best_tj = 0
    for e in examples:
        examples_val.append((df[f][e], df["diagnosis"][e]))
    examples_val.sort()
    k = len(examples)
    for j in range(0,k-1):
        tj = (examples_val[j][0]+examples_val[j+1][0])/2
        IG = IG_dynamic(examples_val, tj)
        if IG > best_IG:
            best_IG = IG
            best_tj = tj
    return best_IG, best_tj


# select feature with the best IG, return the function and his mid value
def selectFeature(features, examples, df):
    total_e = len(examples)
    max_IG, best_f, best_mid= -1, None, None
    for f in features:
        IG, mid = best_IG_mid(f, examples, df)
        if IG > max_IG:
            max_IG = IG
            best_f = f
            best_mid = mid
    return best_f, best_mid

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
# return the new Node
def new_node_s1(examples, features, f, mid):
    new_examples = []
    for e in examples:
        if df[f][e] < mid:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples)

# build new node to s2, when know the best feature to classify with.
# gets the features of his father so need to cut the best feature of it.
# return the new Node
def new_node_s2(examples, features, f, mid):
    new_examples = []
    for e in examples:
        if df[f][e] >= mid:
            new_examples.append(e)
    new_features = features.drop([f])
    return Node(new_features, new_examples)

# pretty much the class TDIDT algorithm
def ID3(cur_node, df):
    if len(cur_node.examples) == 0:
        return
    c = majorityClass(cur_node.examples, df)
    cur_node.classification = c[0]
    if c[1] == 1 or len(cur_node.features) == 0:
        return
    res_SF = selectFeature(cur_node.features, cur_node.examples, df)
    cur_node.f ,cur_node.mid = res_SF[0], res_SF[1]
    cur_node.s1 = new_node_s1(cur_node.examples, cur_node.features, cur_node.f, cur_node.mid)
    cur_node.s2 = new_node_s2(cur_node.examples, cur_node.features, cur_node.f, cur_node.mid)
    ID3(cur_node.s1, df)
    ID3(cur_node.s2, df)

# build decision tree ID3 with the data df
def get_classifier_ID3(df):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])))
    tree = Node(features, examples)
    ID3(tree,df)
    return tree

# pretty much the class TDIDT algorithm but when there examples <= k, the node turn to leaf
def ID3_k(cur_node, df, k):
    if len(cur_node.examples) == 0:
        return
    c = majorityClass(cur_node.examples, df)
    cur_node.classification = c[0]
    if c[1] == 1 or len(cur_node.features) == 0 or len(cur_node.examples) <= k :
        return
    res_SF = selectFeature(cur_node.features, cur_node.examples, df)
    cur_node.f ,cur_node.mid = res_SF[0], res_SF[1]
    cur_node.s1 = new_node_s1(cur_node.examples, cur_node.features, cur_node.f, cur_node.mid)
    cur_node.s2 = new_node_s2(cur_node.examples, cur_node.features, cur_node.f, cur_node.mid)
    ID3_k(cur_node.s1, df, k)
    ID3_k(cur_node.s2, df, k)

# build decision tree ID3 with the k thing with the data df
def get_classifier_ID3_k(df, k):
    features = df.keys().drop(['diagnosis'])
    examples = list(range(0, len(df['diagnosis'])))
    tree = Node(features, examples)
    ID3_k(tree, df, k)
    return tree

# check for the example i if is right in the tree
def is_right_answer(tree, df_test, i):
    if tree.s1 == None and tree.s1 == None:
        return  df_test['diagnosis'][i] == tree.classification
    if df_test[tree.f][i] < tree.mid:
        return is_right_answer(tree.s1, df_test, i)
    else:
        return is_right_answer(tree.s2, df_test, i)

# check all examples in the tree and return the ratio of accuracy
def get_classifier_accuracy(tree, df_test):
    examples = list(range(0, len(df_test['diagnosis'])))
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