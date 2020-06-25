import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# return for the train data 2 dictionaries feature->min_val feature->max_val
def normalize_train_data(train_data):
    features = train_data.keys().drop(['diagnosis'])
    features_min, features_max = {}, {}
    for f in features:
        min = train_data[f].min()
        max = train_data[f].max()
        features_min[f] = min
        features_max[f] = max
    return features_min, features_max

# for the data, 2 dictionaries , f = feature , e = example
# return the normalized val
def get_val(data, features_min, features_max, f, e):
    fo = data[f][e]
    return (fo - features_min[f]) / (features_max[f] - features_min[f])

# return the dist between normalized e_test vector and e_train vector
def distance(train_data, test_data, features_min, features_max, e_test, e_train):
    features = test_data.keys().drop(['diagnosis'])
    sum = 0
    for f in features:
        val_test = get_val(test_data, features_min, features_max, f, e_test)
        val_train = get_val(train_data, features_min, features_max, f, e_train)
        sum += np.square(np.abs(val_test - val_train))
    return np.sqrt(sum)

# check for e_test example the distance of all the e_train examples, takes the lowest k
# from these k takes return the majority class (if equal return true)
def classify_example_knn(train_data, test_data, features_min, features_max, e_test, k):
    dict_dist_to_class, dist = {}, []
    for e_train in range(0, len(train_data['diagnosis'])):
        d =distance(train_data, test_data, features_min, features_max, e_test, e_train)
        dist.append(d)
        dict_dist_to_class[d] = train_data['diagnosis'][e_train]
    dist.sort()
    t, f = 0, 0
    for j in range(k):
        if dict_dist_to_class[dist[j]] == 1:
            t+= 1
        else:
            f+= 1
    if t>=f:
        return 1
    else:
        return 0

# check for each test example the KNN classification
# count the number of the true classification
# return the accuracy ratio
def knn_k(train_data, test_data, k):
    features_min, features_max= normalize_train_data(train_data)
    good = 0
    bad = 0
    for e_test in range(0, len(test_data['diagnosis'])):
        c = classify_example_knn(train_data, test_data, features_min, features_max, e_test, k)
        if c == test_data['diagnosis'][e_test]:
            good += 1
        else:
            bad += 1
    return  good/(good+bad)

if __name__ == "__main__":


    data = pd.read_csv("train.csv")
    train_data = pd.DataFrame(data)
    df_test = pd.read_csv("test.csv")
    test_data = pd.DataFrame(df_test)
    accuracy_9 = knn_k(train_data, test_data, 9)


    print("the accuracy of KNN9 tree is: ", accuracy_9)

    ks = [1,3,9,27]
    accuracies =[]
    for k in ks:
        a = knn_k(train_data, test_data, k)
        accuracies.append(a)
    print(accuracies)
    plt.scatter(ks,accuracies)
    plt.show()


