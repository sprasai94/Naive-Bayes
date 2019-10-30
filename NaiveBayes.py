import random
import math as m
import numpy as np
import operator
import matplotlib.pyplot as plt


def mean(x):
    return sum(x)/(len(x))


def standard_deviation(x):
    avg = mean(x)
    sd = m.sqrt(sum([pow(i - avg, 2) for i in x]) / float(len(x) - 1))
    return sd


def readdatafile(filename):
    with open(filename, 'r') as ifl:
        array = [l.strip().split(',') for l in ifl]
        dataset = list(array)[:-1]
        #shuflle the row of instances in dataset
        random.shuffle(dataset)

        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])

    return dataset


#Make a dictionary with key as classes name and value as list of instances of that class
def group_by_class(dataset):
    dict_class = {}
    for i in range(len(dataset)):
        row = dataset[i]
        #last element of Instance is class of actual dataset
        target_class = row[-1]
        if (target_class not in dict_class):
            dict_class[target_class] = []
        dict_class[target_class].append(row[:-1])
    return dict_class


#For each class find the mean and SD of each features
#Here key is class name and vlaue has four list as [mean,sd], for each instances
def mean_Sd_class( dict_class):
    dict_mean_SD = {}
    for target, features in dict_class.items():
        dict_mean_SD [target] = [(mean(attributes), standard_deviation(attributes)) for attributes in zip(*features)]
    return dict_mean_SD


#Find the probablity from normal distribution
def gaussian_pdf(x, mean, sd):
    e = m.exp(-(m.pow(x-mean,2) / (2*m.pow(sd,2))))
    denominator = m.sqrt(2* m.pi) * sd
    pdf = e / denominator
    return pdf


#Finding likelihood of test data for each classes
#mathematical implementation of probablity density function of normal distribution
def joint_class_probablities(dict_mean_sd, dict_class, test_data, train_data):
    joint_prob = {}
    for target, features in dict_mean_sd.items():
        total_features = len(features)
        likelihood = 1
        for i in range(total_features):
            x = test_data[i]
            mean, sd = features[i]
            normal_prob = gaussian_pdf(x, mean, sd)
            likelihood *= normal_prob
        joint_prob[target] = likelihood
    return joint_prob


#class wtih maximum joint probability is considered as predicted class
#returns the class prediction for each testdata
def classification_function(dict_mean_sd, dict_class,  test_data, train_data):
    predictions = []
    for i in range(len(test_data)):
        probablities = joint_class_probablities(dict_mean_sd, dict_class, test_data[i], train_data)
        prediction = max(probablities, key=probablities.get)
        predictions.append(prediction)
    return predictions


#Finds the accuracy of the predicted result comparing with real test data
def find_accuracy(test_data, predictions):
    correct = 0
    # last element of test set is real classification,
    # so making list of true classification
    actual = [instances[-1] for instances in test_data]
    # comparing each values in list if they are similar
    for x, y in zip(actual, predictions):
        if x == y:
            correct += 1
    accuracy = (correct/float(len(actual))) * 100
    return accuracy


#Generate a confusion matrix from predicted and actual classification
def find_confusion_matrix(predictions,actual):
    actual_class = []
    for x in range(len(actual)):
        test = actual[x][-1]
        actual_class.append(test)
    num_of_classes = len(set(actual_class))
    #replacing the class name with integer value
    p = list(map(lambda x: 0 if x == "Iris-setosa" else x, predictions))
    p = list(map(lambda x: 1 if x == "Iris-versicolor" else x, p))
    p = list(map(lambda x: 2 if x == "Iris-virginica" else x, p))

    a = list(map(lambda x: 0 if x == "Iris-setosa" else x, actual_class))
    a = list(map(lambda x: 1 if x == "Iris-versicolor" else x, a))
    a = list(map(lambda x: 2 if x == "Iris-virginica" else x, a))

    confusion_matrix = np.zeros((num_of_classes,num_of_classes))
    for i in range(len(actual_class)):
        confusion_matrix[p[i]][a[i]] +=1
    #print('confusion matrix:',confusion_matrix)
    return confusion_matrix

#Function to plot the confusion matrix
def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):

    # Compute confusion matrix
    # Find out the unique classes
    y_true = []
    for x in range(len(test_set)):
        test = test_set[x][-1]
        y_true.append(test)
    classes = list(np.unique(list(y_true)))

    #print('Confusion matrix (without normalization):')
    #print(classes)
    #print(cm)
    if cmap is None:
        cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()

def avg_cm(x_list):
    sum = 0
    for x in x_list:
        sum = sum + x
    return sum/len(x_list)

#split the dataset for 5 fold cross validation and does the classification of test data
def naive_bayes_cross_validation(dataset, cv):
    total_sample = len(dataset)
    num_TestSample = int((1 / cv) * total_sample)
    t1 = 0
    t2 = num_TestSample
    accuracies = []
    confusion_matrices = []
    for i in range(cv):
        #splitting tha dataset in test and training sample
        test_set = dataset[t1:t2]
        train_set = dataset[0:t1] + dataset[t2:total_sample]
        t1 = t1 + num_TestSample
        t2 = t2 + num_TestSample

        dict_class = group_by_class(train_set)
        dict_mean_sd = mean_Sd_class(dict_class)
        predictions = classification_function(dict_mean_sd, dict_class, test_set, train_set)

        confusion_matrix = find_confusion_matrix(predictions, test_set)
        confusion_matrices.append(confusion_matrix)
        accuracy = find_accuracy(test_set, predictions)
        accuracies.append(accuracy)


    average_confusion_matrix = avg_cm(confusion_matrices)
    print('Accuracies:', accuracies)
    average_accuracy = sum(accuracies) / len(accuracies)
    print('Average accuracy:', average_accuracy)
    print('Best confusion matrix:', average_confusion_matrix)
    plotConfusionMatrix(test_set, predictions, average_confusion_matrix,
                        normalize=True,
                        title=None,
                        cmap=None, plot=True)


if __name__== "__main__":
    #Read Input Dataset
   dataset = readdatafile('iris.data')
   cv = 5
   naive_bayes_cross_validation(dataset, cv)


