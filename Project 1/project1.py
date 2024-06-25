# Importing the required libraries 

from string import punctuation, digits
import numpy as np
import random

# Forming a unigram (pre defined by MIT)

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

# Calculating the hinge loss of a single point for the perceptron

def hinge_loss_single(feature_vector, label, theta, theta_0):
    agreement = label*((np.dot(theta,feature_vector)) + theta_0)
    if agreement >=1:
        return 0
    else:
        return 1-agreement

# Calculating the hinge loss for the entire data set
    
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    loss=0
    for i in range(len(feature_matrix)):
        feature_vector=feature_matrix[i]
        label=labels[i]
        single_loss=hinge_loss_single(feature_vector, label, theta, theta_0)
        loss+=single_loss
    total_hinge_loss=loss/len(feature_matrix)
    return total_hinge_loss

#  Perceptron update for single point

def perceptron_single_step_update(feature_vector,label,current_theta,current_theta_0):
    value = label*(np.dot(feature_vector,current_theta)+current_theta_0)
    if value <=0:
        theta = current_theta + label*feature_vector
        theta_0 = current_theta_0 + label
    else:
        theta=current_theta
        theta_0=current_theta_0
    return theta, theta_0

# Perceptron update for entire data set, after running T times over all values

def perceptron(feature_matrix, labels, T):
    n=len(feature_matrix[0])
    theta=np.zeros(n)
    theta_0=0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta,theta_0=perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
    return theta,theta_0

# Average perceptron update for entire data set, after running T times over all values

def average_perceptron(feature_matrix, labels, T):
    n=len(feature_matrix[0])
    theta=np.zeros(n)
    theta_0=0
    sum_theta=0
    sum_theta_0=0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta,theta_0=perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            sum_theta+=theta
            sum_theta_0+=theta_0
    theta=sum_theta/(feature_matrix.shape[0]*T)
    theta_0=sum_theta_0/(feature_matrix.shape[0]*T)
    return theta,theta_0

# Pegasos update for single data point

def pegasos_single_step_update(feature_vector,label,L,eta,theta,theta_0):
    value = label*(np.dot(feature_vector,theta)+theta_0)
    if value <= 1:
        theta=(1-L*eta)*theta + eta*label*feature_vector
        theta_0=theta_0+eta*label
    else:
        theta=(1-L*eta)*theta
    return theta,theta_0

# Pegasos update for entire data set, after running T times over all values and lambda value L

def pegasos(feature_matrix, labels, T, L):
    n=len(feature_matrix[0])
    theta=np.zeros(n)
    theta_0=0
    count=1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta=1/(np.sqrt(count))
            theta,theta_0=pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,theta,theta_0)
            count+=1
    return theta,theta_0
    
# Making predictions for each data point for a normal and offset 

def classify(feature_matrix, theta, theta_0):
    classification=[]
    for i in range(len(feature_matrix)):
        value = np.dot(feature_matrix[i],theta)+theta_0
        if value>0:
            classification.append(1)
        else:
            classification.append(-1)
    return np.array(classification)

# Finding a classifer's accuracy on testing data after fitting on training data
# **kwargs are needed for the lambda paramter (L) from pegasos

def classifier_accuracy(classifier,train_feature_matrix,val_feature_matrix,train_labels,val_labels,**kwargs):
    # Finding normal and offset for a given classifier and finding it's training accuracy
    theta_train,theta_0_train=classifier(train_feature_matrix,train_labels,**kwargs)
    preds_train=classify(train_feature_matrix, theta_train, theta_0_train)
    mean_accuracy_train=accuracy(preds_train,train_labels)
    # Finding the accuracy on testng data
    preds_val=classify(val_feature_matrix,theta_train,theta_0_train)
    mean_accuracy_test=accuracy(preds_val,val_labels)
    return mean_accuracy_train,mean_accuracy_test

# Helper function for bag_of_words (pre defined by MIT)

def extract_words(text):
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()

# Converting text data into a matrix (pre defined by MIT)
# stopwords is a list of words that can be removed based on need (feature engineering #1)

with open("stopwords.txt",'r') as file:
    data=file.read()
stopword=extract_words(data)

def bag_of_words(texts, remove_stopword=True):    
    indices_by_word = {}
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word

# One hot encoding, presence of word =1 and absence of word=0 (pre defined by MIT)
# Another way of encoding is to map each word to it's count in the text (feature engineering #2)

def extract_bow_feature_vectors(reviews, indices_by_word, binarize=False):
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        for i in range(feature_matrix.shape[0]):
            for j in range(feature_matrix.shape[1]):
                if feature_matrix[i,j]>0:
                    feature_matrix[i,j]=1
                else:
                    feature_matrix[i,j]=0
    return feature_matrix

# Finding the accuracy of a prediction (pre defined by MIT)

def accuracy(preds, targets):
    return (preds == targets).mean()