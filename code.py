from os import path, walk, listdir
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

test_dir = 'Test'
train_dir = 'Train'


def check_file(filename):
    print(path.exists(filename))
    return path.exists(filename)


def count_emails(directory):
    number = 0
    emails_dirs = [path.join(directory, f) for f in listdir(directory)]
    for emails_dir in emails_dirs:
        dirs = [path.join(emails_dir, f) for f in listdir(emails_dir)]
        for d in dirs:
            emails = [path.join(d, f) for f in listdir(d)]
            number += len(emails)

    return number


def create_dictionary(directory):
    all_words = []
    for directories, subdirs, files in walk(directory):
        for filename in files:
            print(filename)
            with open(path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                all_words += word_tokenize(data)

    filtered_words = []
    stop_words = set(stopwords.words('english'))

    for word in all_words:
        if word not in stop_words:
            filtered_words.append(word)

    dictionary = Counter(filtered_words)

    for item in list(dictionary):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(1000)

    return dictionary


def extract_features_labels(directory, dictionary):
    number_emails = count_emails(directory)
    features_matrix = np.zeros((number_emails, 1000))
    labels = np.zeros(number_emails)
    mail_id = 0

    for directories, subdirs, files in walk(directory):
        for filename in files:
            with open(path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                words = word_tokenize(data)

                for word in words:
                    word_id = 0
                    for i, d in enumerate(dictionary):
                        if d[0] == word:
                            word_id = i
                            features_matrix[mail_id, word_id] = words.count(word)

                if (path.split(directories)[1] == 'ham'):
                    labels[mail_id] = 0
                else:
                    labels[mail_id] = 1

                print(filename, labels[mail_id])

                mail_id += 1

    return features_matrix, labels


if check_file('ref_dict.npy'):
    ref_dict = np.load('ref_dict.npy')

else:
    ref_dict = create_dictionary(train_dir)
    np.save('ref_dict.npy', ref_dict)

if check_file('train_matrix.npy') and check_file('train_labels.npy'):
    train_matrix = np.load('train_matrix.npy')
    train_labels = np.load('train_labels.npy')
else:
    train_matrix, train_labels = extract_features_labels(train_dir, ref_dict)
    np.save('train_matrix.npy', train_matrix)
    np.save('train_labels.npy', train_labels)

if check_file('test_matrix.npy') and check_file('test_labels.npy'):
    test_matrix = np.load('test_matrix.npy')
    test_labels = np.load('test_labels.npy')
else:
    test_matrix, test_labels = extract_features_labels(test_dir, ref_dict)
    np.save('test_matrix.npy', test_matrix)
    np.save('test_labels.npy', test_labels)


model2 = MultinomialNB()

model2.fit(train_matrix, train_labels)

result2 = model2.predict(test_matrix)

print(result2)

print("Accuracy for MultinomialNB is ", accuracy_score(test_labels, result2) * 100)


clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=2, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

clf.fit(train_matrix, train_labels)

resultSVM = clf.predict(test_matrix)

print("Accuracy for SVM model is ", accuracy_score(test_labels, resultSVM) * 100)
