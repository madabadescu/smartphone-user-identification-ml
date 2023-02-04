import csv
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import metrics, svm, preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from tensorflow.keras.optimizers import SGD, Adam

# read train labels
with open("./pml-2022-smart/train_labels.csv", 'r') as file:
    reader = csv.reader(file)
    i = 0
    train_label = []
    for row in reader:
        if i > 0:
            train_label.append(row)
        i += 1
print("1")

# read train data
train_data = []
row_count = []
for label in train_label:
    with open(f"./pml-2022-smart/train/train/{label[0]}.csv", 'r') as file:
        reader = csv.reader(file)
        row_data = []
        count = 0
        for row in reader:
            row_arr = [float(row[0]), float(row[1]), float(row[2])]
            row_data.append(row_arr)
            count += 1
        train_data.append(row_data)
        row_count.append(count)

# counting the mean of numbers of recordings
mean_count = round(np.mean(row_count))

print("2")


# make all the data to have the same size (for the smaller ones, add the mean as necessary; for the bigger ones,
# delete the last values)
for data in train_data:
    while len(data) != mean_count:
        if len(data) > mean_count:
            data.pop(len(data)-1)
        else:
            data.append(tuple([np.mean([coord[0] for coord in data]), np.mean([coord[1] for coord in data]), np.mean([coord[2] for coord in data])]))
print("3")

# keep just the second element of each train_label tuple
for i in range(len(train_label)):
    train_label[i] = train_label[i][1]

# flatten the data
train_data_flat = []
for data in train_data:
    data_flat = np.array(data)
    data_flat = data_flat.flatten()
    train_data_flat.append(data_flat)
print("4")

# scale the data
scaler = preprocessing.StandardScaler()
scaler.fit(train_data_flat)

train_data_scaled = scaler.transform(train_data_flat)
print("5")
"""
# separate the data into train an test in order to calculate score and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled, train_label, test_size=0.2, random_state=7)

X_train, y_train, X_test, y_test  = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
"""



"""


# example of classifier with SVM
classifier = svm.SVC()
classifier.fit(X_train, y_train)

score1 = classifier.score(X_test, y_test)
print("scor SVM cu default", score1)

scores1 = cross_val_score(classifier, X_test, y_test, scoring='accuracy')
print("default accuracy", np.mean(scores1))

"""

"""
# example of classifier with linear discriminant analysis

clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
clf.fit(X_train, y_train)
print("gata LDA")

score2 = clf.score(X_test, y_test)
print("scor LDA", score2)

scores1 = cross_val_score(clf, X_test, y_test, scoring='accuracy')
print("lsqr auto shrinkage accuracy", np.mean(scores1))

"""

"""
# example of format for confusion matrix
predictions = clf.predict(X_test)

cm = metrics.confusion_matrix(y_test, predictions)
disp1 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

disp1.plot()

plt.show()
"""


# transform the labels into floats for the model
for i in range(len(train_label)):
    train_label[i] = float(train_label[i])

# separate the data into train an test in order to calculate score and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(train_data_scaled, train_label, test_size=0.1, random_state=7)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# apply the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(21, activation='softmax')
])

# compile the model
optimizer = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

print("gata NN")


# making the predictions for submission
predictions_data = []
names = []
# reading the test files names and writing the information in predictions_data
for path in os.scandir('./pml-2022-smart/test/test/'):
    names.append(os.path.basename(path).split('.')[0])

    with open(path, 'r') as file:
        reader = csv.reader(file)
        row_data = []
        for row in reader:
            row_arr = [float(row[0]), float(row[1]), float(row[2])]
            row_data.append(row_arr)
            count += 1
        predictions_data.append(row_data)

# preprocessing of the predictions data in the same manner as for the training data
for data in predictions_data:
    while len(data) != mean_count:
        if len(data) > mean_count:
            data.pop(len(data)-1)
        else:
            data.append(tuple([np.mean([coord[0] for coord in data]), np.mean([coord[1] for coord in data]), np.mean([coord[2] for coord in data])]))

predictions_data_flat = []
for data in predictions_data:
    data_flat = np.array(data)
    data_flat = data_flat.flatten()
    predictions_data_flat.append(data_flat)

predictions_data_scaled = scaler.transform(predictions_data_flat)

# making the predictions (this example is for the NN model)
predictions = []

predictions_first = model.predict(predictions_data_scaled)
for prediction in predictions_first:
    predictions.append(np.argmax(prediction))

# preparing the files for submission
predictions_with_name = zip(names, predictions)
print(predictions_with_name)

f = open("submission8.csv", "x")
with open("./pml-2022-smart/submission8.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(('id', 'class'))
    writer.writerows(predictions_with_name)

print("done")


