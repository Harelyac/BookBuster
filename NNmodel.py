import keras
import pandas as pd
import numpy as np
import pickle
#importing csv file
#d = pd.read_csv('heart.csv')

from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(, , test_size = 0.2, random_state = 0)
#Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

#Building model
from keras.models import Sequential
from keras.layers import Dense

np.set_printoptions(precision=3, suppress=True)
from keras.layers import Dropout


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prep_data():
    MULT_CONSTANT = 700
    all_labels, all_vecs = loadinging_feature_vectors()
    removing_baised_tags(all_vecs)
    tuninig_features(MULT_CONSTANT, all_vecs)
    return divide_to_train_test(all_labels, all_vecs)


def divide_to_train_test(all_labels, all_vecs):
    train_x = np.zeros((2860, 404))
    train_y = np.zeros((2860, 1))
    test_x = np.zeros((300, 404))
    test_y = np.zeros((300, 1))
    train_x[0:1430, :] = all_vecs[0:1430, :]
    train_x[1430:2860, :] = all_vecs[-1430:, :]
    train_y[0:1430, :] = all_labels[0:1430, :]
    train_y[1430:2860, :] = all_labels[-1430:, :]
    test_x[0:150, :] = all_vecs[1430:1580, :]
    test_x[150:300, :] = all_vecs[-1580:-1430, :]
    test_y[0:150, :] = all_labels[1430:1580, :]
    test_y[150:300, :] = all_labels[-1580:-1430, :]
    return train_x, train_y, test_x, test_y


def loadinging_feature_vectors():
    data = load_pickle("pickles/feature_vectors_with_labels.pickle")
    authors_with_alot_of_books_data = load_pickle("pickles/new_training_ickles.pickle")
    print(len(authors_with_alot_of_books_data))
    data = data[-1580:]
    all_vecs = np.zeros((1580 * 2, 404))
    all_labels = np.zeros((1580 * 2, 1))
    for i, tuple in enumerate(data):
        all_vecs[i, :] = data[i][0]
        all_labels[i, :] = data[i][1]
    for i, tuple in enumerate(authors_with_alot_of_books_data[:1580]):
        all_vecs[i + 1580] = authors_with_alot_of_books_data[i][0]
        all_labels[i + 1580] = authors_with_alot_of_books_data[i][1]
    return all_labels, all_vecs


def removing_baised_tags(all_vecs):
    all_vecs[:, -159] = 0
    all_vecs[:, -161] = 0
    all_vecs[:, -380] = 0
    all_vecs[:, -382] = 0
    all_vecs[:, -389] = 0
    all_vecs[:, -390] = 0


def tuninig_features(MULT_CONSTANT, all_vecs):
    all_vecs[:, 6] = all_vecs[:, 6] * MULT_CONSTANT
    all_vecs[:, 7] = all_vecs[:, 7] * MULT_CONSTANT
    all_vecs[:, 0] = all_vecs[:, 0] * MULT_CONSTANT
    all_vecs[:, 1] = all_vecs[:, 1] * MULT_CONSTANT * all_vecs[:, 2]
    all_vecs[:, 3] = all_vecs[:, 3] * MULT_CONSTANT * all_vecs[:, 4]
    all_vecs[:, 5] = all_vecs[:, 5] * MULT_CONSTANT
    all_vecs[:, 4] = all_vecs[:, 4] * MULT_CONSTANT
    all_vecs[:, 2] = all_vecs[:, 2] * MULT_CONSTANT
    all_vecs[:, 9] = all_vecs[:, 9] * 0.1
    all_vecs[:, 8] = -all_vecs[:, 8] * MULT_CONSTANT


def network_compile(train_x, train_y, fit=False):
    # kernel_initializeialising the ANN
    classifier = Sequential()
    classifier.add(Dense(units=404, kernel_initializer='uniform', activation='relu', input_dim=404))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=375, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=350, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=325, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=300, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=275, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=250, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=225, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=110, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=80, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if fit:
        classifier.fit(train_x, train_y, batch_size=200, epochs=2500)
    return classifier


def evluate_and_confusion_matrix(classifier, test_x, test_y):
    y_pred = classifier.predict(test_x)
    y_pred = [1 if (i > 0.5) else 0 for i in y_pred]  # (y_pred > 0.5)
    lst = []
    i = 0
    total_cor = 0
    tota = 0
    for true, pred in zip(test_y, y_pred):
        if test_y[i] == y_pred[i]:
            total_cor += 1
        tota += 1
        i += 1
        print(true, pred)
    print(total_cor / tota)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y, y_pred, labels=[1, 0])
    print(cm)


def main():
    fit = False
    train_x, train_y, test_x, test_y = prep_data()
    classifier = network_compile(train_x, train_y, fit=fit)
    classifier.load_weights("pickles/Network_Trained.h5")
    evluate_and_confusion_matrix(classifier, test_x, test_y)


if __name__ == "__main__":
    main()
