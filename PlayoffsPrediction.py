from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
import numpy as np
from nba_py import constants


def SpiltData():
    teams_enumerate = {}
    num = 0
    for t in constants.TEAMS:
        teams_enumerate[t] = num
        num += 1
    data = open('2015-16_reg.txt').readlines()
    train, test = model_selection.train_test_split(data, test_size=0.2)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(train)):
        data = train[i].strip()
        record = data.split(' ')
        train_x.append([teams_enumerate[record[0]], teams_enumerate[record[1]]])
        if int(record[2]) - int(record[3]) > 0:
            train_y.append(1)
        else:
            train_y.append(0)

    for i in range(len(test)):
        data = test[i].strip()
        record = data.split(' ')
        test_x.append([teams_enumerate[record[0]], teams_enumerate[record[1]]])
        if int(record[2]) - int(record[3]) > 0:
            test_y.append(1)
        else:
            test_y.append(0)
    # data format: [home_team_id, away_team_id, 1 if home_team_win else 0]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def LinearRegression(train_x, train_y, test_x, test_y):
    lg = linear_model.LinearRegression()
    lg.fit(train_x, train_y)
    test_predict = lg.predict(X=test_x)
    all = len(test_y)
    correct = 0
    for i in range(len(test_y)):
        if test_predict[i] >= 0.5 and test_y[i] == 1:
            correct += 1
        if test_predict[i] < 0.5 and test_y[i] == 0:
            correct += 1
    print("Linear Regression accuracy: " + str(correct/all))


def MaximumLikelihood(train_x, train_y, test_x, test_y):
    # seperate data by classes(win/lose)
    data = [[], []]
    for i in range(len(train_y)):
        if train_y[i] == 1:
            data[1].append(train_x[i])
        else:
            data[0].append(train_x[i])
    # compute the means of each feature for each class
    means = np.zeros((2, 2))
    for cls in range(2):
        for dim in range(2):
            total = [data[cls][i][dim] for i in range(len(data[cls]))]
            means[cls][dim] = np.mean(total)
    covariances = np.zeros((2, 2, 2))
    # compute the covariance matrix for each class
    for cls in range(2):
        covariances[cls] = np.cov(np.transpose(data[cls]))

    # compute the log-likelihood for each test sample for each class
    n = len(test_x)
    lgl = np.zeros((n, 2))
    for i in range(n):
        for cls in range(2):
            lgl[i][cls] = np.log((2*np.pi)**(-32) * np.linalg.det(covariances[cls])**(-0.5)
                                 * np.e**(-0.5 * np.matmul(np.matmul(np.transpose(test_x[i] - means[cls]),
                                                                     np.linalg.inv(covariances[cls])),
                                                           test_x[i] - means[cls])))
    # compute the conditional log-likelihood
    cl = np.zeros((n, 2))
    for i in range(n):
        p_x = 0
        for cls in range(2):
            p_x += np.e**lgl[i][cls]
        for cls in range(2):
            cl[i][cls] = lgl[i][cls] + np.log(0.5) - np.log(p_x)
    correct = 0
    for i in range(n):
        predicted = np.argmax(cl[i])
        print(predicted)
        if predicted == test_y[i]:
            correct += 1
    print("Maximum Likelihood Estimate accuracy: " + str(correct/n))


def LogisticRegression(train_x, train_y, test_x, test_y):
    lr = linear_model.LogisticRegression()
    lr.fit(train_x, train_y)
    predictions = lr.predict(test_x)
    print(predictions)


def DecisionTree(train_x, train_y, test_x, test_y):
    dt = tree.DecisionTreeClassifier(max_depth=9, criterion="entropy")
    dt.fit(train_x, train_y)
    predictions = dt.predict(test_x)
    print(predictions)
    print(dt.score(test_x, test_y))


def SupportVectorMachine(train_x, train_y, test_x, test_y):
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    print(predictions)
    print(clf.score(test_x, test_y))











if __name__ == '__main__':
    train_x, train_y, test_x, test_y = SpiltData()
    # LinearRegression(train_x, train_y, test_x, test_y)
    # MaximumLikelihood(train_x, train_y, test_x, test_y)
    # LogisticRegression(train_x, train_y, test_x, test_y)
    # DecisionTree(train_x, train_y, test_x, test_y)
    SupportVectorMachine(train_x, train_y, test_x, test_y)



