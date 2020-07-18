from numpy import e
from sklearn.datasets import make_blobs
from util.logger import initLogger

log = initLogger(__name__)

log.info("Implementation of gradient descent for LogisticRegression")


# formula
# y = 1.0 / (1.0 + e^(-(b0 + b1 * X1 + b2 * X2)))

def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1 / (1 + e ** (-yhat))


def gradientLogR(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            log.info("expected={}, predicted={}".format(row[-1], yhat))
            error = row[-1] - yhat
            sum_error += error ** 2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    log.info('epoch={}, lrate={}, error={}'.format(epoch, l_rate, sum_error))
    return coef


# dataset
X_, y_ = make_blobs(n_samples=10, centers=2, n_features=2, random_state=0)
log.info("Classes .... {}".format(y_))
log.info("X .... {}".format(X_))

# learning rate and epoch
lrate = 0.001
epochs = 100
res = gradientLogR(X_, lrate, epochs)
log.info("Result: {}".format(res))
