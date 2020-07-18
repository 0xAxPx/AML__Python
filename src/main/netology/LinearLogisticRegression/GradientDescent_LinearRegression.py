from sklearn.datasets import make_regression
from util.logger import initLogger

log = initLogger(__name__)

log.info("Implementation of gradient descent for LinearRegression")


def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


def gradientLinR(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            log.info("expected={}, predicted={}".format(row[-1], yhat))
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        log.info('epoch={}, lrate={}, error={}'.format(epoch, l_rate, sum_error))
    return coef


# test dataset
X1, Y1 = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)
log.info("Dependant(Y)...{}".format(Y1[:10, ]))
log.info("InDependant(X)...{}".format(X1[:10, ]))

# learning rate and epoch
lrate = 0.001
epochs = 100
res = gradientLinR(X1, lrate, epochs)
log.info("Result:{}".format(res))
