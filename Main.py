import pandas as pd
import sys

def Main():

    if len(sys.argv) != 2:
        print("Not enough arguments, please provide the filename of the dataset.")
        return


    try:
        print("Trying to load dataset ", sys.argv[1])
        data = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print("File", sys.argv[1], " not found, exiting.")
        return
    else:
        weights = [1, 1, 1, 1, 1, 1]
        learningRate = .005
        iterations = 200

        for i in range(iterations):
            dE = predict(data, weights, i)
            weights = calculateNewWeights(weights, dE, learningRate)
            if i == iterations - 1:
                print("Final derivative errors: ", dE)

        print("Final weights: ", weights)


def calculateNewWeights(weights, dE, learningRate):
    newWeights = [0, 0, 0, 0, 0, 0]
    for i in range(len(weights)):
        newWeights[i] = weights[i] - (learningRate * dE[i])
    return newWeights


def predict(data, weights, iteration):

    numRows = data.shape[0]
    numCols = data.shape[1]
    meanSquaredError = 0
    derivativeErrors = [0, 0, 0, 0, 0, 0]

    for i in data.iterrows():
        prediction = 0
        for x in range(numCols - 1):
            prediction += (weights[x] * i[1][x])

        actual = i[1][numCols - 1]
        error = prediction - actual

        for e in range(numCols - 1):
            derivativeErrors[e] += (error * i[1][e])

        meanSquaredError += pow(error, 2)

    meanSquaredError /= (2 * numRows)
    print("Mean squared error for iteration ", iteration, ": ", round(meanSquaredError, 4))

    derivativeErrors = [dE / numRows for dE in derivativeErrors]

    return derivativeErrors


Main()
