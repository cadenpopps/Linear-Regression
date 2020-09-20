
class Learner():

    def __init__(self, trainingDataset, iterations, learningRate, startingWeights):
        self.trainingDataset= trainingDataset
        self.iterations = iterations
        self.learningRate = learningRate
        self.weights = startingWeights

    def learnSet(self, dataset):
        startingWeight = self.weights
        currentDerivativeErrors = [0] * 6
        currentMeanSquaredError = 0
        for iteration in range(self.iterations):
            currentDerivativeErrors, currentMeanSquaredError = self.predict(dataset, self.weights)
            self.weights = self.calculateNewWeights(self.weights, currentDerivativeErrors, self.learningRate)

        return self.iterations, self.learningRate, startingWeight, currentDerivativeErrors, self.weights, currentMeanSquaredError

    def testSet(self, dataset):
        currentMeanSquaredError = 0
        currentDerivativeErrors, currentMeanSquaredError = self.predict(dataset, self.weights)

        return currentDerivativeErrors, self.weights, currentMeanSquaredError


    def calculateNewWeights(self, weights, dE, learningRate):
        newWeights = [0, 0, 0, 0, 0, 0]
        for i in range(len(weights)):
            newWeights[i] = weights[i] - (learningRate * dE[i])
        return newWeights


    def predict(self, data, weights):

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
        derivativeErrors = [dE / numRows for dE in derivativeErrors]

        return derivativeErrors, meanSquaredError

