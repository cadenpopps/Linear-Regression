import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_ITERATIONS = 100
DEFAULT_LEARNING_RATE = .001
DEFAULT_STARTING_WEIGHT = 1
LOG_DECIMAL_PLACES = 4
LOG_FILENAME = "scikit_log.txt"

def Main():

    trainingDataset, testDataset = handleCommandLineParameters()

    logFile = safeLoadFile(LOG_FILENAME, "a")

    training_x, training_y = processDataset(trainingDataset)
    test_x, test_y = processDataset(testDataset)

    print("Begin learning...")
    regr = linear_model.LinearRegression()
    regr.fit(training_x, training_y)
    print("Learning complete!")


    print("\nBegin testing...")
    test_pred = regr.predict(test_x)
    print("Testing complete!")

    logScikitOutput(logFile, formatScikitOutput(regr, test_x, test_y, test_pred))

    plot_x = [x[1] for x in test_x]
    print(len(plot_x))
    print(len(test_y))
    plt.scatter(plot_x, test_y,  color='black')
    plt.plot(plot_x, test_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    print("Logging results to", LOG_FILENAME)


def handleCommandLineParameters():

    if len(sys.argv) < 2:
        print("Not enough arguments, please provide the filename of the training dataset.")
        return

    print()
    trainingDataset = safeLoadDataset(sys.argv[1])
    testDataset = trainingDataset
    if len(sys.argv) >= 3:
        testDataset = safeLoadDataset(sys.argv[2])

    return trainingDataset, testDataset


def safeLoadDataset(filename):
    try:
        print("Trying to load dataset:", filename)
        data = pd.read_csv(filename)
        print("Successfully loaded dataset:", filename, "\n")
        return data
    except FileNotFoundError:
        print("File", filename, " not found, exiting.")
        return

def processDataset(dataset):
    attributes = []
    outputs = []
    datasetWidth = dataset.shape[1] - 1
    for i in dataset.iterrows():
        rowAttributes = []
        for x in range(datasetWidth):
            rowAttributes.append(i[1][x])
        attributes.append(rowAttributes)
        outputs.append(i[1][-1])
    return attributes, outputs


def safeLoadFile(filename, mode):
    try:
        return open(filename, mode)
    except OSError:
        print("Could not open/read file:", filename)
        sys.exit()


def logScikitOutput(logFile, formattedOutput):
    logFile.write(formattedOutput)
    logFile.write("\n")


def formatScikitOutput(regr, test_x, test_y, test_pred):
    formattedOutput = "\n"
    formattedOutput += "Coefficients:" + str(regr.coef_) + "\n"
    formattedOutput += "Mean squared error:" + str(mean_squared_error(test_y, test_pred))

    return formattedOutput


Main()
