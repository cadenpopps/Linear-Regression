import pandas as pd
import sys
sys.path.append(".")
import Learner

DEFAULT_TRAINING_DATASET = "http://cadenpopps.com/machine-learning/training.data"
DEFAULT_TESTING_DATASET = "http://cadenpopps.com/machine-learning/full.data"
DEFAULT_ITERATIONS = 100
DEFAULT_LEARNING_RATE = .001
DEFAULT_STARTING_WEIGHT = 1
LOG_DECIMAL_PLACES = 4
LOG_FILENAME = "log.txt"

def Main():

    trainingDataset, testDataset, iterations, learningRate, startingWeights = handleCommandLineParameters()

    logFile = safeLoadFile(LOG_FILENAME, "a")


    print("\nBegin learning...")
    learner = Learner.Learner(trainingDataset, iterations, learningRate, startingWeights)
    trainingOutput = learner.learnSet(trainingDataset)
    print("Learning complete!")
    logLearnerOutput(logFile, formatTrainingOutput(trainingOutput))

    print("\nBegin testing...")
    testOutput = learner.testSet(testDataset)
    print("Testing complete!")
    logLearnerOutput(logFile, formatTestOutput(testOutput))

    print("Logging results to", LOG_FILENAME)


def handleCommandLineParameters():

    print()

    trainingDataset = 0
    testDataset = 0

    if len(sys.argv) >= 2:
        trainingDataset = safeLoadDataset(sys.argv[1])
    else:
        trainingDataset = safeLoadDataset(DEFAULT_TRAINING_DATASET)

    if len(sys.argv) >= 3:
        testDataset = safeLoadDataset(sys.argv[2])
    else:
        testDataset = safeLoadDataset(DEFAULT_TESTING_DATASET)

    iterations = DEFAULT_ITERATIONS
    if len(sys.argv) >= 4:
        iterations = int(sys.argv[3])
        print("Iterations:", iterations, "(set by user)")
    else:
        print("Iterations:", iterations, "(default)")

    learningRate = DEFAULT_LEARNING_RATE
    if len(sys.argv) >= 5:
        learningRate = float(sys.argv[4])
        print("Learning rate:", learningRate, "(set by user)")
    else:
        print("Learning rate:", learningRate, "(default)")

    startingWeights = [DEFAULT_STARTING_WEIGHT] * 6
    if len(sys.argv) >= 6:
        startingWeight = float(sys.argv[5])
        startingWeights = [startingWeight] * 6
        print("Starting weights:", startingWeights, "(set by user)")
    else:
        print("Starting weights:", startingWeights, "(default)")

    return trainingDataset, testDataset, iterations, learningRate, startingWeights


def safeLoadDataset(filename):
    try:
        print("Trying to load dataset:", filename)
        data = pd.read_csv(filename)
        print("Successfully loaded dataset:", filename, "\n")
        return data
    except FileNotFoundError:
        print("File", filename, " not found, exiting.")
        return


def safeLoadFile(filename, mode):
    try:
        return open(filename, mode)
    except OSError:
        print("Could not open/read file:", filename)
        sys.exit()


def logLearnerOutput(logFile, formattedOutput):
    print(formattedOutput)
    logFile.write(formattedOutput)
    logFile.write("\n")


def formatTrainingOutput(learnerOutput):
    formattedOutput = "\n"
    formattedOutput += "Training parameters:\n"
    formattedOutput += "\tIterations: " + str(learnerOutput[0]) + "\n"
    formattedOutput += "\tLearning rate: " + str(learnerOutput[1]) + "\n"
    formattedOutput += "\tStarting weights: " + str(learnerOutput[2]) + "\n"

    derivativeErrors = [round(element, LOG_DECIMAL_PLACES) for element in learnerOutput[3]]
    weights = [round(element, LOG_DECIMAL_PLACES) for element in learnerOutput[4]]

    formattedOutput += "Training results:\n"
    formattedOutput += "\tFinal derivative error per attribute: " + str(derivativeErrors) + "\n"
    formattedOutput += "\tFinal weight per attribute: " + str(weights) + "\n"
    formattedOutput += "\tFinal mean squared error: " + str(learnerOutput[5])
    return formattedOutput


def formatTestOutput(learnerOutput):
    formattedOutput = "\n"

    derivativeErrors = [round(element, LOG_DECIMAL_PLACES) for element in learnerOutput[0]]
    weights = [round(element, LOG_DECIMAL_PLACES) for element in learnerOutput[1]]

    formattedOutput += "Test results:\n"
    formattedOutput += "\tDerivative error per attribute: " + str(derivativeErrors) + "\n"
    formattedOutput += "\tWeights per attribute: " + str(weights) + "\n"
    formattedOutput += "\tTest mean squared error: " + str(learnerOutput[2]) + "\n"
    return formattedOutput


Main()
