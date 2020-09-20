import sys
import math
import random

def Main():
    if len(sys.argv) != 4:
        print("Incorrect number of arguments, please enter the filename of the dataset, the desired filename of the output training dataset, and the percentage of lines the training set should include.")
        return

    fullDataSet = openFile(sys.argv[1], "r")
    trainingDataSet = openFile(sys.argv[2], "w")
    print("Creating training dataset", sys.argv[2], "from full dataset", sys.argv[1])
    makeTrainingSet(fullDataSet, trainingDataSet, int(sys.argv[3]))

    print("Success!")

    return


def openFile(filename, mode):
    try:
        return open(filename, mode)
    except OSError:
        print("Could not open/read file:", filename)
        sys.exit()

def makeTrainingSet(full, training, percent):

    numLines = sum(1 for line in full) - 1
    numLinesTraining = math.floor(numLines / 100 * percent)

    full.seek(0);
    training.write(full.readline())

    print("Full dataset has", numLines, "lines.")
    print("Training dataset will have approximately", numLinesTraining, "lines.")

    skipLines = math.floor(100 / percent)
    randomOffset = math.floor(random.random() * skipLines)

    for offset in range(randomOffset):
        full.readline()

    while True:
        for s in range(skipLines):
            full.readline()
        nextline = full.readline()
        if not nextline:
            return
        training.write(nextline)

Main()
