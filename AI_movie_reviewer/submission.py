#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    d = defaultdict(int)    # default value of int is 0 for default dict
    word_list = x.split()   # ['I', 'am', 'what', 'I', 'am']
    for word in word_list:
        d[word] += 1
    return d                # {'I': 2, 'am': 2, 'what': 1}


############################################################
# Milestone 4: Sentiment Classification


def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight
    epoch = 0
    for epoch in range(numEpochs):
        # for loop for each reviews
        for data, label in trainExamples:       # trainExample: [('I am what', 1), ('he is cool', 0), ...]
            label = 0 if label == -1 else 1     # change all -1 labels to 0

            y = label  # 0 or 1
            x = featureExtractor(data)              # {'I': 2, 'am': 2, 'what': 1}
            k = dotProduct(weights, x)              # k = 0*2 + 0*2 + 0*1
            h = 1/(1+math.exp(-k))                  # h = 0.5
            increment(weights, -alpha*(h-y), x)     # for loop: w_i = w_i - alpha(h-y)x_i -> return weight

        def predictor(j):
            # function that takes an x and returns a predicted y
            fv = featureExtractor(j)
            predict_k = dotProduct(weights, fv)
            if predict_k > 0:
                return 1
            return 0

        print(f"Training Error: ({epoch} epoch): {evaluatePredictor(trainExamples, predictor)}")
        print(f"Validation Error: ({epoch} epoch): {evaluatePredictor(validationExamples, predictor)}")
        epoch += 1
    return weights  # weights = {'I': 2, 'am': 2, 'what': 1, ...}


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and value is exactly 1.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        phi = {}
        for i in range(random.randint(1, len(weights))):
            phi[random.choice(list(weights.keys()))] = 1
        y = -1 if dotProduct(weights, phi) < 0 else 1
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        alt_string = ''
        d2 = defaultdict(int)
        alt_string = ''.join(x.split())
        for i in range(len(alt_string) - n + 1):
            d2[alt_string[i:i+n]] += 1
        return d2
    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

