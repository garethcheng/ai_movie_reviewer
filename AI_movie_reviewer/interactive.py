"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from util import *
from submission import *


def main():
	weights = load_weights()
	interactivePrompt(extractWordFeatures, weights)


def load_weights():
	"""
	This function can load weights file into weight vector
	@return: weightVector, Dict[str, float]
	"""
	with open('weights', 'r', encoding="utf-8") as file:
		weights = {}
		for line in file:
			line = line.strip()
			weights[line.split('\t')[0]] = float(line.split('\t')[1])
		return weights


if __name__ == '__main__':
	main()