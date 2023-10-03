from .Node import Node
import math

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error


import warnings
warnings.filterwarnings("ignore")

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019-2023 J. E. Batista
#

class Individual:
	training_X = None
	training_Y = None
	test_X = None
	test_Y = None

	operators = None # operations allowed for non-terminal nodes
	terminals = None # Features

	head = None # Tree representation of the static (see below) individuals
	semantics = None # Model output (real values)
	normalized = False # Normalized individuals have a sigmoid attached to their root
	weights = None # An individual is a weighted sum of the trees in the population

	# Static = Does this individual belong to the generation 0 trees (immutable)
	static = None 

	# The results are saved to avoid recalculations
	trainingAccuracy = None
	testAccuracy = None
	trainingRMSE = None
	testRMSE = None


	def __init__(self, operators, terminals, max_depth, normalized=False, static=False):
		self.operators = operators
		self.terminals = terminals
		self.max_depth = max_depth
		self.normalized = normalized
		self.static = static


	def create(self, weights, rng, semantics = None, Tr_X=None, Tr_Y=None, Te_X=None, Te_Y=None, makeHead=True):
		if makeHead:
			self.head = Node()
			self.head.create(rng, self.operators, self.terminals, self.max_depth, full=True)
		
		self.training_X = Tr_X
		self.training_Y = Tr_Y
		self.test_X = Te_X
		self.test_Y = Te_Y

		if semantics == None:
			self.setSemantics()
		else:
			self.semantics = semantics
		self.weights = weights


	def calculate(self, X):
		values = list(self.head.calculate(X))

		if self.normalized:
			for i in range(len(values)):
				values[i] = sigmoid(values[i])
		return values


	def setSemantics(self):
		semantics = []
		semantics.extend( self.calculate( self.training_X ) )
		semantics.extend( self.calculate( self.test_X ) )
		self.semantics = semantics


	def getSemantics(self):
		return self.semantics


	def getTrainingSemantics(self):
		return self.semantics[:len(self.training_X)]


	def getTestSemantics(self):
		return self.semantics[len(self.training_X):]


	def getSize(self,forest=None, normalizedForest=None):
		if self.static:
			return self.head.getSize() + (1 if self.normalized else 0)
		size = 0
		for i in range(len(self.weights)):
			if self.weights[i] != 0:
				if i < len(forest):
					size += forest[i].getSize() + 2 # +2 from the "weight*" operation
				else:
					size += normalizedForest[i-len(forest)].getSize() + 2
		return size


	def __str__(self):
		s = str(self.head)
		if self.normalized:
			s = "sigmoid( " + s+ " )"
		return s

	def __gt__(self,other):
		return self.getFitness() > other.getFitness()


	def getFitness(self):
		return self.getRMSE(self.training_X, self.training_Y, pred="Tr") *-1



	def getRMSE(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingSemantics()
		elif pred == "Te":
			pred = self.getTestSemantics()
		else:
			pred = self.predict(X)


		return  mean_squared_error(pred, Y)**0.5

	
	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.classifyArray(self.getTrainingSemantics())
		elif pred == "Te":
			pred = self.classifyArray(self.getTestSemantics())
		else:
			pred = self.predict(X)

		pred = self.classifyArray(pred)

		return accuracy_score(pred, Y)


	def getWaF(self, X, Y,pred=None):
		'''
		Returns the individual's WAF.
		'''
		if pred == "Tr":
			pred = self.classifyArray(self.getTrainingSemantics())
		elif pred == "Te":
			pred = self.classifyArray(self.getTestSemantics())
		else:
			pred = self.predict(X)

		pred = self.classifyArray(pred)

		return f1_score(pred, Y, average="weighted")


	def getKappa(self, X, Y,pred=None):
		'''
		Returns the individual's kappa value.
		'''
		if pred == "Tr":
			pred = self.classifyArray(self.getTrainingSemantics())
		elif pred == "Te":
			pred = self.classifyArray(self.getTestSemantics())
		else:
			pred = self.predict(X)

		pred = self.classifyArray(pred)

		return cohen_kappa_score(pred, Y)


	def classifyArray(self, v):
		v = v[:]
		for i in range(len(v)):
			v[i] = 0 if v[i] < 0.5 else 1
		return v



def sigmoid(x):
	# Avoids overflow on the math.exp() function
	if x < -100:
		return 0
	if x > 100:
		return 1
	return 1 / ( 1 + math.exp(-x) )