from .Node import Node
from .Constants import *
from .Util import *
import math


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

class Individual:
	head = None
	semantics = None
	normalized = False
	weights = None
	static = None

	trainingAccuracy = None
	testAccuracy = None
	trainingRMSE = None
	testRMSE = None


	def __init__(self, node = None, fromString = None, normalized=False, weights=None, semantics = None, static=False):
		self.normalized = normalized
		if fromString == None:
			self.head = Node() if node == None else node
		else:
			self.head = Node(fromString = fromString.split())
		if semantics == None:
			self.setSemantics()
		else:
			self.semantics = semantics
		self.weights = weights
		self.static = static

	def predict(self, sample):
		return 0 if self.calculate(sample) < 0.5 else 1

	def calculate(self, sample):
		value = self.head.calculate(sample)
		if self.normalized:
			value = sigmoid(value)
		return value

	def setSemantics(self):
		training = getTrainingSet()
		test = getTestSet()
		semantics = []
		for sample in training:
			semantics.append(self.calculate(sample))
		for sample in test:
			semantics.append(self.calculate(sample))
		self.semantics = semantics

	def getSemantics(self):
		return self.semantics

	def getHead(self):
		return self.head.clone()

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
		return self.getTrainingRMSE() < other.getTrainingRMSE()



	def getTrainingRMSE(self):
		if self.trainingRMSE == None:
			ds = getTrainingSet()

			semantics = self.semantics
			rmse = 0
			for i in range(len(ds)):
				rmse += (semantics[i]-ds[i][-1])**2
			rmse = rmse ** 0.5
			rmse /= len(semantics)
			self.trainingRMSE = rmse
		return self.trainingRMSE
	
	def getTestRMSE(self):
		if self.testRMSE == None:
			ds = getTestSet()
			trsize = len(getTrainingSet())

			semantics = self.semantics
			rmse = 0
			for i in range(len(ds)):
				rmse += (semantics[i+trsize]-ds[i][-1])**2
			rmse = rmse ** 0.5
			rmse /= len(semantics)
			self.testRMSE = rmse
		return self.testRMSE

	def getTrainingAccuracy(self):
		if self.trainingAccuracy == None:
			ds = getTrainingSet()

			semantics = self.semantics
			hits = 0
			for i in range(len(ds)):
				if (semantics[i] < 0.5 and ds[i][-1] == 0) or (semantics[i] >= 0.5 and ds[i][-1] == 1):
					hits+=1
			self.trainingAccuracy = hits*100/len(ds)
		return self.trainingAccuracy

	def getTestAccuracy(self):
		if self.testAccuracy == None:
			ds = getTestSet()
			trsize = len(getTrainingSet())

			semantics = self.semantics
			hits = 0
			for i in range(len(ds)):
				if (semantics[i+trsize] < 0.5 and ds[i][-1] == 0) or (semantics[i+trsize] >= 0.5 and ds[i][-1] == 1):
					hits+=1
			self.testAccuracy = hits*100/len(ds)
		return self.testAccuracy
