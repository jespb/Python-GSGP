from .Individual import Individual
from .Node import Node
from .Constants import *
from .GeneticOperators import *
from random import random, randint

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

class Population:
	# Static population defined during initialization
	forest = None 
	normalizedForest = None

	# Population to be evolved
	population = None
	
	bestIndividual = None

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingRmseOverTime = None
	testRmseOverTime = None
	sizeOverTime = None
	currentGeneration = None

	def __init__(self):
		self.currentGeneration = 0
		self.population = []
		self.trainingAccuracyOverTime = []
		self.testAccuracyOverTime = []
		self.trainingRmseOverTime = []
		self.testRmseOverTime = []
		self.sizeOverTime = []

		self.forest = []
		self.normalizedForest = []
		self.population = []
		for i in range(POPULATION_SIZE):
			self.normalizedForest.append(Individual(normalized = True,static=True))

			ind = Individual(static=True)
			self.forest.append(ind)
			
			weight = [0]*POPULATION_SIZE*2
			weight[i]=1
			self.population.append(Individual(weights = weight, semantics = ind.getSemantics()))


	def stoppingCriteria(self):
		genLimit = self.currentGeneration >= MAX_GENERATION
		perfectTraining = self.bestIndividual != None
		perfectTraining = perfectTraining and self.bestIndividual.getTrainingRMSE() == 0
		
		return genLimit or perfectTraining


	def train(self):
		while not self.stoppingCriteria():
			self.nextGeneration()
			self.currentGeneration += 1
			self.trainingAccuracyOverTime.append(self.bestIndividual.getTrainingAccuracy())
			self.testAccuracyOverTime.append(self.bestIndividual.getTestAccuracy())
			self.trainingRmseOverTime.append(self.bestIndividual.getTrainingRMSE())
			self.testRmseOverTime.append(self.bestIndividual.getTestRMSE())
			self.sizeOverTime.append(self.bestIndividual.getSize(self.forest, self.normalizedForest))
		while self.currentGeneration < MAX_GENERATION:
			self.currentGeneration += 1
			self.trainingAccuracyOverTime.append(self.bestIndividual.getTrainingAccuracy())
			self.testAccuracyOverTime.append(self.bestIndividual.getTestAccuracy())
			self.trainingRmseOverTime.append(self.bestIndividual.getTrainingRMSE())
			self.testRmseOverTime.append(self.bestIndividual.getTestRMSE())
			self.sizeOverTime.append(self.bestIndividual.getSize(self.forest, self.normalizedForest))	


	def nextGeneration(self):
		# Sort the population from best to worse (the fitness is implicitly calculated here)
		self.population.sort(reverse=True) 

		# Update Best Individual
		if(self.bestIndividual == None or self.population[0]>self.bestIndividual):
			self.bestIndividual = self.population[0]

		if self.currentGeneration % 100 == 0:
			if OUTPUT == "Classification":
				print("Gen#",self.currentGeneration, "- (TrA,TeA,TrRMSE):", self.bestIndividual.getTrainingAccuracy(),self.bestIndividual.getTestAccuracy(),self.bestIndividual.getTrainingRMSE())
			if OUTPUT == "Regression":
				print("Gen#",self.currentGeneration, "- (TrRMSE,TeRMSE):", self.bestIndividual.getTrainingRMSE(), self.bestIndividual.getTestRMSE())
		

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend( getElite(self.population) )
		while len(newPopulation) < POPULATION_SIZE:
			newPopulation.extend( getOffspring(self.population,self.normalizedForest) )
		self.population = newPopulation[:POPULATION_SIZE]
		

	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)


	def getBestIndividual(self):
		return self.bestIndividual


	def getCurrentGeneration(self):
		return self.currentGeneration


	def getFormatedModel(self):
		s = ""
		for i in range(len(self.bestIndividual.weights)):
			if i < POPULATION_SIZE:
				if self.bestIndividual.weights[i] != 0:
					s += str(self.bestIndividual.weights[i]) + "  *  " + str(self.forest[i]) + " + "
			else:
				if self.bestIndividual.weights[i] != 0:
					s += str(self.bestIndividual.weights[i]) + "  *  " + str(self.normalizedForest[i-POPULATION_SIZE]) + " + "
		return s[:-2] #Ignore the last "+ "
