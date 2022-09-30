from .Individual import Individual
from .Node import Node
from .GeneticOperators import *
from random import Random#random, randint
import multiprocessing as mp
import time

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message



class GSGP:
	## __INIT__ arguments
	operators = None
	max_initial_depth = None
	population_size = None
	threads = None
	random_state = 42
	rng = None # random number generator

	max_generation = None
	tournament_size = None
	elitism_size = None

	mutation_step = None

	verbose = None



	# CONST population
	forest = None 
	normalizedForest = None


	## FIT arguments
	terminals = None

	population = None
	currentGeneration = 0
	bestIndividual: Individual = None

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingWaFOverTime = None
	testWaFOverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	trainingMSEOverTime = None
	testMSEOverTime = None
	generationTimes = None

	checkAccuracy = True
	checkRMSE = True

	
	def checkIfTrained(self):
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")




	def __init__(self, operators=[("+",2),("-",2),("*",2),("/",2)], max_initial_depth = 6, 
		population_size = 500, max_generation = 1000, tournament_size = 5, elitism_size = 1, 
		mutation_step = 0.1, threads=1, random_state = 42, verbose = True):

		if sum( [0 if op in [("+",2),("-",2),("*",2),("/",2)] else 0 for op in operators ] ) > 0:
			print( "[Warning] Some of the following operators may not be supported:", operators)

		self.operators = operators

		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.threads = max(1, threads)
		self.random_state = random_state
		self.rng = Random(random_state)

		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size

		self.mutation_step = mutation_step

		self.verbose = verbose



	def __str__(self):
		self.checkIfTrained()
		return str(self.getBestIndividual())



	def getCurrentGeneration(self):
		return self.currentGeneration


	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		self.checkIfTrained()

		return self.bestIndividual

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingAccuracyOverTime, self.testAccuracyOverTime]

	def getWaFOverTime(self):
		'''
		Returns the training and test WAF of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingWaFOverTime, self.testWaFOverTime]

	def getKappaOverTime(self):
		'''
		Returns the training and test kappa values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingKappaOverTime, self.testKappaOverTime]

	def getRMSEOverTime(self):
		'''
		Returns the training and test mean squared error values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingRMSEOverTime, self.testRMSEOverTime]

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.generationTimes


	def getFormatedModel(self):
		s = ""
		for i in range(len(self.bestIndividual.weights)):
			if i < self.population_size:
				if self.bestIndividual.weights[i] != 0:
					s += str(self.bestIndividual.weights[i]) + "  *  " + str(self.forest[i]) + " + "
			else:
				if self.bestIndividual.weights[i] != 0:
					s += str(self.bestIndividual.weights[i]) + "  *  " + str(self.normalizedForest[i-self.population_size]) + " + "
		return s[:-2] #Ignore the last "+ "





	def fit(self,Tr_x, Tr_y, Te_x = None, Te_y = None):
		if self.verbose:
			print("  > Parameters")
			print("    > Random State:       "+str(self.random_state))
			print("    > Operators:          "+str(self.operators))
			print("    > Population Size:    "+str(self.population_size))
			print("    > Max Generation:     "+str(self.max_generation))
			print("    > Tournament Size:    "+str(self.tournament_size))
			print("    > Elitism Size:       "+str(self.elitism_size))
			print("    > Max Initial Depth:  "+str(self.max_initial_depth))
			print("    > Threads:            "+str(self.threads))
			print()

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y
		self.terminals = list(Tr_x.columns)


#	
		self.forest = []
		self.normalizedForest = []
		self.population = []
		for i in range(self.population_size):
			ind = Individual(self.operators, self.terminals, self.max_initial_depth, normalized = True,static=True)
			ind.create(None, self.rng, Tr_X=self.Tr_x, Tr_Y=self.Tr_y, Te_X=self.Te_x, Te_Y = self.Te_y)
			self.normalizedForest.append(ind)

			ind = Individual(self.operators, self.terminals, self.max_initial_depth,static=True)
			ind.create(None, self.rng, Tr_X=self.Tr_x, Tr_Y=self.Tr_y, Te_X=self.Te_x, Te_Y = self.Te_y)
			self.forest.append(ind)
			
			weight = [0]*self.population_size*2
			weight[i]=1
			tmp = Individual(self.operators, self.terminals, self.max_initial_depth,static=True)
			tmp.create(weight, self.rng, semantics = ind.getSemantics(), Tr_X=self.Tr_x, Tr_Y=self.Tr_y, Te_X=self.Te_x, Te_Y = self.Te_y)
			self.population.append(tmp)
#

		self.bestIndividual = self.population[0]

		if not self.Te_x is None:
			self.trainingAccuracyOverTime = []
			self.testAccuracyOverTime = []
			self.trainingWaFOverTime = []
			self.testWaFOverTime = []
			self.trainingKappaOverTime = []
			self.testKappaOverTime = []
			self.trainingMSEOverTime = []
			self.testMSEOverTime = []
			self.generationTimes = []

		'''
		Training loop for the algorithm.
		'''
		if self.verbose:
			print("  > Running log:")

		while self.currentGeneration < self.max_generation:
			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			else:
				duration = 0
			self.currentGeneration += 1
			
			if not self.Te_x is None:
				if self.checkAccuracy:
					self.trainingAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
					self.trainingWaFOverTime.append(self.bestIndividual.getWaF(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testWaFOverTime.append(self.bestIndividual.getWaF(self.Te_x, self.Te_y, pred="Te"))
					self.trainingKappaOverTime.append(self.bestIndividual.getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testKappaOverTime.append(self.bestIndividual.getKappa(self.Te_x, self.Te_y, pred="Te"))
				else:
					self.trainingAccuracyOverTime.append(0)
					self.testAccuracyOverTime.append(0)
					self.trainingWaFOverTime.append(0)
					self.testWaFOverTime.append(0)
					self.trainingKappaOverTime.append(0)
					self.testKappaOverTime.append(0)

				if self.checkRMSE:
					self.trainingMSEOverTime.append(self.bestIndividual.getRMSE(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testMSEOverTime.append(self.bestIndividual.getRMSE(self.Te_x, self.Te_y, pred="Te"))
				else:
					self.trainingMSEOverTime.append(0)
					self.testMSEOverTime.append(0)
				
				self.generationTimes.append(duration)





	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation
		perfectTraining = self.bestIndividual.getRMSE(self.Tr_x, self.Tr_y, pred="Tr") == 1
		
		return genLimit  or perfectTraining




	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()

		# Calculates the accuracy of the population using multiprocessing
		if self.threads > 1:
			with mp.Pool(processes= self.threads) as pool:
				results = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y) for ind in self.population] )
				for i in range(len(self.population)):
					self.population[i].trainingPredictions = results[i][0]
					self.population[i].fitness = results[i][1]
					self.population[i].training_X = self.Tr_x
					self.population[i].training_Y = self.Tr_y
		else:
			#[ ind.fit(self.Tr_x, self.Tr_y) for ind in self.population]
			[ ind.getFitness() for ind in self.population ]

		# Sort the population from best to worse
		self.population.sort(reverse=True)


		# Update best individual
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]
			#self.bestIndividual.prun(min_dim = self.dim_min)

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend(getElite(self.population, self.elitism_size))
		while len(newPopulation) < self.population_size:
			offspring = getOffspring(self.rng, self.population, self.normalizedForest, self.tournament_size, self.mutation_step)
			newPopulation.extend(offspring)
		self.population = newPopulation[:self.population_size]


		end = time.time()


		# Debug
		if self.verbose and self.currentGeneration%5==0:
			if not self.Te_x is None:
				print("   > Gen #%2d:  Fitness: %.6f // Tr-Score: %.6f // Te-Score: %.6f  // Time: %.4f" % (self.currentGeneration, self.bestIndividual.getFitness(), self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y), self.bestIndividual.getAccuracy(self.Te_x, self.Te_y), end- begin )  )
			else:
				print("   > Gen #%2d:  Fitness: %.6f // Tr-Score: %.6f // Time: %.4f" % (self.currentGeneration, self.bestIndividual.getFitness(),  self.bestIndividual.getTrainingMeasure(), end- begin )  )



	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.population.getBestIndividual().predict(dataset)

		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)



def fitIndividuals(a):
	ind,x,y = a
	ind.getFitness(x,y)

	ret = []
	if "FOLD" in ind.fitnessType:
		ret.append(None)
	else:
		ret.append(ind.getTrainingPredictions())
	ret.append(ind.getFitness())

	
	return ret 
