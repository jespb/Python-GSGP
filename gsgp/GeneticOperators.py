from .Constants import *
from .Individual import Individual
from .Node import Node
from random import random, randint
from copy import copy

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

def getElite(population):
	return population[:ELITISM_SIZE]

# Currently, this method is not being used
def getOffspringStyleSTGP(population, normalizedForest):
	isCross = random()<0.9
	offspring = []
	if isCross:
		parents = [tournament(population),tournament(population)]

		osxo = crossover(parents)
		
		isMutation = random() < 0.1
		if isMutation:
			for i in range(len(osxo)):
				osxom = mutation(osxo[i], normalizedForest)
				offspring.extend(osxom)
		else:
			offspring.extend( osxo )
	
	else:
		parent = tournament(population)
		isMutation = random() < 0.1
		if isMutation:
			osm = mutation(parent, normalizedForest)
			offspring.extend(osm)
		else:
			offspring.append(parent)
	
	return offspring

def getOffspring(population, normalizedForest):
	isCross = random()<0.5
	offspring = []

	if isCross:
		parents = [tournament(population),tournament(population)]
		osxo = crossover(parents)
		offspring.extend( osxo )	
	else:
		parent = tournament(population)
		osm = mutation(parent, normalizedForest)
		offspring.extend(osm)
		
	return offspring

def tournament(population):
	candidates = [randint(0,len(population)-1) for i in range(TOURNAMENT_SIZE)]
	return population[min(candidates)]

def crossover(parents):
	ind1,ind2 = parents

	a = random()

	weights=[]
	for i in range(len(ind1.weights)):
		weights.append(ind1.weights[i]*a+ind2.weights[i]*(1-a))

	semantics = []
	for i in range(len(ind1.semantics)):
		semantics.append(ind1.semantics[i]*a+ind2.semantics[i]*(1-a))
	
	# offspring = (parent1) * a + (parent2) * (1-a)
	
	return [Individual(weights =weights, semantics=semantics)]


def mutation(parent, normalizedForest):
	weights = copy(parent.weights)
	tr1=int(random()*POPULATION_SIZE)
	tr2=int(random()*POPULATION_SIZE)

	weights[POPULATION_SIZE+tr1] += MUTATION_STEP
	weights[POPULATION_SIZE+tr2] -= MUTATION_STEP

	semantics = []
	for i in range(len(parent.semantics)):
		semantics.append(parent.semantics[i] + MUTATION_STEP * (normalizedForest[tr1].semantics[i]-normalizedForest[tr2].semantics[i]))
	
	# offspring = parent + ms * (tr1 - tr2)

	return [Individual(weights=weights, semantics=semantics)]