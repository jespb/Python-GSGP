from .Individual import Individual
from .Node import Node
from copy import copy

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

def getElite(population,n):
	'''
	Returns the "n" best Individuals in the population.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	return population[:n]

# Currently, this method is not being used
def getOffspringStyleSTGP(population, normalizedForest):
	isCross = random()<0.9
	offspring = []
	if isCross:
		parents = [tournament(rng, population, tournament_size),tournament(rng, population, tournament_size)]

		osxo = crossover(parents)
		
		isMutation = random() < 0.1
		if isMutation:
			for i in range(len(osxo)):
				osxom = mutation(osxo[i], normalizedForest)
				offspring.extend(osxom)
		else:
			offspring.extend( osxo )
	
	else:
		parent = tournament(rng, population, tournament_size)
		isMutation = random() < 0.1
		if isMutation:
			osm = mutation(parent, normalizedForest)
			offspring.extend(osm)
		else:
			offspring.append(parent)
	
	return offspring

def getOffspring(rng, population, normalizedForest, tournament_size, mutation_step):
	isCross = rng.random()<0.5
	offspring = []

	if isCross:
		parents = [tournament(rng, population, tournament_size),tournament(rng, population, tournament_size)]
		osxo = crossover(rng, parents)
		offspring.extend( osxo )	
	else:
		parent = tournament(rng, population, tournament_size)
		osm = mutation(rng, parent, normalizedForest, mutation_step)
		offspring.extend(osm)
		
	return offspring




def tournament(rng, population,n):
	'''
	Selects "n" Individuals from the population and return a 
	single Individual.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	candidates = [rng.randint(0,len(population)-1) for i in range(n)]
	return population[min(candidates)]



def crossover(rng, parents):
	ind1,ind2 = parents

	a = rng.random()

	weights=[]
	for i in range(len(ind1.weights)):
		weights.append(ind1.weights[i]*a+ind2.weights[i]*(1-a))

	semantics = []
	for i in range(len(ind1.semantics)):
		semantics.append(ind1.semantics[i]*a+ind2.semantics[i]*(1-a))
	
	# offspring = (parent1) * a + (parent2) * (1-a)
	ind = Individual(ind1.operators, ind1.terminals, ind1.max_depth)
	ind.create(weights, rng, semantics = semantics, Tr_X=ind1.training_X, Tr_Y=ind1.training_Y, Te_X=ind1.test_X, Te_Y=ind1.test_Y)

	return [ind]
	


def mutation(rng, parent, normalizedForest, mutation_step):
	pop_size = len(normalizedForest)
	weights = copy(parent.weights)
	tr1=int(rng.random()*pop_size)
	tr2=int(rng.random()*pop_size)

	weights[pop_size+tr1] += mutation_step
	weights[pop_size+tr2] -= mutation_step

	semantics = []
	for i in range(len(parent.semantics)):
		semantics.append(parent.semantics[i] + mutation_step * (normalizedForest[tr1].semantics[i]-normalizedForest[tr2].semantics[i]))
	
	# offspring = parent + ms * (tr1 - tr2)
	ind = Individual(parent.operators, parent.terminals, parent.max_depth)
	ind.create(weights, rng, semantics = semantics, Tr_X=parent.training_X, Tr_Y=parent.training_Y, Te_X=parent.test_X, Te_Y=parent.test_Y)

	return [ind]