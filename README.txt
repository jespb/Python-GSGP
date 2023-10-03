By using this file, you are agreeing to this product's EULA
This product can be obtained in https://github.com/jespb/Python-GSGP
Copyright ©2019 J. E. Batista


This implementation of GSGP uses the following command and flags:

$ python Main_GSGP.py
	
	[-d datasets] 
		- This flag expects a set of csv dataset names separated by ";" (e.g., a.csv;b.csv)
		- By default, the heart.csv dataset is used		

	[-dontshuffle]
		- By using this flag, the dataset will not be shuffled;
		- By default, the dataset is shuffled.

	[-dsdir dir] 
		- States the dataset directory. 
		- By default "datasets/" is used 
		- Use "-dsdir ./" for the root directory	

	[-es elite_size]
		- This flag expects an integer with the elite size;
		- By default, the elite has size 1.

	[-md max_depth]
		- This flag expects an integer with the maximum initial depth for the trees;
		- By default, this value is set to 6.		

	[-mg max_generation]
		- This flag expects an integer with the maximum number of generations;
		- By default, this value is set to 1000.
	
	[-ms mutation_step]
		- This flag expects a float with the mutation step size;
		- By default, the mutation step used is 0.1

	[-odir dir] 
		- States the output directory. 
		- By default "results/" is used 
		- Use "-odir ./" for the root directory
	
	[-op operators]
		- This flag excepts a set of operators separated by ";"
		- Allowed operators: +;-;*;/
		- By default, the used operators are the sum, subtraction, multiplication and protected division.		

	[-ps population_size]
		- This flag expects an integer with the size of the population;
		- By default, this value is set to 500.

	[-r] 
		- States the this is a regression problem. 
		- By default the GSGP tries to classify samples as 0 or 1

	[-runs number_of_runs] 
		- This flag expects an integer with the number of runs to be made;
		- By default, this values is set to 30
	
	[-tf train_fraction]
		- This flag expects a float [0;1] with the fraction of the dataset to be used in training;
		- By default, this value is set to 0.70
	
	[-ts tournament_size]
		- This flag expects an integer with the tournament size;
		- By default, this value is set to 10.



How to import this implementation to your project:
    - Download this repository;
    - Copy the "gsgp/" directory to your project directory;
    - import the GSGP class using "from gsgp.GSGP import GSGP".

How to use this implementation:
    $ from gsgp.GSGP import GSGP
    $ model = GSGP()
    $ model.fit( training_x, training_y, test_x (optional), test_y (optional) 



Arguments for M3GP():
    operators			-> Operators used by the individual (default: [("+",2),("-",2),("*",2),("/",2)] )
    max_initial_depth	-> Max initial depths of the individuals (default: 6)
    population_size		-> Population size (default: 500)
    max_generation		-> Maximum number of generations (default: 1000)
    tournament_size		-> Tournament size (default: 5)
    elitism_size		-> Elitism selection size (default: 1)
    mutation_step		-> Mutation step value (default: 0.1)
    threads 			-> Number of CPU threads to be used (default: 1)
    random_state		-> Random state (default: 42)
    verbose				-> Console prints during training (default: True)


Arguments for model.fit():
    Tr_X 		-> Training samples
    Tr_Y 		-> Training labels
    Te_X 		-> Test samples, used in the standalone version (default: None)
    Te_Y 		-> Test labels, used in the standalone version (default: None)


Useful methods:
    $ model = GSGP()	-> starts the model;
    $ model.fit(X, Y)	-> fits the model to the dataset;
    $ model.predict(X)	-> Returns a list with the prediction of the given dataset. (use after training)




How to edit this implementation:
    Fitness Function ( gsgp.Individual ):
        - Change the getFitness() method to use your own fitness function;
        - This implementation assumes that a higher fitness is always better. To change this, edit the __gt__ method in this class.





Reference:
    Moraglio A., Krawiec K., Johnson C.G. (2012) Geometric Semantic Genetic Programming. In: Coello C.A.C., Cutello V., Deb K., Forrest S., Nicosia G., Pavone M. (eds) Parallel Problem Solving from Nature - PPSN XII. PPSN 2012. Lecture Notes in Computer Science, vol 7491. Springer, Berlin, Heidelberg



