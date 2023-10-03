import numpy as np

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019-2023 J. E. Batista
#

class Node:
	branches = None
	value = None


	def __init__(self):
		pass


	def create(self, rng, operators=None, terminals=None, depth=None,full=False):
		if depth>1 and (rng.random()<0.5 or full ==True ):
			op, n_args = operators[rng.randint(0,len(operators)-1)]
			self.value = op

			self.branches = []
			for i in range(n_args):
				n = Node()
				n.create(rng, operators, terminals, depth-1)
				self.branches.append(n)
		else:
			self.value = terminals[rng.randint(0,len(terminals)-1)] # Sem literais


	def copy(self,value=None, branches=None):
		self.branches = branches
		self.value=value


	def __str__(self):
		if self.branches == None:
			return str(self.value)
		else:
			if len(self.branches) == 2:
				return "( " + str(self.branches[0]) + " " + str(self.value) + " " + str(self.branches[1]) + " )"
			else:
				return str(self.value) + " ( " + " ".join( [str(b) for b in self.branches] ) + " )"


	def getSize(self):
		'''
		Returns the total number of nodes within this Node.
		'''
		if self.branches == None:
			return 1
		else:

			return 1 + sum( [b.getSize() for b in self.branches] )


	def getDepth(self):
		'''
		Returns the depth of this Node.
		'''
		if self.branches == None:
			return 1
		else:
			return 1 + max( [b.getDepth() for b in self.branches] )


	def clone(self):
		'''
		Returns a clone of this node.
		'''
		if self.branches == None:
			n = Node()
			n.copy(value=self.value, branches = None)
			return n
		else:
			n = Node()
			n.copy(value=self.value, branches=[b.clone() for b in self.branches])
			return n



	def calculate(self, sample):
		'''
		Returns the calculated value of a sample.
		'''
		if self.branches == None:
			try:
				return np.array( sample[self.value] )#.astype("float64")
			except:
				return np.array( [float(self.value)]*sample.shape[0] )

				
		else:
			if self.value == "+": #+
				return self.branches[0].calculate(sample) + self.branches[1].calculate(sample)
			if self.value == "-": #-
				return self.branches[0].calculate(sample) - self.branches[1].calculate(sample)
			if self.value == "*": #*
				return self.branches[0].calculate(sample) * self.branches[1].calculate(sample)
			if self.value == "/": #/
				right = self.branches[1].calculate(sample)
				right = np.where(right==0, 1, right)
				return self.branches[0].calculate(sample) / right
			if self.value == "log2": # log2(X)
				res = self.branches[0].calculate(sample)
				res = np.where(res<=0, res, np.log2(res))
				return res
			if self.value == "max": # max( X0, X1, ... Xn)
				calc = [b.calculate(sample) for b in self.branches]
				a = []
				for i in range(len(calc[0])):
					a.append( max([calc[k][i] for k in range(len(calc))]) )
				return np.array(a)
				

	def isLeaf(self):
		'''
		Returns True if the Node had no sub-nodes.
		'''
		return self.branches == None


