import math

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#

def sigmoid(x):
	# Avoids overflow on the math.exp() function
	if x < -100:
		return 0
	if x > 100:
		return 1
	return 1 / ( 1 + math.exp(-x) )