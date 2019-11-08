import pandas

from gsgp.GSGP import GSGP
from sys import argv
from gsgp.Constants import *
import os

import time

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GSGP
#
# Copyright ©2019 J. E. Batista
#


timestamp = time.strftime("%Y%m%d_%H%M")



def callgsgp():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:#["trio_brasil.csv","trio_congo.csv","trio_mocambique.csv","trio_combo.csv"]:#["mcd3.csv","mcd10.csv","brasil.csv","movl.csv","heart.csv","vowel.csv","wav.csv","yeast.csv","seg.csv"]:
		openFile(OUTPUT_DIR+"tmp_gsgp_"+timestamp + "_"+dataset)
		writeToFile(dataset+"\n")
		toWrite=[]
		for i in range(RUNS):
			print(i,"# run with the", dataset,"dataset")
			p = pandas.read_csv(DATASETS_DIR+dataset)
			gsgp = GSGP(p)

			writeToFile(",")
			for i in range(MAX_GENERATION):
				writeToFile(str(i)+",")
			
			accuracy = gsgp.getAccuracyOverTime()
			rmse = gsgp.getRmseOverTime()
			size = gsgp.getSizeOverTime()
			toWrite.append([accuracy[0],accuracy[1],rmse[0],rmse[1],size,gsgp.getFormatedModel()])
			
			writeToFile("\nTraining-Accuracy,")
			for val in accuracy[0]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTest-Accuracy,")
			for val in accuracy[1]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTraining-RMSE,")
			for val in rmse[0]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTest-RMSE,")
			for val in rmse[1]:
				writeToFile(str(val)+",")

			writeToFile("\nSize,")
			for val in size:
				writeToFile(str(val)+",")

			writeToFile("\n"+gsgp.getFormatedModel()+"\n")
		
		closeFile()

		openFile(OUTPUT_DIR+"gsgp_"+timestamp + "_"+dataset) 
		writeToFile("Attribute,Run,")
		for i in range(MAX_GENERATION):
			writeToFile(str(i)+",")
		writeToFile("\n")
		
		attributes= ["Training-Accuracy","Test-Accuracy","Training-RMSE","Test-RMSE","Size","Dimensions","Final_Model"]
		for ai in range(len(toWrite[0])-1):
			for i in range(len(toWrite)):
				writeToFile("\n"+attributes[ai]+","+str(i)+",")
				for val in toWrite[i][ai]:
					writeToFile(str(val)+",")
				#writeToFile(",".join(toWrite[i][ai]))
			writeToFile("\n\n")
		for i in range(len(toWrite)):
			writeToFile("\n"+attributes[-1]+","+str(i)+",")
			writeToFile(str(toWrite[i][-1]))
		writeToFile("\n\n")

		
		closeFile()
		os.remove(OUTPUT_DIR+"tmp_gsgp_"+timestamp + "_"+dataset)

callgsgp()