from random import randrange
import random
from subprocess import check_output as qx
import os
import subprocess
from distutils.core import setup
import operator
import optparse
import numpy as np
from scipy.stats import norm
import math
import subprocess, os, sys
import random
from random import randint
from sklearn.ensemble import *
import operator
import timeit
# Random design generator
INF = np.inf


def RGen(nVL, budget):

	randDesign = [0 for i in range(8*nVL)]
	for budgetLeft in range(budget):
		randTSV = random.choice([i for i in range(len(randDesign))])
		randDesign[randTSV] +=1

	return randDesign

def RGenAlternative(nVL, budget):
	design = list()
	design = [[0 for i in range(8)] for j in range(nVL)]

	vlIndexList = list()

	for i in range(nVL):
		if i in [k for k in range(16,32)]:
			vlIndexList.append(i)
			vlIndexList.append(i)
		else:
			vlIndexList.append(i)

	tsvIndexList = list()

	for i in range(8):
		if i in [0,2,6]:
			tsvIndexList.append(i)
		elif i in [1,3,5,7]:
			tsvIndexList.append(i)
			tsvIndexList.append(i)
		else:
			for j in range(12):
				tsvIndexList.append(i)

	for budgetLeft in range(budget):
		i = random.choice(vlIndexList)
		j = random.choice(tsvIndexList)
		design[i][j] = design[i][j] + 1

	design = np.array(design)
	design = design.ravel()
	design = list(design)

	return design


# Successor State generation alternative version with more successors

def SGenAlternative(d_0, budget):
	designList = list()
	newDesign = list()

	for TSV_to_increase in range(len(d_0)):
		if d_0[TSV_to_increase] == budget:
			pass
		else:
			newDesign = d_0[:]
			newDesign[TSV_to_increase] = d_0[TSV_to_increase] + 1
			for designNumber in range(len(d_0)):
				if designNumber == TSV_to_increase or newDesign[designNumber] == 0:
					pass
				else:
					decrementDesign = newDesign[:]
					decrementDesign[designNumber] = newDesign[designNumber] - 1
					designList.append(decrementDesign)


	return designList


def SGenAlternativeWithPriority(d_0, budget):
	designList = list()
	newDesign = list()
	designDict = dict()

	for TSV_to_increase in range(len(d_0)):
		if d_0[TSV_to_increase] == budget:
			pass
		else:
			newDesign = list(d_0)
			newDesign[TSV_to_increase] = int(d_0[TSV_to_increase]) + 1
			for designNumber in range(len(d_0)):
				if designNumber == TSV_to_increase or newDesign[designNumber] == 0:
					pass
				else:
					decrementDesign = newDesign[:]
					decrementDesign[designNumber] = newDesign[designNumber] - 1
					designList.append(decrementDesign)

	for item in designList:
		designDict[tuple(item)] = designScore(item, budget)

	tempList = list()
	sortedDesigns = sorted(designDict.items(), key = operator.itemgetter(1))
	for item in sortedDesigns:
		tempList2 = list(item)
		tempList.append(tempList2[0])

	del designList[:]
	for j in tempList[::-1]:
		designList.append(j)

	return designList

def designScore(design, budget):
	total = 0

	middleTSVindexes = list()
	middleTSVindexes = [i for i in xrange(4,len(design),8)]
	secondPriorityIndexes = list()
	secondPriorityIndexes = [j for j in xrange(1, len(design), 2)]

	for i in range(len(design)):
		if i in middleTSVindexes:
			total = total + (10 * design[i])
		elif i in secondPriorityIndexes:
			total = total + (2 * design[i])
		else:
			total = total + design[i]
	secTotal = 0


	twoDimDesign = [design[i:i+8] for i in range(0, len(design), 8)]
	for i in range(len(twoDimDesign)):
		if i in [j for j in range(16,32)]:
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
		else:
			secTotal = secTotal + sum(twoDimDesign[i])


	return total + secTotal


# Dummy simulator function
def getDummyLifetime(design):

	return dummySimulate(design)

	start_time = timeit.default_timer()
	params = {}
	design_length = len(design)

	# print design
	with open("sTSVsData.txt", "w") as text_file:

		for line_i in range(0, 48):
			first = True
			for i in range(0, 8):
				if first:
					text_file.write(str(design[line_i * 8 + i]))
					first = False
				else:
					text_file.write(" " + str(design[line_i * 8 + i]))
			text_file.write("\n")

	# os.system('BO_NoC_MultiFrame.exe')
	# fileName = "F:\\Rakib\\NoC\\Canneal\\timeVsEDP_lifetime.txt"
	print(design)

	p = subprocess.Popen(["BO_NoC_MultiFrame.exe"], shell=True, 
			cwd="./", 
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
	# p.wait()
	out, err = p.communicate()
	if(err):
		print("Error in running simulator ", p.returncode, err)
		return 0

	with open('timeVsEDP_lifetime.txt', 'r') as f:
		last = None
		for line in (line for line in f if line.rstrip('\n')):
			last = line

	listSplitted = last.split()
	# print listSplitted[0]
	result = float(listSplitted[0])
	elapsed = timeit.default_timer() - start_time
	with open("TimeToRunSimulator.txt", "a") as myfile:
		myfile.write(str(elapsed))
		myfile.write("\n")
	# SMAC has a few different output fields; here, we only need the 4th output:
	print "Result of algorithm run: SUCCESS, 0, 0, %f, 0" % (result)
	return result

def dummySimulate(design):
	total = 0

	middleTSVindexes = list()
	middleTSVindexes = [i for i in xrange(4,len(design),8)]
	secondPriorityIndexes = list()
	secondPriorityIndexes = [j for j in xrange(1, len(design), 2)]

	for i in range(len(design)):
		if i in middleTSVindexes:
			total = total + (5 * design[i])
		elif i in secondPriorityIndexes:
			total = total + (2 * design[i])
		else:
			total = total + design[i]
	secTotal = 0


	twoDimDesign = [design[i:i+8] for i in range(0, len(design), 8)]
	for i in range(len(twoDimDesign)):
		if i in [j for j in range(16,32)]:
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
			secTotal = secTotal + sum(twoDimDesign[i])
		else:
			secTotal = secTotal + sum(twoDimDesign[i])


	return total + secTotal

# Local search method
def steepestHillClimbingBase(initialDesign,previous_best, clf,budget):
	start_time = timeit.default_timer()
	trajectory = list()
	trajectory_value = ExpectedImprovement(initialDesign, previous_best, clf)
	trajectory.append(initialDesign)

	currentTrajectoryValue = 0
	loopFlag = False
	while loopFlag == False:
		currentTrajectoryValue = trajectory_value
		possibleSteps = SGenAlternativeWithPriority(trajectory[-1], budget)

		for item in possibleSteps:
			# print len(item)
			itemValue = ExpectedImprovement(item, previous_best, clf)
			if itemValue > currentTrajectoryValue:
				trajectory.append(list(item))
				trajectory_value = itemValue
				break
		if int(trajectory_value) <= int(currentTrajectoryValue):
			loopFlag = True
	elapsed = timeit.default_timer() - start_time
	with open("baseSearchDurationStage.txt","a") as myfile:
		myfile.write(str(elapsed))
		myfile.write("\n")

	return trajectory, trajectory_value

def steepestHillClimbingMeta(initialDesign, budget, mf_train_points, mf_train_labels):
	start_time = timeit.default_timer()
	trajectory = list()

	clf = RandomForestRegressor(n_estimators = 15)
	clf.fit(mf_train_points, mf_train_labels)
	predMean = float(clf.predict([initialDesign])[0])
	trajectory_value = predMean

	trajectory.append(initialDesign)
	currentTrajectoryValue = 0
	num_loops = 0
	loopFlag = False
	while loopFlag == False:
		currentTrajectoryValue = trajectory_value
		possibleSteps = SGenAlternativeWithPriority(trajectory[-1], budget)
		# print("Num alternatives ",len(possibleSteps))

		for item in possibleSteps:
			itemValue = float(clf.predict([item])[0])
			if itemValue > currentTrajectoryValue:
				trajectory.append(item)
				trajectory_value = itemValue
				break
		if int(trajectory_value) <= int(currentTrajectoryValue):
			loopFlag = True

		num_loops += 1
	# print("Num loops ", num_loops)

	for i in range(3064):
		design_rand = RGenAlternative(48, budget)
		itemValue = float(clf.predict([design_rand])[0])
		if itemValue > trajectory_value:
			print("Random ", i, trajectory_value)
			trajectory.append(item)
			trajectory_value = itemValue

	elapsed = timeit.default_timer() - start_time
	with open("metaSearchDurationStage.txt", "a") as myfile:
		myfile.write(str(elapsed))
		myfile.write("\n")

	return trajectory, trajectory_value



# Acquisition Function EI
def ExpectedImprovement(testDesign, previous_best_value, clf):
	previous_best_value = previous_best_value + 0.001
	predictive_mean = clf.predict([testDesign])[0]
	per_tree_pred = [tree.predict([testDesign])[0] for tree in clf.estimators_]
	predictions = list()
	for item in per_tree_pred:
		predictions.append(float(item))
	predictive_std = np.std(np.array(predictions))

	if predictive_std == 0:
		EIvalue = 0
		PIvalue = 0
	else:
		EIvalue = (((predictive_mean - previous_best_value)*(norm.cdf((float(predictive_mean - previous_best_value)/float(predictive_std)), loc = 0, scale = 1))) + (predictive_std * (norm.pdf((float(predictive_mean - previous_best_value)/float(predictive_std)), loc = 0, scale = 1))))
		PIvalue = norm.cdf((float(predictive_mean - previous_best_value)/float(predictive_std)), loc = 0, scale = 1)
		# print("EI Value ", EIvalue, predictive_mean, previous_best_value, predictive_std)
	return EIvalue



def initialRGen(nVL, budget):
	indexList = list()
	initDesList = list()
	indexList = [i for i in range(8*nVL)]

	for j in range(5):
		randDesign = [0 for k in range(8*nVL)]
		for budgetLeft in range(budget):
			randTSV = random.choice(indexList)
			randDesign[randTSV] +=1
			indexList.remove(randTSV)
		initDesList.append(randDesign)



	return initDesList

def get_already_generated_data(filepath):
	with open(filepath) as f:
	    content = f.readlines()
	value_list = []
	design_list = []
	for x in content[2:]: # Skipping the first desing and value
		split_x = x.strip().split()
		# print(len(split_x))
		if(len(split_x) > 1 and len(split_x) < 20):
			# print(split_x)
			value_list.append(float(split_x[-1]))
		if(len(split_x) > 1 and len(split_x) > 50):
			new_split_x = x[1:-2].strip().split(",")
			# print(new_split_x, len(new_split_x))
			design = []
			for i in new_split_x:
				design.append(int(i.strip()))
			#print(design, len(design))
			design_list.append(design)
	return design_list, value_list

def main():
	parser = optparse.OptionParser()

	parser.add_option('-v', '--nVL', action="store", dest="nVL", help="number of VLs in input structure", default="48")
	parser.add_option('-s', '--nSpares', action="store", dest="nSpares", help="number of spare TSVs available", default="9")
	parser.add_option('-q', '--qMAX', action="store", dest="maxSimEval", help="maximum number of allowed simulator calls", default="1000")
	parser.add_option('-t', '--stageMAX', action="store", dest="maxStageEval", help="maximum number of allowed stage runs", default="15")
	options, args = parser.parse_args()

	print 'Number of vertical links: ', options.nVL
	print 'Number of spare TSVs available: ', options.nSpares
	print 'Maximum number of simulator evaluations allowed: ', options.maxSimEval
	print 'Mximum number of allowed stage algorithm run: ', options.maxStageEval

	budget = int(options.nSpares)
	nVL = int(options.nVL)
	maxSimEval = int(options.maxSimEval)
	maxStageEval = int(options.maxStageEval)

	# Initialization

	surrogateModelTrain = list()
	surrogateModelTrainValue = list()
	tempDataList = list()

	# surrogateModelTrain.append(goodSol)
	# surrogateModelTrainValue.append(89698.5)

	# Open all the necessary files and set it to zero
	# with open("modelTrainDataStage.txt", "w") as myfile:
	# 	myfile.write("Just to indicate start of file")
	design_list, value_list = get_already_generated_data("modelTrainDataStage.txt")
	print(len(design_list))
	print(len(value_list))
	
	with open("baseSearchDurationStage.txt","w") as myfile:
		myfile.write("Just to indicate start of file")
	with open("metaSearchDurationStage.txt", "w") as myfile:
		myfile.write("Just to indicate start of file")
	with open("TimeToRunSimulator.txt", "w") as myfile:
		myfile.write("Just to indicate start of file")

	# for i in range(2):
	# 	secondDesign = RGenAlternative(nVL, budget)
	# 	surrogateModelTrain.append(secondDesign)
	# 	surrogateModelTrainValue.append(float(getDummyLifetime(secondDesign)))

	for i in range(len(value_list)):
		surrogateModelTrain.append(design_list[i])
		surrogateModelTrainValue.append(float(value_list[i]))



	d_best = list()



	d_best = surrogateModelTrain[0][:]
	O_best = surrogateModelTrainValue[0]
	bestFunctionValueIndex = 0
	for i in range(1 , len(surrogateModelTrainValue)):
		if surrogateModelTrainValue[i] > O_best:
			d_best = surrogateModelTrain[i][:]
			O_best = surrogateModelTrainValue[i]
			bestFunctionValueIndex = i

	

	for iteration in range(maxSimEval):
		# Get an initial design for STAGE
		initialDesignStage = RGenAlternative(nVL, budget)
		d_EIbest = initialDesignStage[:]
		d_base_0 = initialDesignStage[:]


		# Train the surrogate model
		clf = RandomForestRegressor(n_estimators = 15)
		clf.fit(surrogateModelTrain, surrogateModelTrainValue)

		# Get the initial design
		O_EIbest = ExpectedImprovement(d_base_0, surrogateModelTrainValue[bestFunctionValueIndex],clf)
		baseSearchTrajectory = list()
		evalFnDataPoints = list()
		evalFnDataPointsLabel = list()
		metaSearchTrajectory = list()

		print("EI Best ", O_EIbest, maxStageEval, maxSimEval)

		for acqIteration in range(maxStageEval):
			baseSearchTrajectory, baseSearchTrajectoryValue = steepestHillClimbingBase(d_base_0, surrogateModelTrainValue[bestFunctionValueIndex],clf, budget)
			for item in baseSearchTrajectory:
				evalFnDataPoints.append(item)
				evalFnDataPointsLabel.append(baseSearchTrajectoryValue)

			print("EI Best ", acqIteration, baseSearchTrajectoryValue, evalFnDataPointsLabel)

			if False:
				metaSearchTrajectory, metaSearchTrajectoryValue = steepestHillClimbingMeta(baseSearchTrajectory[-1],budget, evalFnDataPoints, evalFnDataPointsLabel)

				# STAGE should use the Actuall EI and not the Meta Search value
				metaSearchTrajectoryValue = ExpectedImprovement(metaSearchTrajectory[-1][:], surrogateModelTrainValue[bestFunctionValueIndex],clf)

				if baseSearchTrajectoryValue == metaSearchTrajectoryValue:
					d_base_0 = RGenAlternative(nVL, budget)
				else:
					d_base_0 = metaSearchTrajectory[-1][:]

				if max([baseSearchTrajectoryValue, metaSearchTrajectoryValue]) > O_EIbest:
					O_EIbest = max([baseSearchTrajectoryValue, metaSearchTrajectoryValue])
					if baseSearchTrajectoryValue > metaSearchTrajectoryValue:
						d_EIbest = baseSearchTrajectory[-1][:]
					else:
						d_EIbest = metaSearchTrajectory[-1][:]

			if True:
				max_rand_value = -999999999
				for i in range(10):
					design_rand = RGenAlternative(nVL, budget)
					design_rand_value = ExpectedImprovement(design_rand, \
						surrogateModelTrainValue[bestFunctionValueIndex], clf)
					if(design_rand_value > max_rand_value):
						max_rand_value = design_rand_value
						max_rand_design = design_rand
				d_base_0 = max_rand_design

				if(baseSearchTrajectoryValue > O_EIbest):
					O_EIbest = baseSearchTrajectoryValue
					d_EIbest = baseSearchTrajectory[-1][:]




		if d_EIbest not in surrogateModelTrain:
			surrogateModelTrain.append(d_EIbest)
			d_EIbestVal = float(getDummyLifetime(d_EIbest))
			surrogateModelTrainValue.append(d_EIbestVal)
			with open("modelTrainDataStage.txt", "a") as myfile:
				myfile.write(str(surrogateModelTrain[-1]))
				myfile.write("\n has a lifetime value of: ")
				myfile.write(str(surrogateModelTrainValue[-1]))
				myfile.write("\n\n")

		maximumTempValue = 0
		for i in range(len(surrogateModelTrainValue)):
			if surrogateModelTrainValue[i] > maximumTempValue:
				bestFunctionValueIndex = i
				maximumTempValue = surrogateModelTrainValue[i]

	prettyDesign = [surrogateModelTrain[bestFunctionValueIndex][i:i+8] for i in range(0, len(surrogateModelTrain[bestFunctionValueIndex]), 8)]

	for item in prettyDesign:
		print item


	print surrogateModelTrainValue[bestFunctionValueIndex]


main()
