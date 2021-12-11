import numpy as np

np.random.seed(7)


class KerasAnalyze():

	@staticmethod
	def loadDataSet():
		dataset = np.loadtxt("data/dataset.txt", delimiter=',')
		independentVariable = dataset[:, 0:8]
		dependentVariable = dataset[:, 8]
		#print(dataset)
		print(independentVariable)
		print(dependentVariable)

		
		
KerasAnalyze.loadDataSet()