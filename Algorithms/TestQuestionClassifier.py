from QuestionClassifier import QuestionClassification
import unittest

"""class BaseTest(unittest.TestCase):
	
	def testSklearn():
		obj = QuestionClassification()
		obj.countVectorizerClassify()
		obj.trainCorpora()
		


if __name__ == 'main':
	unittest.main()"""
	
class RunEverything():
	def testSklearn(self):
		obj = QuestionClassification()
		obj.countVectorizerClassify()
		obj.trainCorpora()

nextOb = RunEverything()
nextOb.testSklearn()