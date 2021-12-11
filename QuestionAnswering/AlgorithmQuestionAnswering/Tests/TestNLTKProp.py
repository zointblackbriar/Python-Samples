import unittest
import logging
from NLTKProp import NLTKProp
from StanfordSpacyNLP import TestConnectionCoreNLP
#import time

logger = logging.getLogger(__name__)

given_item = "incorporated"
expected_item = "incorporate"

class BaseTestClass(unittest.TestCase):
    def test_sentiment_analysis(self):
        logger.info("test_sentiment_analysis")
        import StanfordSpacyNLP
        corenlpObject = StanfordSpacyNLP.TestConnectionCoreNLP()
        statement = "Don't give me the value of sensor1 in machine1?"
        self.assertFalse(corenlpObject.textblob_sentiment_analysis(statement))

    def test_sentiment_analysis(self):
        logger.info("test_sentiment_analysis")
        import StanfordSpacyNLP
        corenlpObject = StanfordSpacyNLP.TestConnectionCoreNLP()
        statement = "Don't give me the value of sensor1 in machine1?"
        self.assertFalse(corenlpObject.textblob_sentiment_analysis(statement))

    def test_spacy_lemmatizer(self):
        logger.info("test_spacy_lemmatizer")
        nlpTask = TestConnectionCoreNLP()
        print(nlpTask.spacy_verb_lemmatizer(given_item))
        self.assertEqual(nlpTask.spacy_verb_lemmatizer(given_item), u' ' + expected_item)


    def test_snowball_stemmer(self):
        logger.info("test_snowball_stemmer")
        #statement = ['What contains linkedfactory?', 'What does linkedfactory contain?']
        #testverb = []
        #nlpTask = TestConnectionCoreNLP()
        #testverb.append(nlpTask.spacyArchMatching(statement[0]))
        #testverb.append(nlpTask.spacyArchMatching(statement[1]))
        #print(testverb[0])
        #print(testverb[1])
        #self.assertEqual(NLTKProp.stemmingSnowball(str(testverb[0])), "contains")
        #self.assertEqual(NLTKProp.stemmingSnowball(str(testverb[1])), "contains")
        print(NLTKProp.stemmingPorter(given_item))
        self.assertEqual(NLTKProp.stemmingSnowball(given_item), expected_item)

    #@unittest.skip("Snowball Stemmer will be tested")
    def test_porter_stemmer(self):
        logger.info("test_porter_stemmer")
        #statement = ['What contains linkedfactory?', 'What does linkedfactory contain?']
        #self.assertEqual(NLTKProp.stemmingPorter(statement[0]), "contains")
        #self.assertEqual(NLTKProp.stemmingPorter(statement[1]), "contains")
        print(NLTKProp.stemmingPorter(given_item))
        self.assertEqual(NLTKProp.stemmingPorter(given_item), expected_item)

    def test_lancaster_stemmer(self):
        logger.info("test_porter_stemmer")
        #statement = ['What contains linkedfactory?', 'What does linkedfactory contain?']
        #self.assertEqual(NLTKProp.stemming_lancaster(statement[0]), "contains")
        #self.assertEqual(NLTKProp.stemming_lancaster(statement[1]), "contains")
        print(NLTKProp.stemming_lancaster(given_item))
        self.assertEqual(NLTKProp.stemming_lancaster(given_item), expected_item)

    def test_lemmatization(self):
        logger.info("test lemmatization")
        print(NLTKProp.lemmatization(given_item))
        self.assertEqual(NLTKProp.lemmatization(given_item), expected_item)



if __name__ == 'main':
    unittest.main()