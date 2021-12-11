import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
import gensim
import codecs
from sner import Ner
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import spacy
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer



class QuestionClassification():

	def trainCorpora(self):
		f_train = open('data/training_set.txt', 'r+')
		f_validation = open('data/validation_set.txt', 'r+')

		#Assign into dataframe
		train = pd.DataFrame(f_train.readlines(), columns = ['Question'])
		test = pd.DataFrame(f_validation.readlines(), columns = ['Question'])

		#print(validation)

		train['QType'] = train.Question.apply(lambda x: x.split(' ', 1)[0])
		train['Question'] = train.Question.apply(lambda x: x.split(' ', 1)[1])
		train['QType-Coarse'] = train.QType.apply(lambda x: x.split(':')[0])
		#print(train['QType-Coarse'])

		test['QType'] = test.Question.apply(lambda x: x.split(' ', 1)[0])
		test['Question'] = test.Question.apply(lambda x: x.split(' ', 1)[1])
		test['QType-Coarse'] = train.QType.apply(lambda x: x.split(':', 1)[0])
		
		train.append(test).describe()

		#print(test.head())
		label_encoder = LabelEncoder()
		label_encoder.fit(pd.Series(train.QType.tolist() + test.QType.tolist()).values)
		train['QType'] = label_encoder.transform(train.QType.values)
		test['QType'] = label_encoder.transform(test.QType.values)
		
		#print(test.head())
		
		label_encoder_coarse = LabelEncoder()
		label_encoder_coarse.fit(pd.Series(train['QType-Coarse'].tolist() + test['QType-Coarse'].tolist()).values)
		train['QType-Coarse'] = label_encoder_coarse.transform(train['QType-Coarse'].values)
		test['QType-Coarse'] = label_encoder_coarse.transform(test['QType-Coarse'].values)
		
		#print(test.head())
		
		#get all corpus 
		corpus = pd.Series(train.Question.tolist() + test.Question.tolist()).astype(str)
		refrainment = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
		#print(corpus)
		#print(self.cleanText(corpus, refrainment))
		all_corpus = self.standartprocess(corpus, refrainment, stem_type = None)
		#print(all_corpus)
		#It should be used en_core_web_md instead of en
		#cls = util.get_lang_class(lang)
		#nlp = cls()
		nlp = spacy.load('en_core_web_md')
		#Split test and train data
		trainData = corpus[0:train.shape[0]]
		testData = corpus[train.shape[0]:]
		
		print("train", trainData)
		print("test", testData)
		
		all_ner = []
		all_lemma = []
		all_tag = []
		all_dep = []
		all_shape = []
		for row in trainData:
			#row.decode('utf8')
			doc = nlp(row.decode('utf-8'))
			present_lemma = []
			present_tag = []
			present_dep = []
			present_shape = []
			present_ner = []
			#print(row)
			for token in doc:
				present_lemma.append(token.lemma_)
				present_tag.append(token.tag_)
				#print(present_tag)
				present_dep.append(token.dep_)
				present_shape.append(token.shape_)
			all_lemma.append(" ".join(present_lemma))
			all_tag.append(" ".join(present_tag))
			all_dep.append(" ".join(present_dep))
			all_shape.append(" ".join(present_shape))
			for ent in doc.ents:
				present_ner.append(ent.label_)
			all_ner.append(" ".join(present_ner))
			
			#CountVectorizer
			count_vec_ner = CountVectorizer(ngram_range=(1, 2)).fit(all_ner)
			ner_ft = count_vec_ner.transform(all_ner)
			count_vec_lemma = CountVectorizer(ngram_range=(1, 2)).fit(all_lemma)
			lemma_ft = count_vec_lemma.transform(all_lemma)
			count_vec_tag = CountVectorizer(ngram_range=(1, 2)).fit(all_tag)
			tag_ft = count_vec_tag.transform(all_tag)
			count_vec_dep = CountVectorizer(ngram_range=(1, 2)).fit(all_dep)
			dep_ft = count_vec_dep.transform(all_dep)
			count_vec_shape = CountVectorizer(ngram_range=(1, 2)).fit(all_shape)
			shape_ft = count_vec_shape.transform(all_shape)

			x_all_ft_train = hstack([ner_ft, lemma_ft, tag_ft])
			x_all_ft_train = x_all_ft_train.tocsr()
			model = svm.LinearSVC()
			model.fit(x_all_ft_train, train['QType-Coarse'].values)
		
	
	def countVectorizerClassify(self):
		corpus = [
				'This is the first document.', 
				'This document is the second document',
				'This document is the third document'
				]
		
		#assign into a vectorizer
		vectorizer = CountVectorizer()
		X = vectorizer.fit_transform(corpus)
		#print(vectorizer.get_feature_names())
		#print("X result: ", X)
		
		
	def cleanText(self, corpus, refrainment):
		cleanedCorpus = pd.Series()
		
		for row in corpus:
			questions = []
			for words in row.split():
				if words not in refrainment:
					p1 = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=words)
					p1 = p1.lower()		
					questions.append(p1)
				else:
					questions.append(words)
			cleanedCorpus = cleanedCorpus.append(pd.Series(' '.join(questions)))
			
		return cleanedCorpus
		
	def standartprocess(self, corpus, refrainment, stem_type = None):
		
		corpus = self.cleanText(corpus, refrainment)
		wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
		stop=set(stopwords.words('english'))
		for word in wh_words:
			stop.remove(word)
		corpus = [[x for x in x.split() if x not in stop] for x in corpus]
		
		#lemmatization
		lemmatization = WordNetLemmatizer()
		corpus = [[lemmatization.lemmatize(x, pos = 'v') for x in x] for x in corpus]
		
		#stem type should be selected
		if stem_type == 'snowball':
			print('snowball stemmer')
			stemmer = SnowballStemmer(language='english')
			corpus = [[stemmer.stem(x) for x in x] for x in corpus]
		else: 
			print('PorterStemmer')
			stemmer = PorterStemmer()
			corpus = [[stemmer.stem(x) for x in x] for x in corpus]
		
		corpus = [' '.join(x) for x in corpus]
		
		return corpus
	
		
	
		