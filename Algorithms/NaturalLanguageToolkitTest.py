#Python TextBlob Test

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

blob = TextBlob("Don't give me the value of sensor1 in machine1?", analyzer=NaiveBayesAnalyzer())
print(type(blob.sentiment))
print(blob.sentiment)