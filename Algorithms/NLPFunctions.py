import json
import requests
import logging
import spacy
import re
from nltk.parse.stanford import StanfordParser

#Test html document
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
#print(soup)

class NLP():
	def dependencyTest(self):
		nlp = spacy.load('en_core_web_md')
		doc = nlp(u'What contains linkedfactory?')
		doc1 = nlp(u'What does linkedfactory contain?')
		doc2 = nlp(u'Could you tell me which one contains linkedfactory?')
		#print(doc.print_tree(light=True))
		print("\n")
		item1 = doc.print_tree(light=True)
		item2 = doc1.print_tree(light=True)
		item3 = doc2.print_tree(light=True)
		#print(item[0]['modifiers'][1]['POS_fine'])
		print(item1[-1]['modifiers'][1])
		print("\n")
		print(item2[-1]['modifiers'][2])
		print("\n")
		print(item3[0])

		
	def chunkDependency(self):
		nlp = spacy.load('en_core_web_md')
		doc = nlp(u'What contains linkedfactory?')
		print('CHUNKS')
		for chunk in doc.noun_chunks:
			pass
			#print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
		
		doc = nlp(u'What does linkedfactory contain?')
		print('CHUNKS Second')
		for chunk in doc.noun_chunks:
			print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
				
		doc = nlp(u'Find me the member which contains linkedfactory?')
		print('CHUNKS Third')
		listItem = []
		for chunk in doc.noun_chunks:
			listItem = chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text
			
		print(listItem)
		if 'dobj' in str(listItem):
			print("found")
		else:
			print("not found")
			
		chunk_text = [chunk.text for chunk in doc.noun_chunks]
		chunk_root = [chunk.root.text for chunk in doc.noun_chunks]
		chunk_root_dep = [chunk.root.dep_ for chunk in doc.noun_chunks]
		chunk_root_head = [chunk.root.head.text for chunk in doc.noun_chunks]
		print("\n")
		print("\n")
		print("\n")
		print(chunk_text)
		print(chunk_root)
		print(chunk_root_dep)
		print(chunk_root_head)

	def partOfSpeechTagging(self):
		nlp = spacy.load('en_core_web_md')
		doc = nlp(u'What are the members of linkedfactory?')
		doc = nlp(u'Give me all of members in linkedfactory?')
		doc = nlp(u'What is the value of sensor1 in machine1?')
		doc = nlp(u'What does linkedfactory contains?')
		
		for token in doc:
			print(token.text, token.lemma, token.pos_, token.tag_, token.dep_, 
					token.shape_, token.is_alpha, token.is_stop)
				
			
#chunkDependency()
obj = NLP()
obj.partOfSpeechTagging()