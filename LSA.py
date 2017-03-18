import nltk
from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics

import re
import numpy as np
import pandas as pd
from itertools import chain
import string
          
  
class NormalizeText:
     def __init__(self):
          self.trans_table = self.get_no_punctuation_table()
          self.PorterStemmer = PorterStemmer()

     def get_no_punctuation_table(self,translate_to=None):
          not_letters_or_digits = unicode(string.punctuation)+'1234567890'
          translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
          return translate_table    
     

     def normalize(self,text):
          orig_text = text
          text = unicode(text)
          trans_table = self.trans_table
          text = text.translate(trans_table)
          text = text.lower()
          text = nltk.word_tokenize(text)
          text = [self.PorterStemmer.stem(word) for word in text if not word in stopwords.words('english')]
          text = " ".join(text)
          return text
     

class MyCorpus:
     def __init__(self, filepath, delim=','):
          self.filepath = filepath
          self.delim = delim
     
     def __iter__(self): #each line assumed to be a separate document
          for line in open(self.filepath):
               yield line.split(self.delim)[1]


class LSA:
     def __init__(self):
          pass

     #singular value decomposition on the term-document matrix
     def fit(self,corpus,min_term_freq,max_df_prop,n_components):
          self.transformer = TfidfVectorizer(min_df=int(min_term_freq),max_df=float(max_df_prop))
          tfidf = self.transformer.fit_transform(corpus) 
          svd = TruncatedSVD(n_components = n_components)
          lsa = svd.fit_transform(tfidf.T)
          self.model = lsa

     
     
class Similarity:
     def __init__(self,transformer,model):
          self.transformer = transformer
          self.model = model
          self.norm = NormalizeText()
     
     #cosine similarity
     def cos_sim(self,X_col,Y_col):
          dot_prod = np.dot(X_col.T,Y_col)
          X_norm = np.sqrt(np.dot(X_col.T,X_col))
          Y_norm = np.sqrt(np.dot(Y_col.T,Y_col))
          cos = dot_prod/(X_norm*Y_norm)
          return cos
     
     def k_closest_terms(self,k,term):
          term = self.norm.normalize(term)
          index = self.transformer.vocabulary_[term]
          
          terms = self.transformer.get_feature_names()
          closestTerms = pd.Series(index=terms)
          
          for i in range(len(self.model)):
               closestTerms.loc[terms[i]] = self.cos_sim(self.model[index,:].T,self.model[i,:].T)
               
          closestTerms.sort_values(ascending=False,inplace=True)
                      
          return closestTerms.index[0:k].tolist()          
          
     
     
if __name__=="__main__":
     corpus_file = "corpus_text.txt"
     min_term_freq = 10 #minimum number of times the word must appear in the corpus
     max_df_prop = 0.10 #maximum proportion of documents in which the word appears
     n_components = 200 #number of components for dimensionality reduction
     
     
     #initialize corpus
     corpus = MyCorpus(corpus_file) #each line in corpus file assumed to be separate document
     
     #fit LSA model
     lsa = LSA()
     lsa.fit(corpus, min_term_freq, max_df_prop,n_components)
     
     #find k most similar terms
     term = 'unethical'
     k = 50
     similarity = Similarity(lsa.transformer,lsa.model)
     k_closest_terms = similarity.k_closest_terms(k=k,term=term)
     print k_closest_terms
