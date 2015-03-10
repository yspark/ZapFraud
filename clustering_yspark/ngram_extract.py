#!/usr/bin/python

import os, sys, re
import sklearn
import operator

from common_def import *

import numpy as np

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


class Ngram_Extractor:
  def generate_ngram(self, tokens, ngram):
    ngram_tokens = []
    ngram_list = []
    for i in range(len(tokens)-ngram):
      ngram_tokens.append("_".join(tokens[i:i+ngram]))
    return ngram_tokens

  def extract_ngarm(self, body, ngram=1):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokenizer = lambda doc: token_pattern.findall(doc)
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    # remove abbreviation
    body = body.replace("won't", "would not")
    body = body.replace("n't", " not")
    body = body.replace("'m", " am")
    body = body.replace("'re", " are")
    body = body.replace("'d ", " would ")
    body = body.replace('did', 'do')
    body = body.replace('\'ve', ' have')
    body = body.replace('\'ll', ' will')
    body = body.replace('\'s', '')
    body = body.replace('$', ' DOLLARMARK ')
    
    # remove all non-alphabetic characters
    body = re.sub(r'[^a-z$ ]', ' ', body)
    
    # tokenize
    tokens = word_tokenize(body)
    # stemming
    tokens = [stemmer.stem(w) for w in tokens]
    #remove stop words
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    #remove 1-char tokens
    tokens = [w for w in tokens if len(w) > 1]
    
    # N-gram
    if ngram > 1:
      tokens = self.generate_ngram(tokens, ngram)

    '''
    print 'XXXXXXXXXXXXXXXXXX'
    print filename
    print '=================='
    print body
    print '=================='
    print tokens
    print 'XXXXXXXXXXXXXXXXXX'
    '''

    return tokens


  def extract_body(self, filename):
    inbody = False
    body = category = ''
    for line in open(filename).readlines():
      if inbody == False:
        if line.startswith('category:'):
          category = line.replace('category:', '').strip()
        if line.startswith('body:'):
          body = line.replace('body:', '').strip().lower() + '\n'
          inbody = True

      elif inbody == True:
        body += line.lower() + '\n'
    #endfor

    return category, body




def main():
  if len(sys.argv) == 2:
    ngram = int(sys.argv[1])
  else:
    ngram = 1
  
  ngram_extractor = Ngram_Extractor()

  for directory in CATEGORY_DIRECTORY_LIST:
    tf_dic = {}
    df_dic = {}
    tfidf_dic = {}
    doc_counter_dic = {}
    
    n_doc = 0
    print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    print 'DIRECTORY: ', directory
    print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    
    for (dirpath, dirnames, filenames) in os.walk(directory):
      for filename in filenames:  
        if '.txt' not in filename: continue    
        filepath = dirpath + '/' + filename
      
        inbody = False
        body = ''
        for line in open(filepath).readlines():
          if inbody == False:
            if line.startswith('category:'):
              category = line.replace('category:', '').strip().lower()
            if line.startswith('body:'):
              body = line.replace('body:', '').strip().lower() + '\n'
              inbody = True
          
          elif inbody == True:
            body += line.lower() + '\n'
        #endfor 
        
        if body == '': continue
        n_doc += 1
        if category not in doc_counter_dic: 
          doc_counter_dic[category] = 1
        else:
          doc_counter_dic[category] += 1
        #if n_doc > 100: break
       
        tokens = ngram_extractor.extract_ngarm(body)

        if category not in tf_dic:
          tf_dic[category] = {}
          df_dic[category] = {}

        for token in tokens:
          if token not in tf_dic[category]:
            tf_dic[category][token] = 1
          else:
            tf_dic[category][token] += 1
        
        for token in set(tokens):
          if token not in df_dic[category]:
            df_dic[category][token] = 1
          else:
            df_dic[category][token] += 1
   
      #endfor
    #endfor

    '''
    for category in tf_dic:
      if category not in tfidf_dic:
        tfidf_dic[category] = {}
      for term in tf_dic[category]:
        tfidf_dic[category][term] = tf_dic[category][term] / (np.log(n_doc/df_dic[category][term]) + 1)
    '''
    
    tf_sorted = {}
    tfidf_sorted = {}
    for category in tf_dic:
      if category not in tf_sorted:
        tf_sorted[category] = {}
        tfidf_sorted[category] = {}

      tf_sorted[category] = sorted(tf_dic[category].items(), key=operator.itemgetter(1), reverse=True)
      #tfidf_sorted[category] = sorted(tfidf_dic[category].items(), key=operator.itemgetter(1), reverse=True)

      print '====================================================='
      print 'TermFrequency source:%s, category:%s, doc#:%d' %  (directory.split('/')[-1], category, doc_counter_dic[category])
      print '====================================================='
      for key, value in tf_sorted[category][:200]:
        print key, value, df_dic[category][key], float(df_dic[category][key])/float(doc_counter_dic[category])
      '''
      print '====================================================='
      print 'TFIDF', category
      print tfidf_sorted[category][:100]
      '''
    #endfor

    #sys.exit()
  #endfor


if __name__ == '__main__':
  main()