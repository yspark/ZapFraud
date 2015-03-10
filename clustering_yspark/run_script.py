#!/usr/bin/python
import os, sys
from common_def import *

if len(sys.argv) != 3: 
  print './run_scrupt.py <start index> <end index>'

# preprocessing
#os.system("./preprocessing.py")


for ngram in [2,3,4,5,1]:
  print "********************************************************************"
  print "NGRAM: ", ngram
  print "********************************************************************"

  print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  print "generate_tfidf"
  print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  command = './generate_tfidf.py ' + str(ngram)
  os.system(command)

  print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  print "randomforest"
  print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  command = './randomforest.py ' + str(ngram)
  os.system(command)