#!/usr/bin/python

import pickle
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from common_def import *


class RandomForest:
	def __init__(self, ngram=1):
		self.ngram = ngram
		self.clf = RandomForestClassifier(n_jobs=2)


	def load_tfidf(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()
		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'

		with open(directory + "trainset_tfidf.dat", "rb") as fin:
			self.trainset_X = (pickle.load(fin)).toarray()
		fin.close()

		with open(directory + "testset_tfidf.dat", "rb") as fin:
			self.testset_X = (pickle.load(fin)).toarray()
		fin.close()

		with open(directory + "trainset_Y.dat", "rb") as fin:
			self.trainset_Y = np.array(pickle.load(fin))
		fin.close()

		with open(directory + "testset_Y.dat", "rb") as fin:
			self.testset_Y = np.array(pickle.load(fin))
		fin.close()

		"""
		for i in range(len(self.trainset_Y)):
			if self.trainset_Y[i] > 0: self.trainset_Y[i]-=2

		for i in range(len(self.testset_Y)):
			if self.testset_Y[i] > 0: self.testset_Y[i]-=2
		"""
		print "Elapsed time: ", time() - t_start


	def train(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()
		self.clf.fit(self.trainset_X, self.trainset_Y)
		print "Elapsed time: ", time() - t_start

	def test(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()
		self.pred_Y = self.clf.predict_proba(self.testset_X)
		# fpr, tpr, thresholds = metrics.roc_curve(self.testset_Y, pred_Y)
		print "Elapsed time: ", time() - t_start

	def dump_pred(self):
		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'
		with open(directory + "pred_Y.dat", "wb") as fout:
			pickle.dump(self.pred_Y, fout)
		fout.close()





if __name__ == '__main__':
	if len(sys.argv) == 2:
		ngram = sys.argv[1]
	else:
		ngram = 1

	rf = RandomForest(ngram)
	rf.load_tfidf()
	rf.train()
	rf.test()
	rf.dump_pred()
	#rf.analysis()

