#!/usr/bin/python

import os
import random
import sys
import pickle
from time import time

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import scipy.sparse as sp
#from scipy.sparse import csr_matrix
#from scipy.sparse import lil_matrix


from common_def import *
from ngram_extract import Ngram_Extractor


class TFIDF:
	def __init__(self, ngram):
		self.ngram = ngram
		self.trainset_len = 0
		self.testset_len = 0
		self.feature_len = 0

	# input: none (actual email data files)
	# output: category, token_list
	def generate_traintest(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		ngram_extractor = Ngram_Extractor()

		trainset_Y = []
		trainset_X = []
		testset_Y = []
		testset_X = []

		count = 0

		for directory in CATEGORY_DIRECTORY_LIST:
			print("DIRECTORY:", directory)
			fcount = 0

			for (dirpath, dirnames, filenames) in os.walk(directory):
				print '\t' + dirpath
				filenames = [f for f in filenames if '.txt' in f]
				if len(filenames) == 0: continue

				"""
				if float(MAX_EMAIL_PER_DIR) > float(len(filenames)) * 0.1:
					idx_list = range(0, len(filenames))
					random.shuffle(idx_list)
					dataset_size = min(MAX_EMAIL_PER_DIR, len(idx_list))
					if 'enron' in dirpath: dataset_size = min(2*MAX_EMAIL_PER_DIR, len(idx_list))
					if dataset_size < 0: dataset_size = len(filenames)

				else:
					if 'enron' in dirpath:
						dataset_size = min(2*MAX_EMAIL_PER_DIR, len(filenames))
					else:
						dataset_size = min(MAX_EMAIL_PER_DIR, len(filenames))
					if dataset_size < 0: dataset_size = len(filenames)

					idx_list = set([])
					while(len(idx_list) < dataset_size):
						idx_list.add(np.random.randint(0, len(filenames)))
					idx_list = list(idx_list)
				#endif

				trainset_idx = idx_list[:int(dataset_size*9.0/10.0)]
				testset_idx = idx_list[int(dataset_size*9.0/10.0):dataset_size]
				"""

				idx_list = range(0, len(filenames))
				random.shuffle(idx_list)
				dataset_size = min(MAX_EMAIL_PER_DIR, len(idx_list))
				if 'enron' in dirpath: dataset_size = min(2 * MAX_EMAIL_PER_DIR, len(idx_list))
				if dataset_size < 0: dataset_size = len(filenames)
				trainset_idx = idx_list[:int(dataset_size * 9.0 / 10.0)]
				testset_idx = idx_list[int(dataset_size * 9.0 / 10.0):dataset_size]
				print 'trainset:%d, testset:%d' % (len(trainset_idx), len(testset_idx))

				for idx in trainset_idx:
					category, body = ngram_extractor.extract_body(dirpath + '/' + filenames[idx])
					label = get_label(category)
					if label >= 0:
						tokens = ngram_extractor.extract_ngarm(body, self.ngram)
						trainset_X.append(tokens)
						trainset_Y.append(label)

						count += 1
						if count % 5000 == 0: print "Processing %d..." % count

				for idx in testset_idx:
					category, body = ngram_extractor.extract_body(dirpath + '/' + filenames[idx])
					label = get_label(category)
					if label >= 0:
						tokens = ngram_extractor.extract_ngarm(body, self.ngram)
						testset_X.append(tokens)
						testset_Y.append(label)

						count += 1
						if count % 5000 == 0: print "Processing %d..." % count


		self.trainset_len = len(trainset_X)
		self.testset_len = len(testset_X)

		self.trainset_X = trainset_X
		self.trainset_Y = trainset_Y
		self.testset_X = testset_X
		self.testset_Y = testset_Y
		print "Elapsed time: ", time() - t_start
		#return trainset_X, trainset_Y, testset_X, testset_Y


		# input: trainset token_list
		# output:

	# feature_dic: term-ID mapping table)
	#   DF_dic:  Document frequency of terms
	def generate_feature_dic(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		term_index_dic = {}
		index_term_dic = {}
		DF_dic = {}
		IDF_dic = {}
		trainset_X = self.trainset_X

		for tokens in trainset_X:
			for token in set(tokens):
				# build term-index dictionary
				if token not in term_index_dic:
					index = len(index_term_dic)
					index_term_dic[index] = token
					term_index_dic[token] = index

				# build DF dictionary
				token_index = term_index_dic[token]
				if token_index not in DF_dic:
					DF_dic[token_index] = 1
				else:
					DF_dic[token_index] += 1

		for index in DF_dic.keys():
			IDF_dic[index] = np.log(float(len(trainset_X)) / float((1 + DF_dic[index])))

		self.feature_len = len(IDF_dic)
		self.term_index_dic = term_index_dic
		self.index_term_dic = index_term_dic
		self.IDF_dic = IDF_dic

		print "Elapsed time: ", time() - t_start
		#return term_index_dic, index_term_dic, IDF_dic


	# input:
	#   feature dictionary (term - ID mapping table)
	#   trainset_X, testset_X: token list
	# output
	#   trainvector_X, testvector_X: term count vector
	def generate_feature_vector(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		term_id_dic = self.term_dic_dic
		IDF_dic = self.ID_dic
		trainset_X = self.trainset_X
		testset_X = self.testset_X

		trainvector_X = sp.lil_matrix(np.zeros((len(trainset_X), len(term_id_dic))))
		testvector_X = sp.lil_matrix(np.zeros((len(testset_X), len(term_id_dic))))
		print "Train vector: ", trainvector_X.shape, "Test vector: ", testvector_X.shape

		print 'Generating train feature vector...'
		row_index = 0
		for tokens in trainset_X:
			for token in tokens:
				term_index = term_id_dic[token]
				trainvector_X[row_index, term_index] += IDF_dic[term_index]
			#normalize trainvector
			trainvector_X[row_index] = normalize(trainvector_X[row_index], norm='l2')
			row_index += 1
			if row_index % 1000 == 0: print '%d/%d completed' % (row_index, len(trainset_X))
		#endfor

		print 'Generating test feature vector...'
		row_index = 0
		for tokens in testset_X:
			for token in tokens:
				if token in term_id_dic:
					term_index = term_id_dic[token]
					testvector_X[row_index, term_index] += IDF_dic[term_index]
			#endfor
			testvector_X[row_index] = normalize(testvector_X[row_index], norm='l2')
			row_index += 1
			if row_index % 1000 == 0: print '%d/%d completed' % (row_index, len(testvector_X))
		#endfor

		self.trainvector_X = trainvector_X
		self.testvector_X = testvector_X

		print "Elapsed time: ", time() - t_start
		#return trainvector_X, testvector_X


	# input:
	# trainset_X, testset_X: term count vector
	# output:
	#   trainset_tfidf, testset_tfidf: TF-IDF weighted vector
	def generate_tfidf(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		trainvector_X = self.trainvector_X
		testvector_X = self.testvector_X
		IDF_dic = self.IDF_dic
		trainset_tfidf = sp.csr_matrix(trainvector_X)
		testset_tfidf = sp.csr_matrix(testvector_X)
		self.trainset_len = len(self.trainset_Y)
		self.testset_len = len(self.testset_Y)

		print "Train TF-IDF matrix: ", trainset_tfidf.shape, "Test TF-IDF matrix: ", testset_tfidf.shape

		# trainset_tfidf
		print 'Trainset TFIDF...'
		(row_index_list, col_index_list, tf_list) = sp.find(trainset_tfidf)
		count = 0
		for (row_index, col_index, tf) in zip(row_index_list, col_index_list, tf_list):
			idf = IDF_dic[col_index]
			trainset_tfidf[row_index, col_index] = tf * idf
			count += 1
			if count % 5000 == 0: print '\t%d/%d completed' % (count, len(row_index_list))
		#normalize
		'''
		print 'Trainset Normalize...'
		for row_index in range(self.trainset_len):
			trainset_tfidf[row_index] = normalize(trainset_tfidf[row_index])
			if row_index % 1000 == 0: print '\t%d/%d completed' % (row_index, self.trainset_len)
		#endfor
		'''

		# testset_tfidf
		print 'Testset TFIDF...'
		(row_index_list, col_index_list, tf_list) = sp.find(testset_tfidf)
		count = 0
		for (row_index, col_index, tf) in zip(row_index_list, col_index_list, tf_list):
			idf = IDF_dic[col_index]
			testset_tfidf[row_index, col_index] = tf * idf
			if count % 5000 == 0: print '\t%d/%d completed' % (count, len(row_index_list))
		'''
		#normalize
		print 'Testset Normalize...'
		for row_index in range(self.testset_len):
			testset_tfidf[row_index] = normalize(testset_tfidf[row_index])
			if row_index % 1000 == 0: print '\t%d/%d completed' % (row_index, self.testset_len)
		#endfor
		'''

		#transformer = TfidfTransformer(norm = 'l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
		#trainset_tfidf = transformer.fit_transform(trainvector_X)
		#testset_tfidf = transformer.transform(testvector_X)
		self.trainset_tfidf = trainset_tfidf
		self.testset_tfidf = testset_tfidf
		print "Elapsed time: ", time() - t_start
		#return trainset_tfidf, testset_tfidf

	def dump_feature(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'
		if not os.path.exists(directory):
			os.mkdir(directory)

		# dump term-index dictionary
		with open(directory + "term_id_dic.dat", "wb") as fout:
			pickle.dump(self.term_id_dic, fout)
		fout.close()

		with open(directory + "idf_dic.dat", "wb") as fout:
			pickle.dump(self.IDF_dic, fout)
		fout.close()

		with open(directory + "trainset_feature.dat", "wb") as fout:
			pickle.dump(self.trainvector_X, fout)
		fout.close()

		with open(directory + "testset_feature.dat", "wb") as fout:
			pickle.dump(self.testvector_X, fout)
		fout.close()

		with open(directory + "trainset_Y.dat", "wb") as fout:
			pickle.dump(self.trainset_Y, fout)
		fout.close()

		with open(directory + "testset_Y.dat", "wb") as fout:
			pickle.dump(self.testset_Y, fout)
		fout.close()

		print "Elapsed time: ", time() - t_start

	def load_feature(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'

		with open(directory + "term_id_dic.dat", "rb") as fout:
			self.term_id_dic = pickle.load(fout)
		fout.close()

		with open(directory + "idf_dic.dat", "rb") as fout:
			self.IDF_dic = pickle.load(fout)
		fout.close()

		with open(directory + "trainset_feature.dat", "rb") as fout:
			self.trainvector_X= pickle.load(fout)
		fout.close()

		with open(directory + "testset_feature.dat", "rb") as fout:
			self.testvector_X = pickle.load(fout)
		fout.close()

		with open(directory + "trainset_Y.dat", "rb") as fout:
			self.trainset_Y = pickle.load(fout)
		fout.close()

		with open(directory + "testset_Y.dat", "rb") as fout:
			self.testset_Y = pickle.load(fout)
		fout.close()

		print "Elapsed time: ", time() - t_start
		#return term_id_dic, IDF_dic, trainvector_X, testvector_X, trainset_Y, testset_Y

	# dump TF-IDF vectors of train/testset, LABEL of train/testset, feature_dic
	def dump_tfidf(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		t_start = time()

		"""
		# for future usage: SAVE TDIDF TRANSFORMER
		with open("tfidf_transformer", "wb") as outfile:
			pickle.dump(transformer, outfile)
		outfile.close()

		with open("tfidf_transformer", "rb") as outfile:
			transformer2 = pickle.load(outfile)
		outfile.close()
		"""

		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'
		if not os.path.exists(directory):
			os.mkdir(directory)

		with open(directory + "trainset_tfidf.dat", "wb") as fout:
			pickle.dump(self.trainset_tfidf, fout)
		fout.close()

		with open(directory + "testset_tfidf.dat", "wb") as fout:
			pickle.dump(self.testset_tfidf, fout)
		fout.close()

		print "Elapsed time: ", time() - t_start


if __name__ == '__main__':
	PHASE = 1

	if len(sys.argv) == 2:
		ngram = int(sys.argv[1])
	else:
		ngram = 1
	print 'ngram:', ngram

	tfidf = TFIDF(ngram)

	if PHASE == 0:
		tfidf.generate_traintest()
		tfidf.generate_feature_dic()
		tfidf.generate_feature_vector()
		tfidf.dump_feature()
	elif PHASE == 1:
		tfidf.load_feature()
		tfidf.generate_tfidf()
		tfidf.dump_tfidf()




















