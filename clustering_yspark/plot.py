#!/usr/bin/python

import sys, pickle
import matplotlib.pyplot as plt
import numpy as np

from common_def import *
from collections import Counter
from sklearn.metrics import *

class Plot:
	def __init__(self, ngram=1):
		self.ngram = ngram

	def load_dump(self):
		directory = TFIDF_DIR + '_' + str(self.ngram) + '/'

		with open(directory + "pred_Y.dat", "rb") as fout:
			self.pred_Y = pickle.load(fout)
		fout.close()

		with open(directory + "testset_Y.dat", "rb") as fout:
			self.testset_Y = pickle.load(fout)
		fout.close()

	def precision_recall(self):
		print '======================================='
		print sys._getframe().f_code.co_name
		print '======================================='
		thresholds = np.arange(0.5, 1.01, 0.1)
		testset_counter = Counter(self.testset_Y)

		print "Overall # prediction:", len(self.testset_Y)
		print "# each cluster:", testset_counter, len(testset_counter)

		precision_mat = np.zeros((len(testset_counter), len(thresholds)))
		recall_mat = np.zeros((len(testset_counter), len(thresholds)))
		f1score_mat = np.zeros((len(testset_counter), len(thresholds)))


		for index in range(len(thresholds)):
			threshold = thresholds[index]
			print '==============================='
			print 'Threshold:', threshold
			print '==============================='
			new_pred_Y = []

			for pred in self.pred_Y:
				if np.max(pred) >= threshold:
					new_pred_Y.append(np.argmax(pred))
				else:
					new_pred_Y.append(-1)

			print Counter(new_pred_Y), len(new_pred_Y)
			print 'Accruacy:', accuracy_score(self.testset_Y, new_pred_Y)

			target_names = ['Unclassified', 'AdvanceFeeFraud', 'Lottery', 'Employment', 'Romance', 'StrandedTraveler',
											'Phishing', 'Enron']
			print(classification_report(self.testset_Y, new_pred_Y, target_names=target_names))

			precision, recall, fscore, support = precision_recall_fscore_support(self.testset_Y, new_pred_Y)

			for label in range(len(testset_counter)):
				precision_mat[label, index] = precision[label + 1]
				recall_mat[label, index] = recall[label + 1]
				f1score_mat[label, index] = fscore[label + 1]
			# endfor
		# endfor


		print precision_mat
		print recall_mat




		labels = ['Advance Fee Fraud', 'Lottery', 'Employment', 'Romance', 'StrandedTraveler', 'Phishing/ID theft', 'Enron']
		colors = ['r', 'b', 'c', 'm', 'g', 'k', 'y']
		markers = ['o', 'x', '^', 'p', '>', '*', '+']


		##########################
		# precision
		##########################
		plt.figure()
		for i in range(len(testset_counter)):
			plt.plot(thresholds, precision_mat[i], label=labels[i], marker=markers[i])

		plt.ylabel("Precision")
		plt.xlabel("Prediction prob. threshold")

		plt.ylim(0.90, 1.01)
		# plt.xlim(-0.5, 3.5)
		#plt.xticks(np.arange(0,4,1), [32, 64, 128, 256])
		plt.legend(loc='lower right', ncol=2)
		#plt.show()
		plt.savefig('./figure/precision_' + str(self.ngram) + '_' + str(t_start) + '.pdf')

		##########################
		# recall
		##########################
		plt.figure()
		for i in range(len(testset_counter)):
			plt.plot(thresholds, recall_mat[i], label=labels[i], marker=markers[i])

		plt.ylabel("Recall")
		plt.xlabel("Prediction prob. threshold")

		#plt.ylim(0.90, 1.01)
		#plt.xlim(-0.5, 3.5)
		#plt.xticks(np.arange(0,4,1), [32, 64, 128, 256])
		plt.legend(loc='lower left', ncol=2)
		#plt.show()
		plt.savefig('./figure/recall_' + str(self.ngram) + '_' + str(t_start) + '.pdf')

	# enddef



if __name__ == '__main__':
	if len(sys.argv) == 2:
		ngram = sys.argv[1]
	else:
		ngram = 1

	p = Plot(ngram=ngram)
	p.load_dump()
	p.precision_recall()

"""
precision = [[0.9602, 1.0, 1.0, 1.0, 1.0, 1.0],
						 [0.978, 0.987, 0.985, 0.982, 0.98, 1],
						 [0.959, 0.979, 1.0, 1.0, 1.0, 1.0],
						 [0.921, 0.946, 0.958, 0.9805, 0.985, 1],
						 [1, 1, 1, 1, 1, 1],
						 [0.949, 0.954, 1, 1, 1, 1],
						 [0.899, 0.931, 0.955, 0.963, 0.987, 0.98]]

recall = [[0.718, 0.605, 0.465, 0.38, 0.25, 0.15],
					[0.92, 0.89, 0.84, 0.78, 0.66, 0.44],
					[0.67, 0.57, 0.53, 0.5, 0.45, 0.35],
					[0.88, 0.82, 0.75, 0.65, 0.47, 0.20],
					[0.40, 0.30, 0.30, 0.20, 0.10, 0.10],
					[0.91, 0.88, 0.84, 0.76, 0.62, 0.47],
					[0.89, 0.84, 0.76, 0.59, 0.44, 0.25]]

labels = ['Advance Fee Fraud', 'Lottery', 'Employment', 'Romance', 'StrandedTraveler', 'Phishing/ID theft', 'Enron']
colors = ['r', 'b', 'c', 'm', 'g', 'k', 'y']
markers = ['o', 'x', '^', 'p', '>', '*', '+']

plt.figure()
for i in range(len(precision)):
	plt.plot(recall[i], precision[i], label=labels[i], marker=markers[i])

plt.ylabel("Precision")
plt.xlabel("Recall")

# plt.ylim(0.90, 1.01)
#plt.xlim(-0.5, 3.5)
#plt.xticks(np.arange(0,4,1), [32, 64, 128, 256])
plt.legend(loc='lower left', ncol=2)
#plt.show()
plt.savefig('./figure/precision_recall.pdf')
"""