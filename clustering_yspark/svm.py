#!/usr/bin/python
import pickle

from common_def import *

class SVM:
  def __init__(self):
    self.load_tfidf()
    self.clf = SVC(probability=True, kernel='linear')

  def load_tfidf(self):
    directory = TFIDF_DIR

    with open(directory + "trainset_tfidf.dat", "rb") as fin:
      self.trainset_X = pickle.load(fin)
    fin.close()

    with open(directory + "testset_tfidf.dat", "rb") as fin:
      self.testset_tfidf = pickle.load(fin)
    fin.close()

    with open(directory + "trainset_Y.dat", "rb") as fin:
      self.trainset_Y = pickle.load(fin)
    fin.close()

    with open(directory + "testset_Y.dat", "rb") as fin:
      self.testset_Y = pickle.load(fin)
    fin.close()

  def train(self):






if __name__ == '__main__':
  svm = SVM()
  svm.train()

  trainset_X, testset_X, trainset_Y, testset_Y = load_tfidf()
  svm(trainset_X, testset_X, trainset_Y, testset_Y)
