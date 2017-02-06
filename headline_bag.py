'''
classifier.py - this file contains a radiology report classifier for real vs. 
fake PE.

Aly Valliani
February 4, 2017
'''

import sys, os.path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

def usage():
	'''
	usage - prints out information about which arguments should be included on 
	the command line. This is called when the user inputs improper arguments.

	Inputs: None
	Returns: None
	'''

	print >> sys.stderr, 'Usage: python classifier.py real_impressions.txt' \
		'fake_impressions.txt'

def checkArgs():
	'''
	checkArgs - checks to see if the command line args are consistent with the 
	program's usage. Returns the list of args if args are valid.

	Inputs: None
	Returns: argList - the list of command line args inputted by the user.
	'''

	argList = []
	argNotFound = False #bool to determine whether file paths for args exist

	if len(sys.argv) == 8: #checks the number of arguments (3)
		for pos in range(1, 8):
			currArg = sys.argv[pos]
			if not os.path.isfile(currArg): #print error if file doesn't exist
				print >> sys.stderr, currArg, 'does not exist.'
				argNotFound = True #if arg isn't found change bool, exit later
			else:
				argList.append(currArg) #list of valid args to return
		if argNotFound:
			exit(1)
	else: #return error if there are an incorrect number of arguments
		print >> sys.stderr, 'Incorrect number of arguments provided.'
		usage()
		exit(1)

	return argList

def readImpressions(impression_files):
	'''
	readImpressions - reads in each impression derived from a radiology 
	report.

	Inputs: impression text file
	Returns: dictionary of real and fake impressions
	'''

	impression_dict = {}
	class_lst = [2, 2, 2, 1, 1, 1, 1]

	for i in range(len(impression_files)):
		myfile = open(impression_files[i], 'r')
		count = 0
		num_lines = sum(1 for line in open(impression_files[i], 'r'))
		#num_lines = 516
		while count < num_lines:
			impression = myfile.readline().rstrip()
			if impression == '':
				break
			impression_dict[impression] = class_lst[i]
			count += 1
		print(count)

	return impression_dict

def lowerCase(impressions):
	ret = {}
	for i in range(len(impressions)):
		impression = impressions.keys()[i].split()
		imp_new = [word.lower() for word in impression]
		ret[' '.join(imp_new)] = impressions.values()[i]

	return ret

def removeStopWords(impressions):
	ret = {}
	stop_words = []
	for line in open('stopWordsDict.txt', 'r'):
		stop_words.append(line.split()[0])

	for i in range(len(impressions)):
		impression = impressions.keys()[i].split()
		imp_new = [word for word in impression if word not in stop_words]
		ret[' '.join(imp_new)] = impressions.values()[i]

	return ret

def negate(impressions):
	#punctuation = ['.', ',', ';', '?', '!', ':', '"', '\'']
	neg_lst = ['no', 'none', 'not', 'negative']
	neg_impressions = {}
	for i in range(len(impressions)):
		impression = impressions.keys()[i].split()
		flag = False
		for j in range(len(impression)):
			if flag:
				impression[j] = 'NOT_' + impression[j]
				#flag = False
			if impression[j].lower() in neg_lst:
				flag = True
			#if any(p in impression[j] for p in punctuation):
			#	flag = False
		neg_impressions[' '.join(impression)] = impressions.values()[i]
		#print(neg_impressions)

	return neg_impressions

def buildTrainTest(impressions, train_split=0.8):
	global vectorizer
	vectorizer = CountVectorizer(ngram_range=(1,5))
	#vectorizer = TfidfVectorizer(ngram_range=(1, 5), sublinear_tf=True, 
	#	max_df=0.90, stop_words='english')
	size = len(impressions)
	train_size = int(size*train_split)
	x_train = vectorizer.fit_transform(impressions.keys()[:train_size])
	x_test = vectorizer.transform(impressions.keys()[train_size:])
	y_train = impressions.values()[:train_size]
	y_test = impressions.values()[train_size:]

	return x_train, x_test, y_train, y_test

def classify(x_train, y_train):
	print(clf)
	print('Training...')
	clf.fit(x_train, y_train)
	print('done.')
	print('Train Accuracy: %.2f' % clf.score(x_train, y_train))

def evaluate(x_test, y_test):
	print('Testing...')
	pred = clf.predict(x_test)
	print('done.')
	#print('Test Accuracy: %.2f' % clf.score(x_test, y_test))
	score = metrics.accuracy_score(y_test, pred)
	print('Accuracy: %.2f' % score)
	fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=2)
	auc = metrics.auc(fpr, tpr)
	print('AUC: %.2f' % auc)
	f_score = metrics.f1_score(y_test, pred, pos_label=2)
	print('F-score: %.2f' % f_score)
	print('Confusion Matrix:')
	print(metrics.confusion_matrix(y_test, pred))
	importances = clf.feature_importances_
	indices = np.argsort(importances)[::-1]
	# Print the feature ranking
	print('Feature ranking:')
	feature_names = np.asarray(vectorizer.get_feature_names())
	for f in range(10):
		print('%d. feature: %s' % (f + 1, feature_names[indices[f]]))

	#feature_names = vectorizer.get_feature_names()
	#feature_names = np.asarray(feature_names)
	#top10 = np.argsort(clf.coef_[0])[-10:]
	#print(' '.join(feature_names[top10]))
	#print(len(clf.coef_[0]))

def main():
	global clf
	#clf = MultinomialNB(alpha=0.01) #second best
	#clf = RandomForestClassifier(n_estimators=100)
	#clf = linear_model.Lasso(alpha=0.001)
	#clf = GradientBoostingClassifier(n_estimators=100)
	#clf = svm.SVC()
	#clf = ExtraTreesClassifier(n_estimators=100)
	clf = AdaBoostClassifier(n_estimators=100) #best so far
	args = checkArgs()
	impressions = readImpressions(args)
	#impressions = lowerCase(impressions) #makes it much worse
	#impressions = removeStopWords(impressions) #makes it somewhat worse
	impressions = negate(impressions)
	x_train, x_test, y_train, y_test = buildTrainTest(impressions)
	classify(x_train, y_train)
	evaluate(x_test, y_test)

if __name__ == '__main__':
	main()
