'''
classifier.py - this file will classify headlines as 
{agree, disagree, unrelated, discusses}

Aly Valliani, Shreyas Lakhtakia
February 5, 2017
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

nClass = 4

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

	if len(sys.argv) == nClass + 1: #checks the number of arguments (nClass + 1)
		for pos in range(1, nClass + 1):
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

def readHeadlines(headline_files):
	'''
	readHeadlines - reads in each impression derived from a radiology 
	report.

	Inputs: impression text file
	Returns: dictionary of real and fake Headlines
	'''

	headline_dict = []
	class_lst = range(1, nClass + 1) #1, 2, 3, 4
	print(class_lst)

	for i in range(len(headline_files)):
		myfile = open(headline_files[i], 'r')
		count = 0
		num_lines = sum(1 for line in open(headline_files[i], 'r'))
		#num_lines = 516
		while count < num_lines:
			headline = myfile.readline().rstrip()
			if headline == '':
				break
			print(i)
			headline_dict.append((headline, i))
			count += 1
		print(count)

	exit(1)
	return headline_dict



def buildTrainTest(headlines, train_split=0.8):
	global vectorizer
	vectorizer = CountVectorizer(ngram_range=(1,3))
	#vectorizer = TfidfVectorizer(ngram_range=(1, 5), sublinear_tf=True, 
	#	max_df=0.90, stop_words='english')
	size = len(headlines)
	train_size = int(size*train_split)
	x_train = vectorizer.fit_transform(headlines.keys()[:train_size])
	x_test = vectorizer.transform(headlines.keys()[train_size:])
	y_train = headlines.values()[:train_size]
	y_test = headlines.values()[train_size:]
	print("y_train:\n")
	print(y_train)
	print("y_test:\n")
	print(y_test)
	exit(1)

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
	headlines = readHeadlines(args)
	# headlines = lowerCase(headlines) #makes it much worse
	# headlines = removeStopWords(headlines) #makes it somewhat worse
	# headlines = negate(headlines)
	x_train, x_test, y_train, y_test = buildTrainTest(headlines)
	classify(x_train, y_train)
	evaluate(x_test, y_test)

if __name__ == '__main__':
	main()


############################## Appendix ##############################

# def lowerCase(impressions):
# 	ret = {}
# 	for i in range(len(impressions)):
# 		impression = impressions.keys()[i].split()
# 		imp_new = [word.lower() for word in impression]
# 		ret[' '.join(imp_new)] = impressions.values()[i]

# 	return ret

# def removeStopWords(impressions):
# 	ret = {}
# 	stop_words = []
# 	for line in open('stopWordsDict.txt', 'r'):
# 		stop_words.append(line.split()[0])

# 	for i in range(len(impressions)):
# 		impression = impressions.keys()[i].split()
# 		imp_new = [word for word in impression if word not in stop_words]
# 		ret[' '.join(imp_new)] = impressions.values()[i]

# 	return ret

# def negate(impressions):
# 	#punctuation = ['.', ',', ';', '?', '!', ':', '"', '\'']
# 	neg_lst = ['no', 'none', 'not', 'negative']
# 	neg_impressions = {}
# 	for i in range(len(impressions)):
# 		impression = impressions.keys()[i].split()
# 		flag = False
# 		for j in range(len(impression)):
# 			if flag:
# 				impression[j] = 'NOT_' + impression[j]
# 				#flag = False
# 			if impression[j].lower() in neg_lst:
# 				flag = True
# 			#if any(p in impression[j] for p in punctuation):
# 			#	flag = False
# 		neg_impressions[' '.join(impression)] = impressions.values()[i]
# 		#print(neg_impressions)

# 	return neg_impressions