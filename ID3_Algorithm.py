import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import datetime

sys.setrecursionlimit(4000)



'''
TreeNode represents a node in your decision tree
TreeNode can be:
	- A non-leaf node: 
		- data: contains the feature number this node is using to split the data
		- children[0]-children[4]: Each correspond to one of the values that the feature can take
		
	- A leaf node:
		- data: 'T' or 'F' 
		- children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
	def __init__(self, data='T',children=[-1]*5):
		self.nodes = list(children)
		self.data = data


	def save_tree(self,filename):
		obj = open(filename,'w')
		pkl.dump(self,obj)


# implementation of id3 algorithm
def id3(examples, target_attr, attr, threshold):
	
	# base case where all examples are positive
	if 0 not in examples[target_attr].unique():
		root = TreeNode(data='T', children=[])
		return root
	# base case where all examples are negative
	if 1 not in examples[target_attr].unique():
		root = TreeNode(data='F', children=[])
		return root

	# no more attributes left to split on
	if len(attr) == 0:
		value_counts = examples[target_attr].value_counts()
		
		x = value_counts[1]
		y = value_counts[0]

		# more positive examples than negative examples
		if x > y:
			root = TreeNode(data='T', children=[])
			return root
		# more negative examples than positive examples
		else:
			root = TreeNode(data='F', children=[])
			return root

	# selecting attribute with least entropy to split on
	best_attr = select_best_attr(examples, target_attr, attr)

	# using chi-squared criterion to decide whether to stop
	stop_split = chi2_splitting(examples, target_attr, best_attr, threshold)

	# if p-value is greater than threshold
	if stop_split == True:
		value_counts = examples[target_attr].value_counts()

		x = value_counts[1]
		y = value_counts[0]

		# more positive examples than negative examples
		if x > y:
			root = TreeNode(data='T', children=[])
			return root
		# more negative examples than positive examples
		else:
			root = TreeNode(data='F', children=[])
			return root

	# setting the best attribute as the attribute to split on
	root = TreeNode(data=str(best_attr + 1))

	new_attr = list(attr)
	# removing the best attribute from the list of attributes to pass recursively
	new_attr.remove(best_attr)
	
	# calling id3 recursively for each child of current node
	for elem in range(5):
		new_examples = examples[examples[best_attr] == elem + 1]
		new_examples = new_examples.drop(new_examples.columns[list(new_examples.columns).index(best_attr)], axis=1)
		root.nodes[elem] = id3(new_examples, target_attr, new_attr, threshold)

	return root

# this function returns the attribute with maximum gain (least entropy)
def select_best_attr(examples, target_attr, attr):
	attr_entropies = {}
	final_count = len(examples)

	# for every attribute, calculate entropy
	for attribute in attr:
		counts = []
		entropies = []
		for elem in examples[attribute].unique():
			temp = examples[examples[attribute] == elem]
			total_count = len(temp)

			value_counts = temp[target_attr].value_counts()
			
			if len(value_counts.index.values) != 2:
				counts.append(total_count)
				entropies.append(0)
				continue

			x = value_counts[1] / float(total_count)
			y = value_counts[0] / float(total_count)
				
			entropy = ((x * np.log2(x)) + (y * np.log2(y))) * (-1)

			counts.append(total_count)
			entropies.append(entropy)

		final_entropy = 0
		# summing entropies for all  values in the current attribute
		for i in range(len(entropies)):
			final_entropy = final_entropy + (counts[i] * entropies[i])

		final_entropy = final_entropy / final_count
		attr_entropies[attribute] = final_entropy

	# returning attribute with minimum entropy
	best = min(attr_entropies.items(), key=lambda x: x[1])

	return best[0]

#  this function calculates p-value for a particular attribute
#  and decides whether to stop or not
def chi2_splitting(examples, target_attr, best_attr, threshold):
	value_counts = examples[target_attr].value_counts()

	p = 0.0
	n = 0.0

	if 1 in value_counts.index.values:
		p = float(value_counts[1])
	if 0 in value_counts.index.values:
		n = float(value_counts[0])

	N = p + n

	S = 0.0

	for elem in examples[best_attr].unique():
		temp = examples[examples[best_attr] == elem]

		value_counts = temp[target_attr].value_counts()
		p_i = 0.0
		n_i = 0.0

		if 1 in value_counts.index.values:
			p_i = float(value_counts[1])
		if 0 in value_counts.index.values:
			n_i = float(value_counts[0])

		T_i = len(temp)

		p_i_prime = p * T_i / float(N)
		n_i_prime = n * T_i / float(N)

		part_1 = ((p_i_prime - p_i) ** 2) / float(p_i_prime)
		part_2 = ((n_i_prime - n_i) ** 2) / float(n_i_prime)
		

		S = S + part_1 + part_2

	p_value = 1 - stats.chi2.cdf(x=S, df=len(examples[best_attr].unique()) - 1)
	if p_value > threshold:
		return True
	else:
		return False

def evaluate_datapoint(root, datapoint):
	if root.data == 'T':
		return 1
	if root.data == 'F':
		return 0
	return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1] - 1], datapoint)


# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain = pd.read_csv(ftrain, header=None, delim_whitespace=True)

	Xtest = []
	with open(ftest, 'rb') as f:
		reader=csv.reader(f)
		for row in reader:
			rw = map(int, row[0].split())
			Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	Ytrain = pd.read_csv(ftrain_label, header=None, delim_whitespace=True)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest




parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")

training_set = pd.concat([Xtrain, Ytrain], axis=1, ignore_index=True)
target_attr = len(training_set.columns) - 1

attr = list()

for elem in training_set.columns:
	attr.append(elem)

attr.remove(target_attr)

startTime = datetime.datetime.now().time()
s = id3(training_set, target_attr, attr, float(pval))
endTime = datetime.datetime.now().time()

s.save_tree(tree_name)


print("Testing...")
Ypredict = []
for i in range(0, len(Xtest)):
	result = evaluate_datapoint(s, Xtest[i])
	Ypredict.append([result])

with open(Ytest_predict_name, "wb") as f:
	writer = csv.writer(f)
	writer.writerows(Ypredict)

print("Output files generated")

