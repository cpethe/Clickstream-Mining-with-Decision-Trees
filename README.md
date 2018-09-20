# Clickstream-Mining-with-Decision-Trees

The project is based on a task posed in KDD Cup 2000. It involves mining click-stream data collected from
Gazelle.com, which sells legware products. The task is to determine: Given a set of page views, will the visitor
view another page on the site or will he leave?

There are 5 files in .csv format:
1. trainfeat.csv: Contains 40000 examples, each with 274 features in the form of a 40000 x 274 matrix.
2. trainlabs.csv: Contains the labels (class) for each training example (did the visitor view another page?)
3. testfeat.csv: Contains 25000 examples, each with 274 features in the form of a 25000 x 274 matrix.
4. testlabs.csv: Contains the labels (class) for each testing example.
5. featnames.csv: Contains the "names" of the features.

This is an implementation of the ID3 decision tree learner in Python, which uses the chi-squared split stopping
criterion with the p-value threshold given as a parameter.
