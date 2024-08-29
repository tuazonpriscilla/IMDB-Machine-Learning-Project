from google.colab import drive
import pandas as pd
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/CS167/IMDB_dataset.csv')
data.head()

import matplotlib.pyplot as plt
## Use cells here to explore the data:

#Samples
#There are 50,000 samples in this dataset
print(data.shape)

#Classes of target variable
#2, positive(1) and negative(0)

#Words per sample
data['words_per_sample']=data['review'].str.split().apply(len)
print(data['words_per_sample'].median())

#Distribution of sample length
print(data['words_per_sample'].hist())

#Something else: Shows how many words were used in all of the data
print(data['words_per_sample'].sum())

from bs4 import BeautifulSoup
import re
import nltk
#only do next line once
nltk.download() #in Corpora tab, download stopwords
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
#The NLTK downloader will open, you need to select (d) for Download, and then 'stopwords'then (q) to quit

#This is a function that takes in a review, makes sure it is only lower case letters and removes stopwords.
#It returns the cleaned review text.
def clean_review(review):
    #input is a string review
    #return is review cleaned of all punctuation, lowercase, and removed nltk stopwords
    letters_only = re.sub("[^a-zA-Z]"," ",review)
    lower_case = letters_only.lower()
    words = lower_case.split()
    for stop_word in stopwords.words("english"):
        while stop_word in words:
            words.remove(stop_word)
    cleaned = " ".join(words)
    return cleaned

#process the data
cleaned_text = []
for i in range(len(data)):
    cleaned_text.append(clean_review(data["review"][i])) 

#check cleaned data
cleaned_text[:5]

#establish training and testing dataset
train_data, test_data, train_sln, test_sln = \
    train_test_split(cleaned_text, data['sentiment'], test_size = 0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import PCA

#Bag of Words with 200 most common words
vectorizer = CountVectorizer(analyzer='word', max_features = 100)
#find the right 200 words
vectorizer.fit(train_data)

#use the vectorizer to transform review strings into word count vectors 
train_data_vectors = vectorizer.transform(train_data).toarray()
test_data_vectors = vectorizer.transform(test_data).toarray()

#PCA
extractor = PCA(n_components=50)
extractor.fit(train_data_vectors)
#print('this is the variance/importance of each component')
#print(extractor.explained_variance_ratio_)

train_transformed = extractor.transform(train_data_vectors)

#print("Here's what the training predictors look like after the transformation.")
#print(train_transformed[0:4])

test_transformed = extractor.transform(test_data_vectors)

## Now you can use train_data_vectors and test_data_vectors to train/test/tune your sklearn models.

from sklearn.svm import SVC
from sklearn import metrics

#baseline SVC
#load up scikit-learn SVC (Support Vector Classifier)
svc = SVC()
svc.fit(train_data_vectors,train_sln)
predictions = svc.predict(test_data_vectors)

#output accuracy
print("SVC accuracy:", metrics.accuracy_score(test_sln, predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))

#tuned SVC
#load up scikit-learn SVC (Support Vector Classifier)
svc = SVC(break_ties=True, cache_size=300)
svc.fit(train_data_vectors,train_sln)
predictions = svc.predict(test_data_vectors)

#output accuracy
print("SVC accuracy:", metrics.accuracy_score(test_sln, predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))

from sklearn.linear_model import Perceptron
#ignore warings -- there are lots of warnings regarding default values of Perceptron; which we accept
import warnings
warnings.filterwarnings("ignore")

#Baseline Perceptron
#load up scikit-learn Perceptron
perc = Perceptron()
perc.fit(train_data_vectors,train_sln)
perc_predictions = perc.predict(test_data_vectors)

#output accuracy
print("Perceptron accuracy:", metrics.accuracy_score(test_sln, perc_predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, perc_predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))

#Tuned Perceptron
#load up scikit-learn Perceptron
perc = Perceptron(alpha = .5, max_iter=1500, early_stopping=True)
perc.fit(train_data_vectors,train_sln)
perc_predictions = perc.predict(test_data_vectors)

#output accuracy
print("Perceptron accuracy:", metrics.accuracy_score(test_sln, perc_predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, perc_predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))

from sklearn.neural_network import MLPClassifier

#Baseline MLP
mlp = MLPClassifier()
mlp.fit(train_data_vectors,train_sln)
predictions = mlp.predict(test_data_vectors)

print("MLP Accuracy: ", metrics.accuracy_score(test_sln,predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))

#Tuned MLP
mlp = MLPClassifier(learning_rate = 'adaptive', hidden_layer_sizes=150)
mlp.fit(train_data_vectors,train_sln)
predictions = mlp.predict(test_data_vectors)

print("MLP Accuracy: ", metrics.accuracy_score(test_sln,predictions))

#Confusion matrix
labels = ["positive", "negative"]
conf_mat = metrics.confusion_matrix(test_sln, predictions, labels=labels)
print(pd.DataFrame(conf_mat,index = labels, columns = labels))
