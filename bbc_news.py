#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# 
# First, we import the libraries that we need using "import".
# 
# Note: All these libraries need to be downloaded beforehand if not using Google Colab. 


import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#download the data if needed
#nltk.download('stopwords')

# # Prepare our dataset
# 
# we need to load the dataset in Python. 
# I load it locally from the hard drive, you need to copy the bbc folder and paste it in category_path below.


categories = ["business","entertainment","politics","sport","tech"]
news_list = []
news_category = []

for folder in categories:
    category_path = 'C:/Users/mmat/OneDrive/Desktop/Application of ML/coursework/datasets_coursework1/bbc/'+ folder +'/'
    files = os.listdir(category_path)
    for text in files:
        text_path = category_path + "/" + text
        with open(text_path, errors = 'replace') as t:
            data = t.readlines()
        data = ' '.join(data)
        news_list.append(data)
        news_category.append(folder)


# here you need the same link as above but replace 'bbc/' to 'bbc.csv', then paste it in df.to_csv and data.


news_csv = {'article':news_list, 'category':news_category}
df = pd.DataFrame(news_csv)
df.to_csv('C:/Users/mmat/OneDrive/Desktop/Application of ML/coursework/datasets_coursework1/bbc.csv')
data = pd.read_csv('C:/Users/mmat/OneDrive/Desktop/Application of ML/coursework/datasets_coursework1/bbc.csv')
data= data[data['article'].notnull()]
data['category_id'] = data['category'].factorize()[0]
columns_list = ['index','article', 'category', 'category_id']
data.columns = columns_list

data


# # Preprocessing our dataset
#  
# Preprocess the texts with nltk in few steps.
# this is original text before preprocessing. 

# Remove the # if you would like to see the changes to the text.



#print(data['article'][0])


# All not English words will remove from the dataset, including spaces and full stops.


stopwords = nltk.corpus.stopwords.words('english')
data['BoW'] = data['article'].apply(lambda x: ' '.join([word for word in x.split() 
                                                                          if word not in (stopwords)]))
#print(data['BoW'][0])


# Removing morphological affixes from words, leaving only the stem word.


ps = PorterStemmer()
data['BoW'] = data['BoW'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
#print(data['BoW'][0])


# Changing the uppercase letters to lowercase.


#tranform every capital letter to small letter
data['BoW'] = data['BoW'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
#print(data['BoW'][0])


# Removing all the non-word characters from the data. 


data['BoW'] = data['BoW'].str.replace('[^\w\s]','')
#print(data['BoW'][0])


# Split the article into words and counted each word’s frequency then make a list of the fewer frequency words.


frequency = pd.Series(' '.join(data['BoW']).split()).value_counts()

less_frequency = list(frequency[frequency <= 3].index.values)
#less_frequency


# this step takes a bit long running time. it is to remove the less frequency words from our Bag of Words


data['BoW'] = data['BoW'].apply(lambda x: ' '.join([word for word in x.split() if word not in (less_frequency)]))


# finally, here our preprocessed dataset ready for feature engineering


data = data[['index', 'category', 'category_id', 'BoW']]
#print(data['BoW'][0])


# # Features Engineering
# 
# I used tfidf vectorizer for feature extraction

# transform our Bag of Words to feature array to use it in feature selection



tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tf_idf.fit_transform(data.BoW).toarray()
labels = data.category_id


category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)


# # Feature Selection
# 
# I used chi-squared test to train the data with each feature


n = 5
for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tf_idf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("- '{}':".format(category))
    print("  . Top Unigram words:\n       . {}".format('\n       . '.join(unigrams[-n:])))
    print("  . Top Bigram words:\n       . {}".format('\n       . '.join(bigrams[-n:])))


# # Model Selection
# 
# here i tried four diffrent models to choose the one with high accuracy


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(random_state=0) ]

CV = 5
cross_val_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cross_val_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
cross_val_df


# # Training and Testing
# as Logistic regression has the higher accuracy, it has been the selected model 

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    features,labels, data.index, test_size=0.33, random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))


# # Examinig the model
# 
# here you can write or copy-paste any articles or titles and it will give you their categories.


texts = [" Virtual musicians play interactive gig.",
         "China's economy grows 18.3% in post-Covid comeback.",
         "FA Cup: Watch all Chelsea's goals from their journey to the semi-finals"]

text_features = tf_idf.transform(texts)
predictions = model.predict(text_features)
n = 0
for text, predicted in zip(texts, predictions):
    n = n+1
    print(n,". '{}'".format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")

