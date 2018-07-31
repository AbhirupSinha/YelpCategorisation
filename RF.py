#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import warnings
import string
#import re
#import sys
import gc
import pickle

#regex = re.compile('[^a-zA-Z]')
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Datasets/yelp_training_set_review.csv',encoding='latin1',usecols=["user_id","text","stars"],dtype={'stars':'uint8'})
dataset = dataset.drop_duplicates(["text","user_id"], keep = 'last')
dataset = dataset.dropna(subset = ['text'], axis = 0)
dataset = dataset.reset_index(drop = True)
#print(sys.getsizeof(dataset))
gc.collect()

rating_text_list = []
for i in range(len(dataset)):
    if int(dataset['stars'][i]) in [4, 5]:
        rating_text_list.append(3)
    elif int(dataset['stars'][i])==3:
        rating_text_list.append(2)
    elif int(dataset['stars'][i]) in [2, 1]:
        rating_text_list.append(1)
rating_text_dataset = pd.DataFrame(rating_text_list, columns = ['Ratings'],dtype='uint8')
dataset = dataset.join(rating_text_dataset)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
try:
    with open ('corpus.pkl', 'rb') as fp:
        corpus = pickle.load(fp)
except FileNotFoundError:
    corpus = []

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def penn_to_wn(tag):
    return get_wordnet_pos(tag)

for i in range(len(corpus), len(dataset)):
    review = dataset['text'][i]
    review = [lemmatizer.lemmatize(word,pos = penn_to_wn(nltk.pos_tag([word])[0][1])) for word 
            in word_tokenize(review) if word not in string.punctuation]
    review = ' '.join(review)
    corpus.append(review)
    
gc.collect()
#print(sys.getsizeof(corpus))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(decode_error='ignore',stop_words='english',lowercase=True, binary=False,
                        analyzer='word',token_pattern='[A-z]{3,}',ngram_range=(1,1),min_df=0.1)
y = dataset.loc[:, ['Ratings']].values
#print(sys.getsizeof(y))
X = tfidf.fit_transform(corpus[:len(dataset)]).toarray().astype(dtype='float32',copy=False)
#print(sys.getsizeof(X))

#Star ratings distribution before sampling
star_count = dataset['Ratings'].value_counts()
labels = ['Good','Bad','Average']
values = list(star_count.values)
colors = ['yellow','green','grey']
plt.pie(values,labels=labels,colors=colors,autopct='%1.2f%%',textprops={'fontsize':14})
plt.show()
gc.collect()

with open('corpus.pkl', 'wb') as fp:
    pickle.dump(corpus, fp)

#del dataset
#del corpus
gc.collect()

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=0,n_jobs=-1) 
X_resampled, y_resampled = sm.fit_sample(X, y)
X_resampled = X_resampled.astype(dtype='float32',copy=False)
gc.collect()

#Star ratings distribution after sampling
from collections import Counter
colors = ['yellow','green','grey']
counter = Counter(y_resampled)
df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
labels = ['Good','Bad','Average']
values = list(df['count'])
plt.pie(values,labels=labels,colors=colors,autopct='%1.2f%%',textprops={'fontsize':14})
plt.show()
gc.collect()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0,n_jobs=-1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from math import sqrt
from imblearn.metrics import classification_report_imbalanced
print("Accuracy: "+str(accuracy_score(y_test,y_pred)))
print("Root Mean Square Error: "+str(sqrt(mean_squared_error(y_test, y_pred))))
print("Confusion Matrix: \n"+str(confusion_matrix(y_test, y_pred).astype(dtype='uint16')))
print("Classification Report: \n"+str(classification_report_imbalanced(y_test,y_pred,labels=[3,2,1],digits=3,target_names=['Good','Average','Bad'])))
gc.collect()
