# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# --------------------------# import statements ------------------------------------


from flask import Flask, jsonify, render_template, request, send_from_directory
import flask
#from flask_caching import Cache
import csv
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
# import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename



from nltk.corpus import stopwords
# You will have to download the set of stop words the first time
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
from nltk.stem import PorterStemmer
porter = PorterStemmer()

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
#from gensim.sklearn_api import W2VTransformer
from gensim.models import phrases, word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


import seaborn as sns; sns.set()  # for plot styling

# from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


# %matplotlib inline
# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
#from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import davies_bouldin_score
from sklearn import metrics

import itertools
import math


import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import numpy as np
from sklearn.cluster import SpectralClustering


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from gensim import corpora, models
import pyLDAvis
#import pyLDAvis.gensim
#import pdfkit

import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import urllib
#from flask_cachebuster import CacheBust
print(gensim.__version__)
input()

# from flask_cors import CORS, cross_origin
# from flask_restful import Resource, Api
# from json import dumps
# from flask_jsonpify import jsonify
# ----------------------------------------------------------------------------------------------------------

# --------------------------------- Get data and PREPROCESSING ---------------------------------------------------------------------------------
# preprocessing
#


# The data frame is generated from the csv file and null values are removed
# The function returns the DataFrame - with text in it

def getdataframe():

    data = pd.read_csv('file.csv')
    data['index'] = data.index

    df = data[data['text'].notna()]
    Text = pd.concat([df['index'], df['text']],  axis=1, keys=['index', 'Description'])
    Text = Text.dropna()
    return Text

# The scarping function to scrape news from google
# The function returns the news articles in a form of a list

def gnews():
    query = ["'corona virus'", "'economy'", "'hollywood'", "'technology'"]
    number_result = 40
    ua = UserAgent()
    links = []
    titles = []
    descriptions = []
    for q in query:
        q = urllib.parse.quote_plus(q)
        google_url = "https://www.google.com/search?q=" + q + "&num=" + str(number_result)
        response = requests.get(google_url, {"User-Agent": ua.random})
        soup = BeautifulSoup(response.text, "html.parser")

        result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})


        for r in result_div:
            # Checks if each element is present, else, raise exception
            try:
                link = r.find('a', href = True)
                title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
                description = r.find('div', attrs={'class':'s3v9rd'}).get_text()

                # Check to make sure everything is present before appending
                if link != '' and title != '' and description != '':
                    links.append(link['href'])
                    titles.append(title)
                    descriptions.append(description)
            # Next loop if one element is not present
            except:
                continue
    return descriptions



# Function to remove stop words
# Input: documents in form of each row in data frame column
# Returns: list

def remove_stopwords(documents):
    x = []
    c = 0

    for line in documents:
        line = line.lower()
        l = line.split()
        if (len(l) < 5):
            c += 1
        else:
            cleaned = [word for word in l if word not in stop_words]
            s = " "
            s = s.join(cleaned)
            x.append(s)
    return x


# Function to remove punctuation marks
# Input: documents in form of each row in data frame column (after removing stop words)
# Returns: individial document as string

def Punctuation(string):

    # punctuation marks
    punctuations = '''!()-[];:_'",'<>./?@#$%^&*_{}~'''
    for x in string:
        if x in punctuations:
            string = string.replace(x, "")
    return string

# Function to stem the documents
# Input: documents in form of each row in data frame column (after removing stop words and punctuation)
# Returns: individial document as string


def stemmer(stringlist):
    s = " "
    for word in stringlist:
        s = s + porter.stem(word) + " "
    return s


# preprocessing function.
# Takes the raw data from data frame column and applies all the above three functions.
# 1. calls Remove stop words
# 2. calls remove punctuation
# 3. calls stemming
# Returns: The updated dataset after all the preprocessing is done.

def preprocessing():
    Text_raw = getdataframe()
    x = remove_stopwords(Text_raw['Description'])

    Temp = pd.DataFrame(x, columns = ['removed_stopword'])
    Temp['index'] = Temp.index


    Text = pd.concat([Temp['index'], Temp['removed_stopword']],  axis=1, keys=['index', 'removed_stopword'])

    u = []
    for line in Text['removed_stopword']:
        x = Punctuation(line)
        u.append(x)
    Text['removed_punctuation'] = u

    s = []
    for line in Text['removed_punctuation']:
        l = line.split(" ")
        x = stemmer(l)
        s.append(x)

    Text['Updated'] = s
    return Text




# ------------------------------------ALGORITHMS--------------------------------------------------




# -------------------------------------1. Word 2 vec -------------------------------------------------------------------
# WORD2VEC

def vectorizer(sent, model):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = model[w]
            else:
                vec = np.add(vec, model[w])
            numw += 1
        except:
            pass
    return np.asarray(vec) / numw

# Word2Vector with bigrams
# Returns: Vectors

def word2vec_bigram(datanew, d, w, m):
    bigrams = phrases.Phrases(datanew)
    # model = Word2Vec(datanew, size=100, window= 5, min_count=2, workers=4)
    w2v_bigram_model = Word2Vec(bigrams[datanew], size=d, window= w, min_count=m, workers=4)
    word_vectors = w2v_bigram_model.wv
    #index2word = w2v_bigram_model.wv.index_to_key
    # print (w2v_bigram_model.similarity('Skin', 'HIV'))
    # print(w2v_bigram_model.most_similar(["Connective"]))
    word_vectors = w2v_bigram_model.wv
    #return w2v_bigram_model
    return word_vectors

# Word2vec with bigrams and pca
# Returns: Vectors

def pca(vectors):
    #print('vectors:', vectors, flush=True)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(vectors)
    word2vec_vectors_pca = pd.DataFrame(data = principalComponents
                 , columns = ['0', '1'])
    return word2vec_vectors_pca

# Word2vec with bigrams and tfidf''
# Returns: Vectors

def w2v_tfidf(Text, datanew, d, w, m):
    model = TfidfVectorizer()
    model.fit(Text['Updated'])
    dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
    tfidf_feat = model.get_feature_names()

    w2v_tfidf_model = Word2Vec(datanew, min_count=m, size=d,window= w, workers = 4)
    w2v_words = list(w2v_tfidf_model.wv.index_to_key)


    tfidf_sent_vectors = []
    row= 0
    for sent in datanew:
        sent_vec = np.zeros(d)
        weight_sum = 0
        for word in sent:
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_tfidf_model.wv[word]
                tf_idf = dictionary[word] * (sent.count(word)/len(sent))

                sent_vec += (vec * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_sent_vectors.append(sent_vec)
        row += 1
    return tfidf_sent_vectors



# Reads data and converts into the list of documents
def word2vec():
    Text = preprocessing()
    data = Text['Updated'].to_list()
    datanew = []

    # Converts the dataframe to list and has words in the form of array of lists
    for d in data:
        datanew.append(d.split())

    return datanew


# Setting window size as a hyperprameter
# Retrurns the wndow size - parameter and respecrive distortion value as a form of list
def hyperparametersW(datanew, clusters):
    bigrams = phrases.Phrases(datanew)
    params = []
    distortions = []
    for p in range(1, 10):
        model1 = Word2Vec(datanew, vector_size=50, window=p, min_count=2, workers=4)
        word_vectors = model1.wv
        #print('word_vectors:',word_vectors, flush=True)
        l = []
        for i in datanew:
            l.append(vectorizer(i, word_vectors))

        word2vec_vectors = pd.DataFrame([l[0]])
        for i in range(1, len(l)):
            additional_row = pd.DataFrame([l[i]])
            word2vec_vectors = word2vec_vectors.append(additional_row)
        word2vec_vectors[word2vec_vectors==np.inf]=np.nan
        word2vec_vectors.fillna(word2vec_vectors.mean(), inplace=True)

        params.append(p)
        no = clusters
        kmeans = KMeans (n_clusters = no,
                        max_iter = 100,
                        init = 'k-means++',
                        n_init = 1)

        K_labels = kmeans.fit_predict(word2vec_vectors)
        distortions.append(sum(np.min(cdist(word2vec_vectors, kmeans.cluster_centers_, 'cosine'), axis=1)) / word2vec_vectors.shape[0])
    return params, distortions



# Setting dimension size as a hyperprameter
# Retrurns the dimension size - parameter and respecrive distortion value as a form of list
def hyperparametersD(datanew, clusters):
    bigrams = phrases.Phrases(datanew)
    params = []
    distortions = []
    for n in range(10, 500, 50):
        model1 = Word2Vec(datanew, vector_size=n, window=5, min_count=2, workers=4)
        word_vectors = model1.wv
        l = []
        for i in datanew:
            l.append(vectorizer(i, word_vectors))

        word2vec_vectors = pd.DataFrame([l[0]])
        for i in range(1, len(l)):
            additional_row = pd.DataFrame([l[i]])
            word2vec_vectors = word2vec_vectors.append(additional_row)
        word2vec_vectors[word2vec_vectors==np.inf]=np.nan
        word2vec_vectors.fillna(word2vec_vectors.mean(), inplace=True)

        params.append(n)
        no = clusters
        kmeans = KMeans (n_clusters = no,
                        max_iter = 100,
                        init = 'k-means++',
                        n_init = 1)

        K_labels = kmeans.fit_predict(word2vec_vectors)
        distortions.append(sum(np.min(cdist(word2vec_vectors, kmeans.cluster_centers_, 'euclidean'), axis=1)) / word2vec_vectors.shape[0])
    return params, distortions



# ----------------------------------------------2. LSA ------------------------------------------------------
# LSA
#
# Returns: Similarity matrix
# LSA algorithm is called.

def lsa_similarity_matrix(Text):
#     vectorizer values
    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    dtm = vectorizer.fit_transform(Text['Updated'])
    vectorizer.get_feature_names()

#     Lsa method - reduced to 2 vectors (USES SVD)
    lsa = TruncatedSVD(2, algorithm = 'randomized')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

#     Prints the data frame with 2 components as rows and words as columns
    pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns =
    vectorizer.get_feature_names())

#     Prints the data frame with 2 vector components as columns and documents as rows
    pd.DataFrame(dtm_lsa,index = Text['index'], columns = ["component_1","component_2"])

#     Similarity between the 2 Component vectors.
    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)

#     prints the similarity between all the documents in the form of matrix
    similarity_matrix = pd.DataFrame(similarity,index=Text['index'], columns=Text['index'])
    return similarity_matrix


# ------------------------------------------3. NMF --------------------------------------------------------
# NMF

# Returns nmf model and nmf labels

def nmf(Text, num_topics):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(Text['Updated'])
    nmf_model = NMF(n_components=num_topics, random_state=101)
    # print("nmf_model")
    # print(nmf_model)
    nmf_model = nmf_model.fit(dtm)
    freq_words = {}
    # print("Words in the model\n")
    for index, topic in enumerate(nmf_model.components_):
        # print(f"The new models# {index}")
        words = [tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]]
        freq_words[index] = words
        # print(words)
        # print('\n')


    topic_model = nmf_model.transform(dtm)
    nmf_labels = topic_model.argmax(axis =1)
    # print("nmf lables")
    # print(nmf_labels)
    Text['topic'] = topic_model.argmax(axis =1)
    # model = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
    # Text['Title'] = Text['topic'].map(model)
    # Text.head()
    counts = Text['topic'].value_counts()
#     print("nmf Counts")
#     print(counts)
    return nmf_model, freq_words, topic_model, nmf_labels, counts

# ---------------------------------------------------------------------------------------------------







# -------------------------------------VISUALIZATION--------------------------------------------
# VISUALIZATION

# Converts lables into figure and frequency count of labels
def pie_chart(labels):
    import matplotlib.pyplot as plt

    clusters = Counter(labels).keys() # equals to list(set(words))
    count = Counter(labels).values() # counts the elements' frequency

    list_of_tuples = list(zip(clusters, count))
    df = pd.DataFrame(list_of_tuples, columns = ['Cluster', 'Count'])
#     print(df)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    ax.pie(count, labels = clusters,autopct='%1.2f%%')
    return fig, df

# PCA plot - Scatter
# Plots 2D labels along with cluster centres
def plot(df, labels, model):
    import matplotlib.pyplot as plott
    plott.scatter(df['0'], df['1'], c= labels, s=20, cmap='viridis')

    centers = model.cluster_centers_
    plott.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    return plott



def wordfreq(labels):
    datanew = word2vec()
    wordfreq = {}
    for index, doc in enumerate(datanew):
    #     print(str(labels[index]) + ":" + str(doc))
        i = labels[index]
        if i not in wordfreq:
            temp = {}

            for word in doc:
                if word not in temp:
                     temp[word] = 1
                else:
                     temp[word] += 1
            wordfreq[i] = temp
        else:
            atemp = wordfreq[i]
            for word in doc:
                if word not in atemp:
                     atemp[word] = 1
                else:
                     atemp[word] += 1
            wordfreq[i] = atemp
    for k in wordfreq:
        d = sorted(wordfreq[k].items(), key = lambda l:(l[1]), reverse = True)
        wordfreq[k] = d[:21]
    return wordfreq

# ----------------------------------------- CLUSTERING -------------------------------------------
# KMEANS AND HIERARCHEAL CLUSTERING, SPECTRAL CLUSTERING

# Returns models and labels
def kmeans_euclidean(df, n):
    X = df
    number = n
    kmeans = KMeans (n_clusters = number,
                    max_iter = 100,
                    init = 'k-means++',
                    n_init = 1)
    euclidean_labels = kmeans.fit_predict(X)
#     print(euclidean_labels)
    return kmeans, euclidean_labels

# Returns models and labels
def h_euclidean(X, n):
    Hcluster_euclidean = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    Hlabel_euclidean = Hcluster_euclidean.fit_predict(X)
    return Hcluster_euclidean, Hlabel_euclidean


# Returns labels
def spectral_cluster(similarity_matrix, n):

    lsa_cluster_array = SpectralClustering(n).fit_predict(similarity_matrix)
    return lsa_cluster_array


# -----------------------------------------------------------------------------------------------------
# Evaluations


# Internal Evaluations - Silhouette, DAvies and Calinski
# Returns ; List of all scores
def clusterReport(dataset, labels, name):
    report = []
    silhoutte = metrics.silhouette_score(dataset, labels, metric='euclidean')
    report.append(silhoutte)
    calinski = metrics.calinski_harabasz_score(dataset, labels)
    report.append(calinski)
    davies = davies_bouldin_score(dataset, labels)
    report.append(davies)
    report.insert(0, name)
    return report



#counting pair evaluation
# Returns ; List of all scores
def counting_pair_measures(labels1, labels2):
    n11 = n10 = n01 = n00 = 0
    n = len(labels1)
    for i, j in itertools.combinations(range(n), 2):
        c1 = labels1[i] == labels1[j]
        c2 = labels2[i] == labels2[j]
        if c1 and c2:
            n11 += 1
        elif c1 and not c2:
            n10 += 1
        elif not c1 and c2:
            n01 += 1
        elif not c1 and not c2:
            n00 += 1
    jacard = float(n11) / (n11 + n10 + n01)
    rand = (n11 + n00)/(n)
    p = n11/(n11+n10)
    r = n11/(n11+n01)
    fowlkes = math.sqrt(p*r)
    mirkin = 2*(n01 + n10)
    partition = n00
    dice = (2*n11)/((2*n11) + n10 + n01)
    report = []
    report.extend([n00, n11, n10, n01,p, r, jacard, rand,fowlkes, mirkin, partition, dice ])
    return report


# The array of all the evaluation results of all the above algorithms
def evaluation_array(Text, num_topics):
    # W2V
    datanew = word2vec()
    dimension = 60
    window = 5
    mincount = 2
    l = []
    w2v_bigram_model = word2vec_bigram(datanew, dimension, window, mincount)
    for i in datanew:
        l.append(vectorizer(i, w2v_bigram_model))
    #print('l:', l, flush=True)
    word2vec_vectors = []

    word2vec_vectors = pd.DataFrame([l[0]])

    for i in range(1, len(l)):
        additional_row = pd.DataFrame([l[i]])
        word2vec_vectors = word2vec_vectors.append(additional_row)


    word2vec_vectors[word2vec_vectors==np.inf]=np.nan
    word2vec_vectors.fillna(word2vec_vectors.mean(), inplace=True)

    # WORD2VEC PCA
    #print('word2vec_vectors:', word2vec_vectors, flush=True)
    word2vec_vectors_pca = pca(word2vec_vectors)

    # WORD2VEC TFIDF
    tfidf_sent_vectors = w2v_tfidf(Text, datanew, dimension, window, mincount)
    word2vec_vectors_tfidf = np.array(tfidf_sent_vectors)

    # WORD2VEC PCA TFIDF
    word2vec_vectors_tfidf_pca = pca(word2vec_vectors_tfidf)


    # KMeans
    w2v_euclidean_model, w2v_euclidean_labels = kmeans_euclidean(word2vec_vectors, num_topics)
    w2vpca_euclidean_model, w2vpca_euclidean_labels = kmeans_euclidean(word2vec_vectors_pca, num_topics)
    w2vtfidf_euclidean_model, w2vtfidf_euclidean_labels = kmeans_euclidean(word2vec_vectors_tfidf, num_topics)
    w2v_tfidf_pca_euclidean_model, w2v_tfidf_pca_euclidean_labels = kmeans_euclidean(word2vec_vectors_tfidf_pca, num_topics)

    # HIERARCHEAL
    Hw2v_euclidean_model, Hw2v_euclidean_labels = h_euclidean(word2vec_vectors, num_topics)
    Hw2v_pca_euclidean_model, Hw2v_pca_euclidean_labels = h_euclidean(word2vec_vectors_pca, num_topics)
    Hw2v_tfidf_euclidean_model, Hw2v_tfidf_euclidean_labels = h_euclidean(word2vec_vectors_tfidf, num_topics)
    Hw2v_tfidf_pca_euclidean_model, Hw2v_tfidf_pca_euclidean_labels = h_euclidean(word2vec_vectors_tfidf_pca, num_topics)

    word2vec_Kmeans_report = clusterReport(word2vec_vectors, w2v_euclidean_labels, "word2vec_Kmeans")
    word2vec_pca_Kmeans_report = clusterReport(word2vec_vectors_pca, w2vpca_euclidean_labels, "word2vec_pca_Kmeans")
    word2vec_tfidf_Kmeans_report = clusterReport(word2vec_vectors_tfidf, w2vtfidf_euclidean_labels, "word2vec_tfidf_Kmeans")
    word2vec_tfidf_pca_Kmeans_report = clusterReport(word2vec_vectors_tfidf_pca, w2v_tfidf_pca_euclidean_labels, "word2vec_tfidf_pca_Kmeans")

    word2vec_h_report = clusterReport(word2vec_vectors, Hw2v_euclidean_labels, "word2vec_Hierarchical")
    word2vec_pca_h_report = clusterReport(word2vec_vectors_pca, Hw2v_pca_euclidean_labels, "word2vec_pca_Hierarchical")
    word2vec_tfidf_h_report = clusterReport(word2vec_vectors_tfidf, Hw2v_tfidf_euclidean_labels, "word2vec_tfidf_Hierarchical")
    word2vec_tfidf_pca_h_report = clusterReport(word2vec_vectors_tfidf_pca, Hw2v_tfidf_pca_euclidean_labels, "word2vec_tfidf_pca_Hierarchical")


    # NMF
    nmf_model, freq_words, nmf_vec, nmf_labels, counts = nmf(Text, num_topics)
    nmf_report = clusterReport(nmf_vec, nmf_labels, "NMF")


    # LSA
    similarity_matrix = lsa_similarity_matrix(Text)
    lsa_cluster_array = spectral_cluster(similarity_matrix, num_topics)
    from sklearn.cluster import DBSCAN
    lsa_dbscan = DBSCAN(min_samples=1).fit_predict(similarity_matrix)
    lsa_spectral_report = clusterReport(similarity_matrix, lsa_cluster_array, "LSA with spectral")
    if len(set(lsa_dbscan)) == 1:
        lsa_dbscan_report = ["LSA with DB", np.NaN,np.NaN,np.NaN]
    else:
        lsa_dbscan_report = clusterReport(similarity_matrix, lsa_dbscan, "LSA with DBScan")


    # D2V
    vec = doc2vec(Text)
    d2v_euclidean_model, d2v_euclidean_labels = kmeans_euclidean(vec, num_topics)
    d2v_report = clusterReport(vec, d2v_euclidean_labels, "D2V")


    lda_report = ["LDA", np.NaN, np.NaN, np.NaN]


    evaluation = [nmf_report, lsa_spectral_report, lsa_dbscan_report, d2v_report, lda_report,
    word2vec_Kmeans_report, word2vec_pca_Kmeans_report, word2vec_tfidf_Kmeans_report, word2vec_tfidf_pca_Kmeans_report,
    word2vec_h_report, word2vec_pca_h_report, word2vec_tfidf_h_report, word2vec_tfidf_pca_h_report]

    return evaluation




# ------------------------------------------ FLASK ----------------------------------------------------------------

app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Home page
@app.route('/')
def index():
   return render_template('index.html')

# rederning the upload page to upload files
# The file is in csv or xlxr format and saves into local folder named file.csv
# Takes cluster count as an input
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      print(f)
      no_clusters = request.form['no_clusters']
      f.save(secure_filename(f.filename))
      if '.csv' in f.filename:
          dataset = pd.read_csv(f.filename)
      elif'.xlsx' in f.filename:
          dataset = pd.read_excel(f.filename)
      dataset.to_csv('file.csv')

      Text = preprocessing()
      a = int(no_clusters)
      evaluation = evaluation_array(Text, a)
      #print('eval:',evaluation, flush=True)
      dataframe = pd.DataFrame(evaluation, columns =["Name", "Silhoutte Score", "Calinski Score", "Davies Score"])
      dataframe = dataframe.sort_values(by=['Silhoutte Score'], ascending=False)

      # print("heyyyyyyyyyyy", no_clusters)
      return render_template("upload.html",old = dataset.shape, new = Text.shape, res = "Analysis of uploaded file", clusters= no_clusters,
        evaluation = evaluation,
        dataframe = [dataframe.to_html(classes='data')], titles1=dataframe.columns.values)

# rederning the demo page to show three demo examples
@app.route('/demo', methods = ['GET', 'POST'])
def demo():
    return render_template('demo.html')


# renders the uploader page
@app.route('/uploader_gnews', methods = ['GET', 'POST'])
def upload_gnews():

    no_clusters = 4
    des = gnews()
    # print(des)


    dataset = pd.DataFrame(des, columns = ['text'])
    dataset.to_csv('file.csv')

    Text = preprocessing()
    evaluation = evaluation_array(Text, no_clusters)
    dataframe = pd.DataFrame(evaluation, columns =["Name", "Silhoutte Score", "Calinski Score", "Davies Score"])
    dataframe = dataframe.sort_values(by=['Silhoutte Score'], ascending=False)


    return render_template("upload.html",old = dataset.shape, new = Text.shape, res = "Analysis of Text taken from Google news", clusters= no_clusters,
        evaluation = evaluation,
        dataframe = [dataframe.to_html(classes='data')], titles1=dataframe.columns.values)


# renders the uploader page
@app.route('/uploader_amazon', methods = ['GET', 'POST'])
def upload_amazon():

    no_clusters = 6
    dataset = pd.read_excel('AmazonReviews.xlsx')

    dataset.to_csv('file.csv')

    Text = preprocessing()
    evaluation = evaluation_array(Text, no_clusters)
    dataframe = pd.DataFrame(evaluation, columns =["Name", "Silhoutte Score", "Calinski Score", "Davies Score"])
    dataframe = dataframe.sort_values(by=['Silhoutte Score'], ascending=False)

    return render_template("upload.html",old = dataset.shape, new = Text.shape, res = "Analysis of Text taken from Amazon - Product Reviews", clusters= no_clusters,
        evaluation = evaluation,
        dataframe = [dataframe.to_html(classes='data')], titles1=dataframe.columns.values)

# renders the uploader page
@app.route('/uploader_nih', methods = ['GET', 'POST'])
def upload_nih():

    no_clusters = 4
    data = pd.read_excel('nih.xlsx')
    data['text'] = data['Description'].combine_first(data['Topics'])
    data.to_csv('file.csv')

    Text = preprocessing()
    evaluation = evaluation_array(Text, no_clusters)
    dataframe = pd.DataFrame(evaluation, columns =["Name", "Silhoutte Score", "Calinski Score", "Davies Score"])
    dataframe = dataframe.sort_values(by=['Silhoutte Score'], ascending=False)

    return render_template("upload.html",old = data.shape, new = Text.shape, res = "Analysis of Text taken from Amazon - Product ReviewsNIH medical data", clusters= no_clusters,
        evaluation = evaluation,
        dataframe = [dataframe.to_html(classes='data')], titles1=dataframe.columns.values)





# -------------------------------------------------------------------------------------------------------------------

# word 2 vec page tp set the hyperparamaters
@app.route('/word2Vec/<no>',methods = ['POST', 'GET'])
def word2vecresult(no):
    print("heyyyyyyyyyyyyyyyy", no)
    Text = preprocessing()
    # fetches word2vec data
    datanew = word2vec()
    a = int(no)
    clusters = a


    #setting hyperparamaters
    params_dimension , distortion_dimension = hyperparametersD(datanew, clusters)
    params_window , distortion_window = hyperparametersW(datanew, clusters)
    import matplotlib.pyplot as plt

    plt.plot(params_dimension, distortion_dimension, 'bx-')
    plt.xlabel('parameters - dimension size')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal parameters for dimension')
    plt.savefig('static/images/w2v_hyper_dimension.jpg')
    plt.clf()

    import matplotlib.pyplot as plt
    plt.plot(params_window, distortion_window, 'bx-')
    plt.xlabel('parameters - window size')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal parameters for window size')
    plt.savefig('static/images/w2v_hyper_window.jpg')
    plt.clf()

    return render_template("word2vec.html",clusters = clusters)


# w2v analysis page
@app.route('/word2VecAnalysis/<no>',methods = ['POST', 'GET'])
def word2vecAnalysis(no):
    if request.method == 'POST':
        dimension = request.form['dimension']
        window = request.form['window']
        mincount = request.form['mincount']

    Text = preprocessing()
    # fetches word2vec data
    datanew = word2vec()
    dimension = int(dimension)
    window = int(window)
    mincount = int(mincount)


    # WORD2VEC BIGRAMS
    l = []
    w2v_bigram_model = word2vec_bigram(datanew, dimension, window, mincount)
    for i in datanew:
        l.append(vectorizer(i, w2v_bigram_model))

    word2vec_vectors = []

    word2vec_vectors = pd.DataFrame([l[0]])

    for i in range(1, len(l)):
        additional_row = pd.DataFrame([l[i]])
        word2vec_vectors = word2vec_vectors.append(additional_row)


    word2vec_vectors[word2vec_vectors==np.inf]=np.nan
    word2vec_vectors.fillna(word2vec_vectors.mean(), inplace=True)

    # WORD2VEC PCA
    word2vec_vectors_pca = pca(word2vec_vectors)

    # WORD2VEC TFIDF
    tfidf_sent_vectors = w2v_tfidf(Text, datanew, dimension, window, mincount)
    word2vec_vectors_tfidf = np.array(tfidf_sent_vectors)

    # WORD2VEC PCA TFIDF
    word2vec_vectors_tfidf_pca = pca(word2vec_vectors_tfidf)

    a = int(no)
    num_topics = a

    print(a)

    # KMeans
    w2v_euclidean_model, w2v_euclidean_labels = kmeans_euclidean(word2vec_vectors, num_topics)
    w2vpca_euclidean_model, w2vpca_euclidean_labels = kmeans_euclidean(word2vec_vectors_pca, num_topics)
    w2vtfidf_euclidean_model, w2vtfidf_euclidean_labels = kmeans_euclidean(word2vec_vectors_tfidf, num_topics)
    w2v_tfidf_pca_euclidean_model, w2v_tfidf_pca_euclidean_labels = kmeans_euclidean(word2vec_vectors_tfidf_pca, num_topics)

    # HIERARCHEAL
    Hw2v_euclidean_model, Hw2v_euclidean_labels = h_euclidean(word2vec_vectors, num_topics)
    Hw2v_pca_euclidean_model, Hw2v_pca_euclidean_labels = h_euclidean(word2vec_vectors_pca, num_topics)
    Hw2v_tfidf_euclidean_model, Hw2v_tfidf_euclidean_labels = h_euclidean(word2vec_vectors_tfidf, num_topics)
    Hw2v_tfidf_pca_euclidean_model, Hw2v_tfidf_pca_euclidean_labels = h_euclidean(word2vec_vectors_tfidf_pca, num_topics)

    Text['w2v kmeans'] = w2v_euclidean_labels
    Text['w2v pca kmeans'] = w2vpca_euclidean_labels
    Text['w2v tfidf kmeans'] = w2vtfidf_euclidean_labels
    Text['w2v tfidf pca kmeans'] = w2v_tfidf_pca_euclidean_labels
    Text['w2v Hierarchical'] = Hw2v_euclidean_labels
    Text['w2v pca Hierarchical'] = Hw2v_pca_euclidean_labels
    Text['w2v tfidf Hierarchical'] = Hw2v_tfidf_euclidean_labels
    Text['Hw2v_tfidf_pca_euclidean_labels'] = Hw2v_tfidf_pca_euclidean_labels

    Text = Text.sort_values(by=['w2v pca kmeans'])

    Text.to_csv('file_w2v.csv')
    # Save plots - kmeans and hierarcheal
    w2v_kmeans_pie, w2v_kmeans_count = pie_chart(w2v_euclidean_labels)
    w2v_kmeans_pie.savefig('static/images/w2v_kmeans_pie.jpg')
    w2v_kmeans_pie.clf()
    words1 = wordfreq(w2v_euclidean_labels)

    w2vpca_kmeans_pie, w2vpca_kmeans_count = pie_chart(w2vpca_euclidean_labels)
    w2vpca_kmeans_pie.savefig('static/images/w2vpca_kmeans_pie.jpg')
    w2vpca_kmeans_pie.clf()
    words2 = wordfreq(w2vpca_euclidean_labels)


    w2vtfidf_kmeans_pie, w2vtfidf_kmeans_count = pie_chart(w2vtfidf_euclidean_labels)
    w2vtfidf_kmeans_pie.savefig('static/images/w2vtfidf_kmeans_pie.jpg')
    w2vtfidf_kmeans_pie.clf()
    words3 = wordfreq(w2vtfidf_euclidean_labels)

    w2vpca_tfidf_kmeans_pie, w2vpca_tfidf_kmeans_count = pie_chart(w2v_tfidf_pca_euclidean_labels)
    w2vpca_tfidf_kmeans_pie.savefig('static/images/w2vpca_tfidf_kmeans_pie.jpg')
    w2vpca_tfidf_kmeans_pie.clf()
    words4 = wordfreq(w2v_tfidf_pca_euclidean_labels)


    w2v_h_pie, w2v_h_count = pie_chart(Hw2v_euclidean_labels)
    w2v_h_pie.savefig('static/images/w2v_h_pie.jpg')
    w2v_h_pie.clf()
    words5 = wordfreq(Hw2v_euclidean_labels)

    w2vpca_h_pie, w2vpca_h_count = pie_chart(Hw2v_pca_euclidean_labels)
    w2vpca_h_pie.savefig('static/images/w2vpca_h_pie.jpg')
    w2vpca_h_pie.clf()
    words6 = wordfreq(Hw2v_pca_euclidean_labels)


    w2vtfidf_h_pie, w2vtfidf_h_count = pie_chart(Hw2v_tfidf_euclidean_labels)
    w2vtfidf_h_pie.savefig('static/images/w2vtfidf_h_pie.jpg')
    w2vtfidf_h_pie.clf()
    words7 = wordfreq(Hw2v_tfidf_euclidean_labels)

    w2vpca_tfidf_h_pie, w2vpca_tfidf_h_count = pie_chart(Hw2v_tfidf_pca_euclidean_labels)
    w2vpca_tfidf_h_pie.savefig('static/images/w2vpca_tfidf_h_pie.jpg')
    w2vpca_tfidf_h_pie.clf()
    words8 = wordfreq(Hw2v_tfidf_pca_euclidean_labels)



    # Scatter plot for pca
    # scatter_plot = plot(word2vec_vectors_pca, w2vpca_euclidean_labels, w2vpca_euclidean_model)
    import matplotlib.pyplot as plott


    plott.scatter(word2vec_vectors_pca['0'], word2vec_vectors_pca['1'], c= w2vpca_euclidean_labels, s=20, cmap='viridis')

    centers = w2vpca_euclidean_model.cluster_centers_
    plott.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plott.savefig('static/images/scatter.jpg')
    plott.clf()

    # Internal evaluation
    word2vec_Kmeans_report = clusterReport(word2vec_vectors, w2v_euclidean_labels, "word2vec_Kmeans_report")
    word2vec_pca_Kmeans_report = clusterReport(word2vec_vectors_pca, w2vpca_euclidean_labels, "word2vec_pca_Kmeans_report")
    word2vec_tfidf_Kmeans_report = clusterReport(word2vec_vectors_tfidf, w2vtfidf_euclidean_labels, "word2vec_tfidf_Kmeans_report")
    word2vec_tfidf_pca_Kmeans_report = clusterReport(word2vec_vectors_tfidf_pca, w2v_tfidf_pca_euclidean_labels, "word2vec_tfidf_pca_Kmeans_report")

    word2vec_h_report = clusterReport(word2vec_vectors, Hw2v_euclidean_labels, "word2vec_h_report")
    word2vec_pca_h_report = clusterReport(word2vec_vectors_pca, Hw2v_pca_euclidean_labels, "word2vec_pca_h_report")
    word2vec_tfidf_h_report = clusterReport(word2vec_vectors_tfidf, Hw2v_tfidf_euclidean_labels, "word2vec_tfidf_h_report")
    word2vec_tfidf_pca_h_report = clusterReport(word2vec_vectors_tfidf_pca, Hw2v_tfidf_pca_euclidean_labels, "word2vec_tfidf_pca_h_report")


    table = [word2vec_Kmeans_report, word2vec_pca_Kmeans_report, word2vec_tfidf_Kmeans_report, word2vec_tfidf_pca_Kmeans_report,
             word2vec_h_report, word2vec_pca_h_report, word2vec_tfidf_h_report, word2vec_tfidf_pca_h_report]

    internal_df = pd.DataFrame(table,
               columns =["Cluster Name", "Silhoutte Score", "Calinski Score", "Davies Score"])


    # counting_pair_list
    word2vec_k_h = counting_pair_measures(w2v_euclidean_labels, Hw2v_euclidean_labels)
    word2vec_pca_k_h = counting_pair_measures(w2vpca_euclidean_labels, Hw2v_pca_euclidean_labels)
    word2vec_tfidf_k_h = counting_pair_measures(w2vtfidf_euclidean_labels, Hw2v_tfidf_euclidean_labels)
    word2vec_tfidf_pca_k_h = counting_pair_measures(w2v_tfidf_pca_euclidean_labels, Hw2v_tfidf_pca_euclidean_labels)
    counting_pair_list = [word2vec_k_h, word2vec_pca_k_h, word2vec_tfidf_k_h, word2vec_tfidf_pca_k_h]

    counting_pair_df = pd.DataFrame(counting_pair_list,
                   columns =['n00', 'n11', 'n10', 'n01', 'precision', 'recall',
                             'Jacard Similarity', 'Rand Similarity', 'Fowlkes Similarity', 'Mirkin Similarity',
                             'Partition Similarity', 'Dice Similarity'], index = ['w2v', 'w2v with pca', 'w2v with tfidf', 'w2v with tfidf and pca'])


    return render_template("word2VecAnalysis.html",clusters = no, dimension = dimension, window= window, mincount = mincount,
                                 w2v_kmeans_count = [w2v_kmeans_count.to_html(classes='data')], titles1=w2v_kmeans_count.columns.values,
                                 w2vpca_kmeans_count = [w2vpca_kmeans_count.to_html(classes='data')], titles2=w2vpca_kmeans_count.columns.values,
                                 w2vtfidf_kmeans_count = [w2vtfidf_kmeans_count.to_html(classes='data')], titles3=w2vtfidf_kmeans_count.columns.values,
                                 w2vpca_tfidf_kmeans_count = [w2vpca_tfidf_kmeans_count.to_html(classes='data')], titles4=w2vpca_tfidf_kmeans_count.columns.values,

                                 w2v_h_count = [w2v_h_count.to_html(classes='data')], titles5=w2v_h_count.columns.values,
                                 w2vpca_h_count = [w2vpca_h_count.to_html(classes='data')], titles6=w2vpca_h_count.columns.values,
                                 w2vtfidf_h_count = [w2vtfidf_h_count.to_html(classes='data')], titles7=w2vtfidf_h_count.columns.values,
                                 w2vpca_tfidf_h_count = [w2vpca_tfidf_h_count.to_html(classes='data')], titles8=w2vpca_tfidf_h_count.columns.values,
                                 internal_df = [internal_df.to_html(classes='data')], titles_internal=internal_df.columns.values,
                                 counting_pair_df = [counting_pair_df.to_html(classes='data')], titles_cp=counting_pair_df.columns.values,
                                 words1 = words1, words2 = words2, words3 = words3, words4 = words4, words5 = words5, words6 = words6, words7 = words7, words8 = words8)



# lsa analysis page
@app.route('/LSA/<no>',methods = ['POST', 'GET'])
def lsaresult(no):
    Text = preprocessing()
    a = int(no)
    num_topics = a
    similarity_matrix = lsa_similarity_matrix(Text)
    lsa_cluster_array = spectral_cluster(similarity_matrix, num_topics)
    wordslsa = wordfreq(lsa_cluster_array)


    # lsa spectral
    lsa_spectral_pie, lsa_spectral_count = pie_chart(lsa_cluster_array)
    lsa_spectral_pie.savefig('static/images/lsa_spectral_pie.jpg')
    lsa_spectral_pie.clf()


    # lsa dbscan
    from sklearn.cluster import DBSCAN
    lsa_dbscan = DBSCAN(min_samples=1).fit_predict(similarity_matrix)
    lsa_db_pie, lsa_db_count = pie_chart(lsa_dbscan)
    lsa_db_pie.savefig('static/images/lsa_db_pie.jpg')
    lsa_db_pie.clf()
    wordsdb = wordfreq(lsa_dbscan)



    Text['LSA spectral label'] = lsa_cluster_array
    Text['LSA db label'] = lsa_dbscan
    Text = Text.sort_values(by=['LSA spectral label'])

    Text.to_csv('file_lsa.csv')

    # internal evaluation
    lsa_spectral_report = clusterReport(similarity_matrix, lsa_cluster_array, "LSA with spectral")
    if len(set(lsa_dbscan)) == 1:
        lsa_dbscan_report = ["LSA with DB", np.NaN,np.NaN,np.NaN]
    else:
        lsa_dbscan_report = clusterReport(similarity_matrix, lsa_dbscan, "LSA with DBScan")


    lsa_report = [lsa_spectral_report, lsa_dbscan_report]
    lsa_internal_df = pd.DataFrame(lsa_report,
                   columns =["Cluster Name", "Silhoutte Score", "Calinski Score", "Davies Score"])

    # counting_pair_measures
    lsa_spectral_dbscan = counting_pair_measures(lsa_cluster_array, lsa_dbscan)
    lsa_spectral_dbscan = [lsa_spectral_dbscan]

    lsa_counting_pair_df = pd.DataFrame(lsa_spectral_dbscan,
               columns =['n00', 'n11', 'n10', 'n01', 'precision', 'recall',
                         'Jacard Similarity', 'Rand Similarity', 'Fowlkes Similarity', 'Mirkin Similarity',
                         'Partition Similarity', 'Dice Similarity'], index = ["LSA between spectral and dbscan"])


    return render_template("lsa.html",clusters = no, lsa_spectral_count = [lsa_spectral_count.to_html(classes='data')], titles1=lsa_spectral_count.columns.values,
        lsa_db_count = [lsa_db_count.to_html(classes='data')], titles2=lsa_db_count.columns.values,
        lsa_internal_df = [lsa_internal_df.to_html(classes='data')], titles3=lsa_internal_df.columns.values,
        lsa_counting_pair_df = [lsa_counting_pair_df.to_html(classes='data')], titles4=lsa_counting_pair_df.columns.values,
        wordslsa = wordslsa, wordsdb = wordsdb)


# nmf analysis page
@app.route('/NMF/<no>',methods = ['POST', 'GET'])
def NMFresult(no):
    Text = preprocessing()
    a = int(no)
    num_topics = a
    nmf_model, freq_words, nmf_vec, nmf_labels, counts = nmf(Text, num_topics)

    # nmf clusters
    nmf_pie, nmf_count = pie_chart(nmf_labels)
    nmf_pie.savefig('static/images/nmf_pie.jpg')
    nmf_pie.clf
    wordsnmf = wordfreq(nmf_labels)

    # Internal evaluation
    nmf_report = clusterReport(nmf_vec, nmf_labels, "NMF")
    nmf_report = [nmf_report]
    nmf_df = pd.DataFrame(nmf_report,
                   columns =["NMF", "Silhoutte Score", "Calinski Score", "Davies Score"])

    Text['NMF labels'] = nmf_labels
    Text = Text.sort_values(by=['NMF labels'])
    Text.to_csv('file_nmf.csv')
    return render_template("nmf.html",clusters = no, nmf_count = [nmf_count.to_html(classes='data')], titles1=nmf_count.columns.values,
        nmf_df = [nmf_df.to_html(classes='data')], titles2=nmf_df.columns.values, wordsnmf = wordsnmf)


# LDA Function
def lda(corpus, dictionary_LDA, num_topics):

    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    lda_words = {}
    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
        # print(str(i)+": "+ topic)
        lda_words[i] = topic

    all_labels = []
    one_labels = []
    for i in corpus:
        a = lda_model[i]
        dic = {}
        for i in range(0, len(a)):
            dic[a[i][0]] = a[i][1]
        sort_orders = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        one_labels.append(sort_orders[0][0])
        all_labels.append(a)

    return lda_model, lda_words, all_labels, one_labels

# LDA analysis page
@app.route('/LDA/<no>',methods = ['POST', 'GET'])
def ldaresult(no):
    Text = preprocessing()
    a = int(no)
    num_topics = a
    # fetches word2vec data
    datanew = word2vec()
    dictionary_LDA = corpora.Dictionary(datanew)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in datanew]
    lda_model, lda_words, lda_all, lda_one = lda(corpus, dictionary_LDA, num_topics)

    lda_pie, lda_count = pie_chart(lda_one)
    lda_pie.savefig('static/images/lda_pie.jpg')
    lda_pie.clf()
    wordslda = wordfreq(lda_one)


    Text['lda one labels'] = lda_one
    Text['all labels'] = lda_all
    Text = Text.sort_values(by=['lda one labels'])
    Text.to_csv('file_lda.csv')
    return render_template("ldaAnalysis.html", lda_count = [lda_count.to_html(classes='data')], titles1=lda_count.columns.values, clusters = num_topics, wordslda = wordslda)


# LDA words page
@app.route('/LDA_words/<no>',methods = ['POST', 'GET'])
def ldawords(no):
    a = int(no)
    num_topics = a
    Text = preprocessing()
    # fetches word2vec data
    datanew = word2vec()
    dictionary_LDA = corpora.Dictionary(datanew)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in datanew]
    lda_model, lda_words, lda_all, lda_one = lda(corpus, dictionary_LDA, num_topics)

    p = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
    pyLDAvis.save_html(p, 'templates/ldaWords.html')
    return render_template("ldaWords.html")


# D2V function
# Returns: vectors
def doc2vec(Text):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    data = Text['Updated']

    tokenized_doc = []
    for d in data:
        tokenized_doc.append(word_tokenize(d.lower()))

    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
    n = Text.shape[0]
    Y = []
    for i in range(0, n):
        Y.append(model.dv[i])

    vec = np.array(Y)
    return vec

# D2V page
@app.route('/D2V/<no>', methods = ['GET', 'POST'])
def doc2vect(no):
    # a = int(no)
    # num_topics = a
    Text = preprocessing()
    # print("done")
    vec = doc2vec(Text)

    d2v_euclidean_model, d2v_euclidean_labels = kmeans_euclidean(vec, 4)

    d2v_pie, d2v_count = pie_chart(d2v_euclidean_labels)
    d2v_pie.savefig('static/images/d2v_pie.jpg')
    d2v_pie.clf()
    wordsd2v = wordfreq(d2v_euclidean_labels)

    d2v_report = clusterReport(vec, d2v_euclidean_labels, "D2V")
    d2v_report = [d2v_report]
    d2v_df = pd.DataFrame(d2v_report,
                   columns =["D2V", "Silhoutte Score", "Calinski Score", "Davies Score"])


    Text['D2V labels'] = d2v_euclidean_labels
    Text = Text.sort_values(by=['D2V labels'])
    Text.to_csv('file_d2v.csv')
    return render_template("d2v.html",clusters = no, d2v_count = [d2v_count.to_html(classes='data')], titles1=d2v_count.columns.values,
        d2v_df = [d2v_df.to_html(classes='data')], titles2=d2v_df.columns.values, wordsd2v = wordsd2v)



# --------------------------------------------FILE DOWNLOADS --------------------



@app.route('/NMF_data',methods = ['POST', 'GET'])
def nmf_data():
    return flask.send_file('file_nmf.csv',
                     mimetype='text/csv',
                     attachment_filename='file_nmf.csv',
                     as_attachment=True)

# @app.route('/NMF',methods = ['POST', 'GET'])
# def nmf_pdf():
#     from flask_wkhtmltopdf import Wkhtmltopdf
#     wkhtmltopdf = Wkhtmltopdf(app)
#     return render_template_to_pdf('nmf.html', download=True, save=False, param='hello')



@app.route('/LSA_data',methods = ['POST', 'GET'])
def lsa_data():
    return flask.send_file('file_lsa.csv',
                     mimetype='text/csv',
                     attachment_filename='file_lsa.csv',
                     as_attachment=True)


@app.route('/W2V_data',methods = ['POST', 'GET'])
def w2v_data():
    return flask.send_file('file_w2v.csv',
                     mimetype='text/csv',
                     attachment_filename='file_w2v.csv',
                     as_attachment=True)

@app.route('/LDA_data',methods = ['POST', 'GET'])
def lda_data():
    return flask.send_file('file_lda.csv',
                     mimetype='text/csv',
                     attachment_filename='file_lda.csv',
                     as_attachment=True)


@app.route('/D2V_data',methods = ['POST', 'GET'])
def d2v_data():
    return flask.send_file('file_d2v.csv',
                     mimetype='text/csv',
                     attachment_filename='file_d2v.csv',
                     as_attachment=True)

if __name__ == '__main__':
        app.run(debug=True)
