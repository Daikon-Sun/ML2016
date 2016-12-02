from __future__ import print_function
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import csv
import sys
import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer

STOPWORDS = frozenset(['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'fifty', 'four', 'not', 'own', 'through', 'yourselves', 'go', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'list',
    'during', 'thus', 'now', 'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above',
    'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'line', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part',
    'everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'your', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter',
    'meanwhile', 'use', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once', 'using', 'does', 'did', 'didn', 'really', 'kg', 'regarding',
    'unless', 'fify', 'say', 'km', 'used', 'various', 'just', 'quite', 'doing', 'don', 'doesn', 'make', 'file', 'creat', 'function', 'page', 'way', 'error', 'type', 'doe', 'custom', 'queri', 'data', 'work' , 'set', 'whi', 'applic', 'chang', 'add', 'multipl', 'best', 'code', 'server', 'user', 'view', 'differ', 'tabl', 'run', 'class', '2'])

num_clusters = 20

stemmer = SnowballStemmer("english")
data = open(sys.argv[1]+'title_StackOverflow.txt', 'r').read().splitlines()
data = [ re.sub("\.+\s", " ", datum) for datum in data ]
data = [ re.sub("\.+$", " ", datum) for datum in data ]
data = [ re.sub("[^\w]", " ", datum) for datum in data ]
data = [ [ stemmer.stem(word) for word in datum.lower().split() ]\
        for datum in data ]
data = [ [ word for word in datum if word not in STOPWORDS ] for datum in data ]

vectorizer = TfidfVectorizer(min_df=8, stop_words=STOPWORDS,\
        sublinear_tf=True, binary=True, analyzer=lambda x: x)
X = vectorizer.fit_transform(data)

print(X.shape)

print('done vectorize')
svd = TruncatedSVD(19)#, algorithm='arpack')
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

X = cosine_similarity(X)

print('done')

predict = []
with open('data/check_index.csv', 'r') as csvfile:
    csvf = csv.reader(csvfile)
    next(csvf)
    for row in csvf:
        idx1 = int(row[1])
        idx2 = int(row[2])
        if X[idx1, idx2] > 0.855 or idx1 == idx2:
            predict.append(1)
        else:
            predict.append(0)

with open(sys.argv[2], 'w') as csvfile:
    csvf = csv.writer(csvfile)
    csvf.writerow(['ID', 'Ans'])
    for i in range(len(predict)):
        csvf.writerow([i, predict[i]])
