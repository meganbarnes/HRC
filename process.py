import pandas
import collections
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import math
from scipy.spatial import distance
from scipy import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import random
import scipy
from nltk.corpus import stopwords
import fastcluster
import string
import re

#formatting stop words
header = ['unclassified', 'u.s.', 'department', 'of', 'state', 'case', 'no.', 'f-2015-04841', 'doc', 'no.', 'c05739545', 'date:', '05/13/2015', 'state', 'dept.', '-', 'produced', 'to', 'house', 'select', 'benghazi', 'comm.', 'subject', 'to', 'agreement', 'on', 'sensitive', 'information', '&', 'redactions.', 'no', 'foia', 'waiver.', 'release', 'in', 'full']
footer = ['unclassified', 'u.s.', 'department', 'of', 'state', 'case', 'no.', 'f-2015-04841', 'doc', 'no.', 'c05739545', 'date:', '05/13/2015', 'state', 'dept.', '-', 'produced', 'to', 'house', 'select', 'benghazi', 'comm.', 'subject', 'to', 'agreement', 'on', 'sensitive', 'information', '&', 'redactions.', 'no', 'foia', 'waiver.', 'state-5cb0045247', 'f-2014-20439','c05760209','06/30/2015', '08/31/2015']
more = ['to:', 'from:', 'sent:', 'message', 'original', 'h', 're:', '2009', 'call', '2010', '07/31/2015', 'huma', 'pm', 'cheryl', 'abedin,', 'would', 'mills,', 'fw:', 'cc:', 'may', 'sullivan,', 'jacob', '\xe2\x80\x94', 'said', 'get', '<hdr22@clintonemail.com>', 'new', 'secretary', 'know', 'talk', 'see', 'office', 'w', 'speech', 'meeting', 'also', '<abedinh@state.gov>']

#returns complete vocabulary of dataset
def make_vocab():
    df = pandas.read_csv("data/Emails.csv", sep=',', quotechar='"', quoting=1, keep_default_na=False)
    vocab = []
    docs = []
    stop = stopwords.words('english')
    stop = set(stop).union(set(header)).union(set(footer)).union(set(more))
    for i in xrange(df.shape[0]):
        if i % 1000 == 0:
            print i
        doc = [x.lower() for x in df['RawText'][i].split() if x.lower() not in stop and not (re.match('c[0-9]+', x.lower())) and not (re.match('state-[0-9](.)*', x.lower()))]
        try:
            ind_subj = doc.index('subject:')
        except ValueError:
            pass
        doc = doc[ind_subj+1:]
        vocab.extend(doc)
        docs.append(doc)
    
    x=collections.Counter(vocab)
    ordered = [word for word,count in x.most_common()]
    return ordered, docs



#makes dictionary to keep track of word row/column index in matrix   
def make_index_dic(v_words, vc_words):
    v_dic = {}
    vc_dic = {}
    
    for i in xrange(len(v_words)):
        v_dic[v_words[i]] = i
        
    for j in xrange(len(vc_words)):
        vc_dic[vc_words[j]] = j
        
    return v_dic, vc_dic 



#returns count matrix for corpus
def make_count_matrix(v_words, vc_words, lines):
    ret = np.zeros((len(lines), len(vc_words)))   
    v_dic, vc_dic = make_index_dic(v_words, vc_words)
    
    start = time.time()

    m = 0
    #reads corpus line by line 
    for i in xrange(len(lines)):
        m = m + 1
        if m % 1000 == 0:
            print m
        
        for word in lines[i]:
            ret[i, v_dic[word]] += 1
        
    end = time.time()
    print(end - start)    
    return ret
    
#normalizes counts by document length
def normalize_counts(mat):
    ret = np.zeros(mat.shape)
    for i in xrange(len(mat)):
        s = np.sum(mat[i])
        ret[i] = mat[i] / float(max(s,1))
    
    return ret
    
#returns inverse document frequency of each word in vocab
def idf(mat):
    count_docs = [np.count_nonzero(mat[:,i]) for i in xrange(mat.shape[1])]
    docs = mat.shape[0]
    ret = [(1 + math.log(float(docs) / x)) for x in count_docs]
    
    print count_docs[0], docs, ret[0]
    return ret
   
#returns tfidf normalized version of count matrix 
def tfidf(idf, mat):
    ret = np.zeros(mat.shape)
    for i in xrange(mat.shape[0]):
        ret[i] = np.multiply(mat[i], idf)
    return ret
    

            
    
#returns cluster assignments and centroids of k-means clustering  
def k_means(k, mat, centroids=[]):
    #initialize centroids
    if len(centroids) == 0:
        print 'rand'
        m = 1
        while m <=k:
            ind = random.randint(0, mat.shape[0]-1)
            centroids.append(mat[ind])
            m = m + 1
        
    cont = 1
    count = 0
    dist_list = []
    old_dist_total = 0
    while cont:     #while distortion improving
        print "round", count
        count = count + 1
        cluster_dic = {}
        m = 1
        #initialize cluster dic
        while m <= k:
            cluster_dic[m] = []
            m = m + 1
            
        i = 0
        #update step 1
        dist_total = 0
        while i < mat.shape[0]:
            if i % 1000 == 0:
                print i
            m = 1
            min_dist = sys.float_info.max
            min_cent = 0
            
            while m <=k:
                #try:
                dist = distance.euclidean(centroids[m-1], mat[i])
#                 except ValueError:  #in case centroid is point of comparison
#                     print i
#                     print np.sum(centroids[m-1])
#                     print np.sum(mat[i])
                if dist < min_dist:
                    min_cent = m
                    min_dist = dist
                m = m + 1
            
            dist_total = dist_total + min_dist*min_dist #distortion as sum of squared distance
            cluster_dic[min_cent].append(mat[i])
            i = i + 1
            
        print [len(cluster_dic[key]) for key in cluster_dic.keys()]
	
	    #update step 2
        centroids_new = []
        for centroid in cluster_dic.keys():
            if len(cluster_dic[centroid]) == 0:     #case where cluster is empty terminates
                return (1, [], [], [])
            
            points = np.array(cluster_dic[centroid])
            new = np.mean(points, axis=0)
            centroids_new.append(new)

        print np.array(centroids).shape
    
        if abs(dist_total - old_dist_total) < 0.5:      #threshold for improving distortion
            cont = 0
        else:
            cont = 1
            old_dist_total = dist_total
        centroids = centroids_new
        dist_list.append((count - 1, dist_total))
        print "dist", dist_total
    

    return (0, dist_list, cluster_dic, centroids)
    
    
#returns initial centroids for buckshot + k-means clustering
def buckshot(k, mat):
    size = int((k*mat.shape[0])**.5)
    print size
    samp = np.zeros((size, mat.shape[1]))
    inds = np.random.randint(0, mat.shape[0], size)
    print inds
    
    for i in xrange(size):
        samp[i] = mat[inds[i]]
        
    #agglomerative clusting on sample
    hier = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='euclidean', compute_full_tree=True)
    flat = hier.fit_predict(samp)
    
    centroids = []
    #find centroids
    for j in xrange(k):
        i_s = [i for i, l in enumerate(flat) if l == j]
        print len(i_s)
        points = [samp[m] for m in i_s]
        points = np.array(points)
        cent = np.mean(points, axis=0)
        centroids.append(cent)
    
    return centroids
    
#returns clustering assignments and centroids of bisecting k-means clustering
def bisecting_kmeans(k, mat):
    q = [np.array(mat)]
    cents = [0]
    num_clust = len(q)
    i = 1
    ret_dic = {}
    round = 0
    #continue until number of clusters is reached
    while num_clust < k:
        round += 1
        pair = max(enumerate(q), key = lambda tup: tup[1].shape[0])     #find largest cluster
        to_bisect = pair[1]
        del q[pair[0]]
        del cents[pair[0]]
        min_s = sys.float_info.max
        min_clusters = {}
        min_centroids = []
        p = 0
        while p < 5:        #try 5 different splits
            no, dist_list, cluster_dic, centroids = k_means(2, to_bisect, centroids=[])
            if no == 1:
                continue
            else:
                p += 1
#             s = 0
#             for key in cluster_dic.keys():
#                 Y = distance.pdist(np.array(cluster_dic[key]), 'euclidean')
#                 a = np.sum(Y)/float(Y.size)     #minimize average pairwise distance within clusters
#                 if Y.size == 0:
#                     a = 0
#                 s += a
            s = dist_list[-1][1]
            if s < min_s:
                min_s = s
                min_clusters = cluster_dic
                min_centroids = centroids
        q.extend([np.array(min_clusters[key]) for key in min_clusters.keys()])
        cents.extend(min_centroids)
        print "Q", len(q)
        num_clust = len(q)
        
    return q, cents
            
        
        
    
#produce list of most frequent words in cluster
def eval(vecs, vocab, centroid):
    s = np.zeros(vecs[0].shape)
    for vec in vecs:
        sim = distance.euclidean(centroid, vec)
        if sim == 0:
            sim = 0.01
        s = s + (1/sim)*vec
        
    print np.min(s), np.max(s)
        
    highest = np.argpartition(s, -100)[-100:]
    sorted_highest = highest[np.argsort(np.array(s)[highest])]
    sorted_highest = np.flipud(sorted_highest)
    w = []
    for high in sorted_highest:
        w.append(vocab[high])
    return w
    
    
#find central documents to each cluster
def digest(vecs, vocab, centroid, docs):
    sims = np.zeros(vecs.shape[0])
    for i in xrange(len(vecs)):
        sim = distance.euclidean(centroid, vecs[i])
        sims[i] = sim
        
    print sims.shape
    z = min(5, len(vecs)-1)
    lowest = np.argpartition(sims, z)[:z]
    sorted_lowest = lowest[np.argsort(np.array(sims)[lowest])]
    ret = [string.join(docs[i]) for i in sorted_lowest]
    print 'len', len(ret)
    return ret
    
#print cluster digest
def review(cluster_dic, centroids, vocab, docs, mat):
    for i in xrange(len(centroids)):
        words = eval(cluster_dic[i+1], vocab, centroids[i])
        ds = digest(np.array(cluster_dic[i+1]), vocab, centroids[i], docs)
        print i+1
        print words
        print ds
        
    ret = np.zeros(mat.shape)
    labels = np.zeros(mat.shape[0])
    i = 0
    for key in cluster_dic.keys():
        for point in cluster_dic[key]:
            ret[i] = point
            labels[i] = key
            i += 1
        
    print "SILHOUETTE SCORE", metrics.silhouette_score(ret, labels)
        
    
    
    
    
    
    
    
    
    
    
    
