import random
import gensim
import string
from itertools import groupby
import re
from nltk.stem.porter import PorterStemmer

# NOTE: PARAGRAPH == DOCUMENT

# ## Task 1 ## #

# random numbers generator
random.seed(123)

# A word stemmer based on the Porter stemming algorithm
stemmer = PorterStemmer()

# reading file to string and converting everything to lowercase
with open("pg3300.txt", "r", encoding="utf-8") as file:
    fileString = file.read()


# splitting paragraphs by the linebreaks
def paragraph(lines):
    for group_separator, line_iteration in groupby(lines.splitlines(True), key=str.isspace):
        if not group_separator:
            yield ''.join(line_iteration)


# adding each paragraph as an list-item in documents list and filtering unwanted word
def make_paragraphs(file, filter_word):
    paragraph_list = []
    for p in paragraph(file):
        if filter_word.casefold() not in p.casefold():
            paragraph_list.append(p)
    return paragraph_list


# Splitting each document into word-elements (tokenizing)
def tokenize_document(documents):
    tokenized_documents = []
    for d in documents:
        # this will also remove punctuations
        tokenized_documents.append(re.sub("[^\w]", " ", d).split())
    return tokenized_documents


# stemming each word in the documents
def stem_document(document):
    stemmed_document = []
    for d in document:
        words_stemmed = []
        for word in d:
            words_stemmed.append(stemmer.stem(word).lower())
        stemmed_document.append(words_stemmed)
    return stemmed_document


documents = make_paragraphs(fileString, "Gutenberg")
# making a copy of the original documents before doing additional modifications
documents_edited = documents.copy()
documents_edited = tokenize_document(documents_edited)
documents_edited = stem_document(documents_edited)

# testing: removing filtered word, tokenizing and stemming
# for x in documents_edited:
#    print(x)


# ## Task 2 ## #

# will contain "bag of words"
bags = []

# stop words that will be used for further filtering of the documents content
stopString = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,' \
             'cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,' \
             'how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,' \
             'not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,' \
             'their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,' \
             'who,whom,why,will,with,would,yet,you,your'

# string to list
stop_word_list = stopString.split(',')

# Building dictionary with the stemmed and tokenized documents
dictionary = gensim.corpora.Dictionary(documents_edited)


# finding the word-indexes of all the stopwords in the given dictionary
def stop_word_ids(stop_words, dictionary):
    ids = []
    for word in stop_words:
        try:
            ids.append(dictionary.token2id[word])
        except:
            pass
    return ids


# list of the id's
stop_ids = stop_word_ids(stop_word_list, dictionary)

# filter out the stopwords in the dictionary
dictionary.filter_tokens(stop_ids)

# converting document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples
for p in documents_edited:
    bags.append(dictionary.doc2bow(p))


# ## Task 3 ## #

# 3.1
# building TF-IDF model using list of documents from the BoW
tfidf_model = gensim.models.TfidfModel(bags)

# 3.2
# mapping BoW into TF-IDF weights (list of pairs (word-index,word-weights)
tfidf_corpus = tfidf_model[bags]

# 3.3
# matrixSimilarity object
matrix_sim = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# 3.4
# setting number of topics to 100
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[bags]
lsi_matrix = gensim.similarities.MatrixSimilarity(lsi_corpus)

# 3.5
print("[3.5 - First 3 LSI topics]")
# printing first three LSI topics
topics = lsi_model.show_topics(3)
for topic in topics:
    print(topic)


# ## Task 3 ## #

# 4.1
# removing punctuations in the query-list
def remove_punctuations_list(word_list):
    words = []
    for word in word_list:
        w = ""
        for char in word:
            if (string.punctuation + "\n\r\t").__contains__(char):
                if w != "":
                    words.append(w.lower())
                    w = ""
                continue
            w += char
        if w != "":
            words.append(w)
    return words


# stemming the query-list
def stem_list(words):
    for i, word in enumerate(words):
        words[i] = stemmer.stem(word.lower())
    return words


# transforming the query by: splitting, removing punctuations, and stemming
def transformation(query):
    query = query.lower()
    query = query.split()
    query = remove_punctuations_list(query)
    query = stem_list(query)
    return query


# 4.1
# query used for searching
query = "What is the function of money?"
# transforming the query
query = transformation(query)
# converting to BoW representation
query = dictionary.doc2bow(query)

# 4.2
# converting BoW to TF-IDF representation
tfidf_index = tfidf_model[query]
print("\n[4.2 - TF_IDF Weights]")
# printing TF-DF weights of query
for word in tfidf_index:
    word_index = word[0]
    word_weight = word[1]
    print("index", word_index, "| word:", dictionary.get(word_index, word_weight), "| weight:", word_weight)

# 4.3
print("\n[4.3 - Top 3 Relevant Documents", end="")
# similar documents
doc2sim = enumerate(matrix_sim[tfidf_index])
# sorting
top_results = sorted(doc2sim, key=lambda x: x[1], reverse=True)[:3]
# printing top 3 most relevant documents
for result in top_results:
    doc = documents[result[0]]
    doc = doc.split('\n')
    print("\n[Document %d]" % result[0])
    # printing only 5 lines of the document
    for line in range(5):
        print(doc[line])

#4.4
print("\n[4.4.1 - Top 3 Topics with the most Significant Weights]",end="")
# converting query TF-IDF representation into LSI-topics weights
lsi_query = lsi_model[query]
# sorting
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
for topic in enumerate(topics):
    t = topic[1][0]
    print("\n[Topic %d]" % t)
    print(lsi_model.show_topics()[t])

print("\n[4.4.2 - Top 3 Most Relevant Paragraphs]", end="")
# similar documents
lsi_doc2sim = enumerate(lsi_matrix[lsi_query])
# sorting
lsi_documents = sorted(lsi_doc2sim, key=lambda kv: -abs(kv[1]))[:3]
for result in lsi_documents:
    doc = documents[result[0]]
    doc = doc.split('\n')
    print("\n[Document %d]" %result[0])
    for line in range(5):
        print(doc[line])