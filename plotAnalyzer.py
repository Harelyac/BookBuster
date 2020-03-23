import os
import pickle
import random

import numpy as np
import goodreads as gr
from goodreads import client
import re
from string import punctuation
import nltk
import wikipedia as wk
import re
import os
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import plotly.express as px

EUCLIDEAN = 2

COSINE = 1

nltk.download('punkt')


def text_to_tokens(text):
    book_punctuation = set(punctuation).union({'‘', '’', '“', '”'}).union('0-9')
    if "<br />" in text:
        text = text.replace("<br />", ' ')
    for char in book_punctuation:
        text = text.replace(char, '')
    tokens = word_tokenize(text.lower())

    stop_words = stopwords.words('english')

    list_of_words = [word for word in tokens if word not in stop_words]
    # stemmer = PorterStemmer()
    # list_of_words = [stemmer.stem(word) for word in list_of_words]
    return list_of_words


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(full_w2v, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec: dict, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    words_in_dict = 0
    avg_vec = np.zeros(embedding_dim)
    normalizing_factor = 1
    bottom_of_harominc_seris = 1
    words_in_sent_counter = 0
    for word in sent:
        words_in_sent_counter += 1
        if type(word_to_vec.get(word, 0)) == int:
            continue
        else:
            words_in_dict += 1

            avg_vec += word_to_vec[word] * normalizing_factor
    if words_in_dict == 0:
        return np.zeros(embedding_dim)
    else:
        return avg_vec / words_in_sent_counter


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def LDA(plots):
    num_words = 50
    num_topics = 1
    dictionary = corpora.Dictionary(plots)
    # dictionary.filter_extremes(no_below=1, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in plots]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, update_every=5,
                          chunksize=100, passes=1)
    # topics = lda.print_topics(num_topics, num_words=num_words)
    # lda = models.LdaModel(corpus, num_topics=1, id2word=dictionary, update_every=5,
    #                       chunksize=10000, passes=100)
    # topics = lda.print_topics(1, num_words=20)

    topics_matrix = lda.show_topics(num_topics=num_topics, formatted=False, num_words=num_words)
    topics_matrix = np.array([topic[1] for topic in topics_matrix])
    topic_words = topics_matrix[:, :, 0]
    return [str(word) for word in topic_words[0]]


def book_embedding(book_disc, word2vec_dict):
    most_promenent_words = LDA(([text_to_tokens(book_disc)]))
    embedding = get_w2v_average(most_promenent_words, word2vec_dict, 300)
    return embedding



import pandas as pd



def find_sim(book1_title, book2_title, book2emd):
    """
    Finding the euclidean similitry between 2 books(books objects are from good reads api) based on their summary.
    :param book1: gr book object
    :param book2: gr book object
    :param word2vec_dict:  trained word2vec embeddings.
    :return: euclidean distance
    """

    return np.linalg.norm(book2emd[book1_title] - book2emd[book2_title])


def calcaulte_cosine(bk1_features_emb, bk2_features_emb):
    norm1 = np.linalg.norm(bk1_features_emb)
    norm2 = np.linalg.norm(bk2_features_emb)
    point_wise_mult = bk1_features_emb * bk2_features_emb
    return point_wise_mult / (norm1 * norm2)


# Usefull dont delete

import pandas as pd


def create_word_voc(despcretion_list):
    set_words = set()
    i = 0
    for desc in despcretion_list:
        book_words = LDA([text_to_tokens(str(desc))])
        set_words = set_words.union(book_words)
        i += 1
        print(i)
    return set_words


def get_org_score(books, chosen_book, word2vec_dict):
    scores = np.zeros(30000)
    # TODO maybe theere should be a range and not the highest since books could be too much (man) sometimes.
    for book, i in zip(books, range(30000)):
        scores[i] = find_sim(chosen_book, book, word2vec_dict, COSINE)
    originality_score = scores.sort()[-10:].mean()  # TODO TRY LESHLEL
    return originality_score


from requests import get

from bs4 import BeautifulSoup


def find_all(string, substring):
    """
    Function: Returning all the index of substring in a string
    Arguments: String and the search string
    Return:Returning a list
    """
    length = len(substring)
    c = 0
    indexes = []
    while c < len(string):
        if string[c:c + length] == substring:
            indexes.append(c)
        c = c + 1
    return indexes


def find_total_amount_of_generes(urls):
    genres_set = {}
    for url in urls:
        get_book_genres(genres_set, url)
    save_pickle(genres_set, "genres_dict.pickle")


import matplotlib.pyplot as plt


def get_book_genres(genres_set, url):
    page = get(url)
    content = BeautifulSoup(page.content, "html.parser")
    keys = '<a class="actionLinkLite bookPageGenreLink" href="/genres/'
    str_cont = str(content)
    genre_index = (find_all(str_cont, keys))
    curr_book_vector = []
    for ind in genre_index:
        ind = ind + len(keys)
        string_that_contain_genre = str_cont[ind:ind + 50]
        current_genr = string_that_contain_genre.split("\"")[0]
        if current_genr in genres_set.keys():
            curr_book_vector.append(genres_set[current_genr])
    return curr_book_vector


def get_books_genres_by_ff(books_url):
    genres = load_pickle("genre2num.pickle")
    all_books_vec = []
    for url in books_url:
        all_books_vec.append(get_book_genres(genres, url))
    return all_books_vec


def dist_from_book(book_title, title2emb, all, Graphs=False):
    dist = []
    for book in (all):
        dist.append(find_sim(book_title, str(book), title2emb))
    dist.sort()
    if Graphs == True:
        x = np.arange(1000)
        y = dist[:1000]
        lst = []
        names = []
        for k in y:
            lst.append(k[0])
            names.append(k[1])
        print(len(x), len(y))
        print(y)
        print(x)
        print(lst)
        fig, ax = plt.subplots()
        lst = np.log(lst) / np.log(50)
        ax.scatter(x, lst)
        for i in range(len(names)):
            names[i] = names[i].split('(')[0]
            print(names[i])
        for i, txt in enumerate(names):
            ax.annotate(txt, (x[i], lst[i]))
        ax.annotate("Harry Potter", (0, 0))
        plt.show()
    return np.array(dist[:15]).mean()


def get_all_books_embeddings():
    all = exctract_data_csv("book-description", True)
    all2 = exctract_data_csv("title", True)
    w2v = load_pickle("word2vecDict.pickle")
    all_books_emb = {}
    i = 0
    for book_disc, title in zip(all, all2):
        i += 1
        print(i)
        all_books_emb[title] = book_embedding(str(book_disc), w2v)
    save_pickle(all_books_emb, "all_books_emb.pickle")


def get_average_orig():
    orig_scores = []
    w2v = load_pickle("/cs/usr/deven14423/Desktop/CoursesExcerises/BookBuster11/pickles/all_books_emb.pickle")
    all = exctract_data_csv("title", True)
    i = 0
    for title in all:
        plt = str(title)
        orig_scores.append(dist_from_book(plt, w2v, all))
        i += 1
        print(i)
    return (np.array(orig_scores).mean())


