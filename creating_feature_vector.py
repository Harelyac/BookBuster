import pickle

from plotAnalyzer import *
from DataGraphsJustifications import *

THRESH_HOLD_ORIG = 0.4292828985483242

hometown_dict = {"NEW YORK": (53, 27 / 53), "PARIS": (18, 13 / 18), "LONDON": (64, 26 / 64), "CHICAGO": (48, 27 / 48),
                 "LOS ANGELES": (35, 12 / 33)}

import numpy as np

MULT_THRESHHOLD_FACTOR = 50

THRESH_HOLD_5_DIST = 0.41677192681451136

THRESH_HOLD_4_DIST = 0.3282388524287777

AUTHOR_IND = 0
POP_IND = 1
INTO_MOVIE_IND = 2
GENDER_IND = 3
HOMETOWN_IND = 4
POP_SHELVES_IND = 5
RATING_DIST_IND = 6
PUBLISHER_IND = 7
DISC_IND = 8
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def map_genre_to_ind(plots):
    map_genre_to_ind = {}
    list_of_genres = load_pickle("genres_meaningful.pickle")
    for j, genre in enumerate(list_of_genres):
        map_genre_to_ind[genre] = j
    print(map_genre_to_ind)
    save_pickle(map_genre_to_ind, "genres2ind.pickle")

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))




def ff_vector_from_genres(genres_list: list):
    feature_v = np.zeros(393)
    genre2num = load_pickle("pickles/genres2ind.pickle")
    for genre in genres_list:
        feature_v[genre2num[genre]] = 1
    return feature_v


def pop_genres_intersection(shelfs):
    pop_genres = list(load_pickle("pickles/genres2ind.pickle").keys())
    if (type(shelfs) == float):
        return []
    shelfs = shelfs.strip('[]').replace('"', '').replace(' ', '').split(',')
    pop_genres_in_book = intersection(pop_genres, shelfs)
    return pop_genres_in_book


np.set_printoptions(precision=3, suppress=3)


def feature_vector_from_book_ind(curr_book_index, book, title2emb, author_dict, publisher_dict):



    f_vec = np.zeros(393 + 2 + 2 + 2 + 1 + 1 + 2 + 1)

    input_pop_genres_to_ff(book, f_vec, curr_book_index)
    input_gender_to_ff(book, curr_book_index, f_vec)
    input_popular(book, curr_book_index, f_vec)
    input_dist_difference(book, curr_book_index, f_vec)
    origanlity_score_input(book, curr_book_index, f_vec, title2emb)
    input_author_feature(book, curr_book_index, f_vec, author_dict)
    input_publisher_feature(book, curr_book_index, f_vec, publisher_dict)
    home_town_input_feature(book, curr_book_index, f_vec)
    return f_vec


def input_author_feature(book, curr_book_ind, f_vec, author_dict):
    tuple_author = author_dict[book["author"][curr_book_ind]]
    total, succ_rate = tuple_author[0], tuple_author[1]
    f_vec[0] = total
    f_vec[1] = ((succ_rate - 0.5) * 2)


def input_publisher_feature(book, curr_book_ind, f_vec, pub_dict):
    if (type(book["publisher"][curr_book_ind]) == float):
        return
    tuple_author = pub_dict[book["publisher"][curr_book_ind]]

    total, succ_rate = tuple_author[0], tuple_author[1]
    f_vec[2] = total
    f_vec[3] = (succ_rate - 0.5) * 2


def home_town_input_feature(book, curr_book_ind, f_vec):
    home_town = book["hometown"][curr_book_ind]
    if type(home_town) == float:
        return
    hometown_string = home_town.split(',')[0].upper()
    total = 0
    succ = 0
    try:
        total, succ = hometown_dict[hometown_string]
    except:
        f_vec[4] = total
        f_vec[5] = succ

    f_vec[4] = total
    f_vec[5] = (succ - 0.5) * 2


def origanlity_score_input(book, curr_book_ind, f_vec, title2emb):
    book_orig = (dist_from_book(book["title"][curr_book_ind], title2emb, book["title"]))
    f_vec[8] = (THRESH_HOLD_ORIG - book_orig) * 10


def input_dist_difference(df, curr_book_ind, f_vec):
    book_dist = df["rating-dist"][curr_book_ind]
    dist_counts = np.zeros(5)
    if type(book_dist) == float:
        return 0
    first = book_dist.split(':')[1:]
    for s, j in zip(first, range(5)):
        rating = s.split('|')
        dist_counts[j] += int(rating[0])
    dist_counts = dist_counts / dist_counts.sum()
    dist_5 = dist_counts[0]
    dist_4 = dist_counts[1]
    threshold_5 = THRESH_HOLD_5_DIST
    threshold_4 = THRESH_HOLD_4_DIST
    dis_from_5 = (dist_5 - threshold_5) * MULT_THRESHHOLD_FACTOR
    dis_from_4 = -(threshold_4 - dist_4) * MULT_THRESHHOLD_FACTOR
    f_vec[6] = dis_from_5
    f_vec[7] = dis_from_4


def input_popular(df, curr_book_ind, f_vec):
    f_vec[9] = df["voters_count"][curr_book_ind]


def input_gender_to_ff(df, curr_book_ind, f_vec):
    gender = df["gender"][curr_book_ind]
    if gender == "male":
        f_vec[10] = 1
    if gender == "female":
        f_vec[11] = 1
    return f_vec


def input_pop_genres_to_ff(df, f_vec, book_ind):
    pop_shelves_intersection = pop_genres_intersection(df["popular-shelves"][book_ind])
    f_v_pop_shelf = ff_vector_from_genres(pop_shelves_intersection)
    f_vec[-393:] = f_v_pop_shelf
    return f_vec


def create_all_features_vectors():
    book_all = exctract_data_csv("rating", combined=True, all_data=True)
    book_ind = exctract_data_csv("Unnamed: 0", combined=True)
    book_in2m = exctract_data_csv("into_movie", combined=True)
    feature_vectors = []
    title2emb = load_pickle("pickles/all_books_emb.pickle")
    author_dict = load_pickle("pickles/authors_dict.pickle")
    publisher_dict = load_pickle("pickles/publisher_dict.pickle")
    i = 0
    for ind in book_ind:
        i += 1
        print(i)
        book_ff = feature_vector_from_book_ind(ind, book_all, title2emb, author_dict, publisher_dict)
        feature_vectors.append((book_ff, book_in2m[ind]))
    print(feature_vectors)
    save_pickle(feature_vectors, "feature_vectors_with_labels.pickle")


create_all_features_vectors()
