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
import matplotlib.pyplot as plt

from plotAnalyzer import load_pickle, save_pickle
import pandas as pd


def exctract_data_csv(column, combined=False, all_data=False):
    b2m = pd.read_csv("/cs/usr/deven14423/Desktop/CoursesExcerises/BookBuster11/DataCollecter/augmented.csv")

    cominbed_df = pd.DataFrame(b2m)

    df_bookNot2m = cominbed_df[:28245]
    df_books2m = cominbed_df[28245:]

    df_books2m = df_books2m[df_books2m['voters_count'] > 100]
    df_bookNot2m = df_bookNot2m[df_bookNot2m['voters_count'] > 100]

    df_books2m = df_books2m[column]
    df_bookNot2m = df_bookNot2m[column]

    if combined:
        cominbed_df = cominbed_df[cominbed_df['voters_count'] > 100]
        if all_data:
            return cominbed_df
        else:
            return cominbed_df[column]

    return df_bookNot2m, df_books2m


def sccater(df, alpha, color):
    df = df[["rating", "voters_count"]]

    df = df.sort_values(by=['voters_count'])
    scatter_plt = plt.scatter(df['voters_count'], df["rating"], c=color, alpha=alpha, label="3")
    plt.ylim([0, 6])
    return scatter_plt


def voters_count_vs_rating(dfb, dfbnm, alpha_btm, alpha_bnm):
    fig = plt.figure(figsize=(10, 15))
    mean_dfb = dfb.mean()
    mean_dfbnm = dfbnm.mean()

    plt.subplot(3, 1, 1)
    a = [0, 1, 2, 3, 4, 5, 5]
    plt.ylabel("Rating")
    plt.xlabel("Votes Count")
    voters_rating_mean = plt.axvline(x=mean_dfb[1], color='darkred', alpha=0.4)

    rating_line = plt.axhline(y=mean_dfb[0], color='grey', alpha=0.4)
    plt.title("1000 random samples Rating vs Voters count , Books that were made into movies.")
    sct_plt = sccater(dfb, alpha_bnm, 'red')

    plt.yticks(a)
    plt.legend((sct_plt, voters_rating_mean, rating_line), (
        "Books that were made into movies", "rating mean : " + str(np.round(mean_dfb[0], 2)),
        "popularity mean : " + str(np.round(mean_dfb[1], 2))))
    plt.subplot(3, 1, 2)

    voters_rating_mean = plt.axvline(x=mean_dfbnm[1], color='c', alpha=0.4)
    rating_line = plt.axhline(y=mean_dfbnm[0], color='grey', alpha=0.4)
    sct_plt = sccater(dfbnm, alpha_btm, 'black')
    plt.ylabel("Rating")
    plt.xlabel("Votes Count")
    plt.title("Rating vs Voters count , Books that weren't made into movies.")
    plt.yticks(a)
    plt.legend((sct_plt, voters_rating_mean, rating_line), (
        "Books that weren't made into movies", "rating mean : " + str(np.round(mean_dfbnm[0], 2)),
        "popularity mean : " + str(np.round(mean_dfbnm[1], 2))))

    plt.subplot(3, 1, 3)
    scatter_plt_1 = sccater(dfb, alpha_bnm, 'red')
    scatter_plt_2 = sccater(dfbnm, alpha_btm, 'black')
    plt.yticks(a)
    plt.ylabel("Rating")
    plt.xlabel("Votes Count")
    plt.title("1000 Random samples from each group on top of each other")
    plt.legend((scatter_plt_1, scatter_plt_2),
               ("Books that were made into movies", "Books that were'nt made into movies"))

    plt.savefig("VotingVsRating.png")
    plt.show()


# voters_count_vs_rating(df_books2m,df_bookNot2m,0.2,0.2)
def urls_into_ids(id_list):
    urls_list = []
    for url in id_list:
        num_uncleaned = url.split("/")[-1]
        if "." in num_uncleaned:
            urls_list.append(num_uncleaned.split('.')[0])
        else:
            urls_list.append(num_uncleaned.split('-')[0])
    return urls_list




def create_dict_freq(shelfs, d):
    for shelf in shelfs:
        if type(shelf) == float:
            continue
        shelf = shelf.strip('[]').replace('"', '').replace(' ', '').split(',')
        for genre in shelf:
            try:
                d[genre] += 1
            except:
                d[genre] = 1
    return d


def init_dict(all_genres):
    init = {}
    for shelf in all_genres:
        if type(shelf) == float:
            continue
        shelf = shelf.strip('[]').replace('"', '').replace(' ', '').split(',')
        for genre in shelf:
            init[genre] = 0
    return init


def create_meaningfl_genres():
    bn2m, b2m = exctract_data_csv("popular-shelves")
    all = exctract_data_csv("popular-shelves", True)
    # print(all)
    b2m_d = (init_dict(all))
    bn2m_d = init_dict(all)
    bn2m = bn2m.sample(1595)
    freq_d_n2m = create_dict_freq(bn2m, bn2m_d)
    freq_d_2m = create_dict_freq(b2m, b2m_d)
    tuples_list = []
    for key in b2m_d.keys():
        if np.abs(freq_d_2m[key] - freq_d_n2m[key]) > 50:
            tuples_list.append(key)
        # tuples_list.append((freq_d_2m[key]-freq_d_n2m[key],key))
    print((tuples_list))
    save_pickle(tuples_list, "genres_meaningful.pickle")






# TODO add in the report what's the definition of Abs genre diff
def genres_diff(diff_thresh_hold):
    ff_b2m = load_pickle("b2m_gen_ff.pickle")
    ff_b2m = np.array(ff_b2m).reshape(1000, 135)
    ff_b2m = ff_b2m[:, :].cumsum(axis=0)[-1, :]
    ff_bn2m = load_pickle("bn2m_gen_ff.pickle")
    ff_bn2m = np.array(ff_bn2m).reshape(1000, 135)
    ff_bn2m = ff_bn2m[:, :].cumsum(axis=0)[-1, :]
    num2genre = load_pickle("num2genre.pickle")
    genres = ff_b2m - ff_bn2m
    abs_genre = np.abs(genres)
    genres_count = np.array(genres)
    only_pos = np.argwhere(genres_count > diff_thresh_hold)
    only_neg = np.argwhere(genres_count < -diff_thresh_hold)
    only_neg = np.abs(only_neg)
    plt.figure(figsize=(10, 8))

    only_pos_names = []
    for ind in only_pos:
        only_pos_names.append(num2genre[ind[0]])

    only_neg_names = []
    for ind in only_neg:
        only_neg_names.append(num2genre[ind[0]])

    blue = (plt.scatter(y=genres_count[only_pos], x=np.arange(0, len(only_pos)), alpha=0.8, s=46))
    red = (plt.scatter(y=np.abs(genres_count[only_neg]), x=np.arange(0, len(only_neg)), c='r', alpha=0.8, s=46))
    l = []
    for name, i, value in zip(only_pos_names, np.arange(0, len(only_pos)), genres_count[only_pos]):
        l.append((name, i, value))

    for name, i, value in zip(only_neg_names, np.arange(0, len(only_neg)), np.abs(genres_count[only_neg])):
        l.append((name, i, value))

    for x in l: plt.annotate(x[0], (x[1], x[2]))

    line = plt.axhline(y=diff_thresh_hold, color='darkred', alpha=0.6)
    plt.annotate("Threshold line with value : " + str(diff_thresh_hold), (11, 150))
    plt.legend((blue, red, line), ("Absolute genre counts  difference from book that were made into movies",
                                   "Genre counts difference from book that weren't made into movies",
                                   "diff threshold : " + str(diff_thresh_hold)), loc='upper left')
    plt.title("Absolute genre popularity difference in books that were made into movies and book that weren't"
              "\n   showing only genre with count difference higher then :  " + str(diff_thresh_hold))
    plt.ylabel("Abs genre counts difference")
    plt.xticks([])

    plt.savefig("matrix_con_+" + str(diff_thresh_hold) + ".png")
    plt.show()


def reduce_genres_by_abs_count(thr):
    ff_b2m = load_pickle("b2m_gen_ff.pickle")
    ff_b2m = np.array(ff_b2m).reshape(1000, 135)
    ff_b2m = ff_b2m[:, :].cumsum(axis=0)[-1, :]
    ff_bn2m = load_pickle("bn2m_gen_ff.pickle")
    ff_bn2m = np.array(ff_bn2m).reshape(1000, 135)
    ff_bn2m = ff_bn2m[:, :].cumsum(axis=0)[-1, :]
    num2genre = load_pickle("num2genre.pickle")
    genres = ff_b2m - ff_bn2m
    abs_genre = np.abs(genres)
    bigger_then_thr = np.argwhere(abs_genre > thr)
    print(bigger_then_thr)

    genres_list = []
    for i in bigger_then_thr:
        genres_list.append(num2genre[i[0]])
    print(genres_list)
    print(len(genres_list))


def num_pages_graph(dfb, dfbnm, alpha_btm, alpha_bnm):
    fig = plt.figure(figsize=(10, 15))
    dfb = dfb[dfb < 2000]
    dfbnm = dfbnm[dfbnm < 2000]
    mean_dfb = dfb.mean()
    mean_dfbnm = dfbnm.mean()

    plt.subplot(3, 1, 1)
    a = [0, 1, 2, 3, 4, 5, 5]
    plt.xlabel("Num pages")
    voters_rating_mean = plt.axvline(x=mean_dfb, color='darkred', alpha=0.4)

    plt.title("1000 random samples of num pages from books that were made into movies")
    sct_plt = book_num_pages_scatter(dfb, alpha_bnm, 'red')

    plt.yticks(a)
    plt.legend((sct_plt, voters_rating_mean), (
        "Books that were made into movies", "rating mean : " + str(np.round(mean_dfb, 2))))
    plt.subplot(3, 1, 2)

    voters_rating_mean = plt.axvline(x=mean_dfbnm, color='c', alpha=0.4)
    sct_plt = book_num_pages_scatter(dfbnm, alpha_btm, 'black')
    plt.ylabel("Rating")
    plt.xlabel("Votes Count")
    plt.title("1000 random samples of num pages from books that were not made into movies")
    plt.yticks(a)
    plt.legend((sct_plt, voters_rating_mean), (
        "Books that weren't made into movies", "num pages mean mean : " + str(np.round(mean_dfbnm, 2))))

    plt.subplot(3, 1, 3)
    scatter_plt_1 = book_num_pages_scatter(dfb, alpha_bnm, 'red')
    scatter_plt_2 = book_num_pages_scatter(dfbnm, alpha_btm, 'black')
    plt.yticks(a)
    plt.ylabel("Rating")
    plt.xlabel("Votes Count")
    plt.title("1000 Random samples from each group on top of each other")
    plt.legend((scatter_plt_1, scatter_plt_2),
               ("Books that were made into movies", "Books that were'nt made into movies"))

    plt.savefig("books_num.png")
    plt.show()


def book_num_pages_scatter(df, alpha, color):
    scatter_plt = plt.scatter(df, np.arange(len(df)), c=color, alpha=alpha, label="3")
    plt.xlim([0, 1500])
    return scatter_plt


def creating_3dscatter_for_authors_succ_vs_published(df_bookNot2m, df_books2m):
    df_bookNot2m = df_bookNot2m['author']
    df_books2m = df_books2m['author']
    made2movie = {}
    ntmade2movie = {}
    all_authors = set()
    for author in df_books2m:
        try:
            made2movie[author] += 1
        except:
            made2movie[author] = 1
        all_authors.add(author)
    for author in df_bookNot2m:
        try:
            ntmade2movie[author] += 1
        except:
            ntmade2movie[author] = 1
        all_authors.add(author)

    final_dict = {}
    for author in all_authors:

        try:
            totalmade2m = made2movie[author]
        except:
            totalmade2m = 0

        try:
            total_ntmade = ntmade2movie[author]
        except:
            total_ntmade = 0

        total_books = totalmade2m + total_ntmade
        final_dict[author] = (total_books, totalmade2m / total_books)
    save_pickle(final_dict, "authors_dict.pickle")

    data = np.zeros((len(final_dict.keys()), 3))
    for info, i in zip(final_dict.values(), range(len(final_dict.keys()))):
        data[i, 0] = i
        data[i, 1] = info[0]
        data[i, 2] = info[1]

    data[:, 1] = np.log(data[:, 1])
    print(data)
    data = pd.DataFrame(data, columns=["Index", "Total books made by author", "Success rate of author"])
    import plotly.express as px
    df = px.data.iris()
    print(data)
    px.scatter_3d(data["Total books made by author"], data["Success rate of author"], data["Index"], opacity=0.6)
    plt.show()
    return df_bookNot2m, df_books2m


def gender(dfntm, dftm):
    dftm = pd.DataFrame(dftm, columns=["gender"])
    dfntm = pd.DataFrame(dfntm, columns=["gender"])

    import plotly.graph_objects as go

    labels = ['Male', 'Female']
    colors = ['blue', 'red']
    values = [(dftm["gender"] == "male").sum(), (dftm["gender"] == "female").sum()]

    # pull is given as a fraction of the pie radius
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0])])
    fig.update_layout(title="Books that were made into movie")
    fig.update_traces(marker=dict(colors=colors))
    fig.show()

    labels = ['Female', 'Male']
    colors = ['red', 'blue']
    values = [(dfntm["gender"] == "female").sum(), (dfntm["gender"] == "male").sum()]

    # pull is given as a fraction of the pie radius
    fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.2, 0])])
    fig1.update_layout(title="Books that weren't made into movie")
    fig1.update_traces(marker=dict(colors=colors))
    fig1.show()


def hometown(dfntm, dftm):
    # take specific coloumn from tm
    dftm = dftm.fillna("")
    dftm = dftm[dftm != ""]
    dftm = dftm.str.split(",", n=1, expand=True)

    # take specific coloumn from ntm
    dfntm = dfntm.fillna("")
    dfntm = dfntm[dfntm != ""]
    dfntm = dfntm.str.split(",", n=1, expand=True)

    # hometown
    df_bookNot2m = random.sample(list(filter(lambda x: x != "", dfntm[0])), 1000)
    df_books2m = random.sample(list(filter(lambda x: x != "", dftm[0])), 1000)

    with open("/cs/usr/harelyac/PycharmProjects/Needle/BookBuster-master/DataCollecter/cities.json") as f:
        cities = json.load(f)
    # get names Geojson
    get_cities = [dict.get("properties").get("NAME") for dict in cities.get("features")]

    # filter data on rows that exist in Geojson
    res = [list for list in dftm.values.tolist() if list[0].upper() in get_cities]

    # sort elements in data using the element index in Geojson
    res.sort(key=lambda x: get_cities.index(x[0].upper()))

    dftm = pd.DataFrame(res, columns=["NAME", "unemp"])

    dftm["NAME"] = dftm["NAME"].str.upper()

    fig = px.choropleth(dftm, geojson=cities, locations='NAME', color='unemp',
                        featureidkey="properties.NAME",
                        color_continuous_scale="Viridis",
                        range_color=(0, 30),
                        labels={'unemp': 'to movie'},
                        )
    fig.update_layout(title="to movie", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    # take only rows which in parsed json names
    res = [list for list in dfntm.values.tolist() if list[0].upper() in get_cities]

    # sort elements in res using the element index in parsed json names
    res.sort(key=lambda x: get_cities.index(x[0].upper()))

    dfntm = pd.DataFrame(res, columns=["NAME", "unemp"])

    dfntm["NAME"] = dfntm["NAME"].str.upper()

    fig = px.choropleth(dfntm, geojson=cities, locations='NAME', color='unemp',
                        featureidkey="properties.NAME",
                        color_continuous_scale="Viridis",
                        range_color=(3, 30),
                        labels={'unemp': 'not to movie'}
                        )
    fig.update_layout(title="not to movie", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def init_data_for_publishers():
    result = exctract_data_csv()
    ntmade2movie = {k: result[0].count(k) for k in set(result[0])}
    made2movie = {k: result[1].count(k) for k in set(result[1])}
    hometown(pd.DataFrame(ntmade2movie.items(), columns=["NAME", "unemp"]),
             pd.DataFrame(made2movie.items(), columns=["NAME", "unemp"]))
    sum = list(ntmade2movie.keys()) + list(made2movie.keys())
    all_publisher = set(sum)
    return made2movie, ntmade2movie, all_publisher


def publisher_plot():
    made2movie, ntmade2movie, all_publisher = init_data_for_publishers()
    final_dict = {}

    for publisher in all_publisher:

        try:
            totalmade2m = made2movie[publisher]
        except:
            totalmade2m = 0

        try:
            total_ntmade = ntmade2movie[publisher]
        except:
            total_ntmade = 0

        total_books = totalmade2m + total_ntmade
        final_dict[publisher] = (total_books, totalmade2m / total_books)

    data = np.zeros((len(final_dict.keys()), 3))
    for info, i in zip(final_dict.values(), range(len(final_dict.keys()))):
        data[i, 0] = i
        data[i, 1] = info[0]
        data[i, 2] = info[1]
    data[:, 1] = np.log(data[:, 1])
    print(data)
    data = pd.DataFrame(data, columns=["Index", "Total books published by publisher", "Success rate of publisher"])
    import plotly.express as px
    df = px.data.iris()
    print(data)
    fig = px.scatter_3d(data, x="Total books published by publisher", y="Index", z="Success rate of publisher",
                        opacity=0.6, color='Success rate of publisher')
    fig.show()


def get_distrubtions():
    a = pd.read_csv("DataCollecter/augmented.csv")
    dont_have_movies = a[:28247]
    has_movies = a[28247:30004]
    dont_have_movies = dont_have_movies[dont_have_movies["voters_count"] > 100]
    has_movies = has_movies[has_movies["voters_count"] > 100]
    print(len(has_movies))
    print(len(dont_have_movies))
    dont_have_movies = dont_have_movies
    dist_has = has_movies["rating-dist"]
    dist = dont_have_movies["rating-dist"]

    dist_counts = np.zeros(5)
    for dist_string in dist:
        if type(dist_string) == float:
            continue
        first = dist_string.split(':')[1:]
        for s, j in zip(first, range(5)):
            rating = s.split('|')
            dist_counts[j] += int(rating[0])
    print(dist_counts)
    dist_counts = dist_counts / dist_counts.sum()
    dist_counts_has_movie = np.zeros(5)
    for dist_string in dist_has:
        if type(dist_string) == float:
            continue
        first = dist_string.split(':')[1:]
        for s, j in zip(first, range(5)):
            rating = s.split('|')
            dist_counts_has_movie[j] += int(rating[0])
    print(dist_counts_has_movie)
    dist_counts_has_movie = dist_counts_has_movie / dist_counts_has_movie.sum()
    print(dist_counts_has_movie)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.bar([5, 4, 3, 2, 1], height=dist_counts_has_movie)
    plt.sca(ax1)
    plt.xticks([1, 2, 3, 4, 5])
    plt.sca(ax2)
    plt.xticks([1, 2, 3, 4, 5])
    plt.sca(ax3)

    plt.xticks([1, 2, 3, 4, 5])
    ax2.bar([5, 4, 3, 2, 1], height=dist_counts)
    ax3.bar([5, 4, 3, 2, 1], height=dist_counts_has_movie - dist_counts)
    print(dist_counts)
    print(dist_counts_has_movie)
    print((dist_counts[0] + dist_counts_has_movie[0]) / 2)
    print((dist_counts[1] + dist_counts_has_movie[1]) / 2)

    import plotly.graph_objects as go
    animals = ['5', '4', '3', '2', '1']

    fig = go.Figure(data=[
        go.Bar(name='Rating distribution of books with movies , 1594 samples', x=animals, y=dist_counts),
        go.Bar(name="Rating distrubtion of books with out movies , 20798 samples", x=animals, y=dist_counts_has_movie),
        go.Bar(name="Rating distribution difference by subtraction ", x=animals, y=-dist_counts + dist_counts_has_movie)
    ])

    fig.update_layout(barmode='group')
    fig.show()
