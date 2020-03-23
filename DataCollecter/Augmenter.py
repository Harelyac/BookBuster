import eventlet
from goodreads import client
gc = client.GoodreadsClient('f9ybxsFDlYxpzJioX7AqQ', 'k3aSeqV6hrCBh8vLTM3f1sPfzfbLfa2P2SazTeUzs')
import pandas as pd
df = pd.read_csv("augmented.csv")
# should be faster

progress = 0

 #28244
for index, row in df.iterrows():
    if 23001 <= index <= 28244:
        # print progress
        progress += 1
        print(str((progress/24000)*100) + "%")
        print(index)

        author_id = str(row.author_url).split("/")[-1].split(".")[0]
        book_id = str(row.url).split("/")[-1].split("-")[0]

        try:
            current_author = gc.author(author_id)
            current_book = gc.book(book_id)
        except:
            continue

        # author related
        try:
            df.at[index, 'gender'] = current_author.gender
        except:
            df.at[index, 'gender'] = None

        try:
            df.at[index, 'hometown'] = current_author.hometown
        except:
            df.at[index, 'hometown'] = None

        try:
            df.at[index, 'books-num'] = len(current_author.books)
        except:
            df.at[index, 'books-num'] = None

        try:
            df.at[index, 'born-at'] = current_author.born_at
        except:
            df.at[index, 'born-at'] = None

        try:
            df.at[index, 'works-count'] = current_author.works_count
        except:
            df.at[index, 'works-count'] = None


        # book related
        try:
            df.at[index, 'publication-date'] = "/".join(current_book.publication_date)
        except:
            df.at[index, 'publication-date'] = None

        try:
            df.at[index, 'num-pages'] = current_book.num_pages
        except:
            df.at[index, 'num-pages'] = None

        try:
            df.at[index, 'popular-shelves'] = str(current_book.popular_shelves)
        except:
            df.at[index, 'popular-shelves'] = None

        try:
            df.at[index, 'rating-dist'] = current_book.rating_dist
        except:
            df.at[index, 'rating-dist'] = None

        try:
            df.at[index, 'publisher'] = current_book.publisher
        except:
            df.at[index, 'publisher'] = None

        with eventlet.Timeout(5):
            try:
                df.at[index, 'book-description'] = current_book.description
            except:
                df.at[index, 'book-description'] = None


df.to_csv("augmented.csv", index=False)

