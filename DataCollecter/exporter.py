import csv
from typing import List

from DataCollecter.models import Book


def export_books_to_csv(books: List[Book], filename: str, is_append_to_csv: bool):
    if is_append_to_csv:
        file_param = 'a'
    else:
        file_param = 'w'
    with open(filename, file_param) as books_csv:
        fieldnames = ['title', 'author', 'author_url', 'rating', 'voters_count', 'url']
        csv_writer = csv.DictWriter(books_csv, quoting=csv.QUOTE_ALL, fieldnames=fieldnames)
        csv_writer.writeheader()
        for book in books:
            csv_writer.writerow({
                'title': book.title,
                'author': book.author,
                'author_url': book.author_url,
                'rating': book.rating,
                'voters_count': book.voters_count,
                'url': book.url,
            })
