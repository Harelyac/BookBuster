from typing import List
from typing import Set
import pandas as pd
import click
from bs4 import BeautifulSoup

from DataCollecter.exporter import export_books_to_csv
from DataCollecter.models import Book
from DataCollecter.parser import finilize_book_elem_list
from DataCollecter.parser import get_first_page_content
from DataCollecter.parser import get_page_last
from DataCollecter.parser import init_book_bs_list
from DataCollecter.parser import parse_books


def run(list_url: str, filename_for_export: str, is_append_to_csv: bool):
    page_first_content = get_first_page_content(list_url)
    
    book_elems_list: List[BeautifulSoup] = init_book_bs_list(page_first_content=page_first_content)
    page_last: int = get_page_last(page_first_content)
    
    print(f'loading {page_last} pages...')
    
    book_elems_list = finilize_book_elem_list(
        book_elems_list=book_elems_list,
        list_url=list_url,
        page_last=page_last,
    )
    
    books: Set[Book] = parse_books(book_elems_list)

    print(f'exporting...')
    
    export_books_to_csv(books, filename=filename_for_export, is_append_to_csv=is_append_to_csv)

run("https://www.goodreads.com/list/show/2451.I_Saw_the_Movie_Read_the_Book","booksIntoMovies.csv",True)
