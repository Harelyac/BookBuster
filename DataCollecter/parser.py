import re
from typing import List
from typing import Set

import grequests
from bs4 import BeautifulSoup
from requests import Request
from tqdm import tqdm

from DataCollecter.models import Book


class FeedbackCounter:
    counter: int = 0

    # noinspection PyUnusedLocal
    def feedback(self, request: Request, **kwargs) -> Request:
        self.counter += 1
        print(f'page #{self.counter} fetched...')
        return request


def get_first_page_content(list_url: str) -> str:
    request_async = [grequests.get(list_url)]
    page_first_http_response = grequests.map(request_async)[0]
    return page_first_http_response.text


def init_book_bs_list(page_first_content: str) -> List[BeautifulSoup]:
    soup = BeautifulSoup(page_first_content, 'html.parser')
    book_list_bs: List[BeautifulSoup] = soup.select('tr[itemtype="http://schema.org/Book"]')
    return book_list_bs


def get_page_last(page_first_content: str) -> int:
    soup = BeautifulSoup(page_first_content, 'html.parser')
    page_last_elem = soup.select('div.pagination a:not([rel])')[-1]
    page_last: int = int(page_last_elem.getText())
    return page_last


def finilize_book_elem_list(book_elems_list: List[BeautifulSoup], list_url: str, page_last: int) -> List[BeautifulSoup]:
    pages_urls: List[str] = []
    for page in range(2, page_last + 1):
        pages_urls.append(f'{list_url}?page={page}')

    counter = FeedbackCounter()
    
    requests_list = (grequests.get(url, callback=counter.feedback) for url in pages_urls)
    pages_http_responses = grequests.map(requests_list)
    print('fetching is done')
    for page_response in tqdm(pages_http_responses, desc=f'parsing pages'):
        soup = BeautifulSoup(page_response.text, 'html.parser')
        page_book_list: List[BeautifulSoup] = soup.select('tr[itemtype="http://schema.org/Book"]')
        book_elems_list += page_book_list
    del pages_http_responses
    return book_elems_list


def parse_books(book_elems_list: List[BeautifulSoup]) -> Set[Book]:
    books: List[Book] = []
    for book_elem in tqdm(book_elems_list, desc=f'parsing books'):
        rating_str_full = book_elem.select_one('.minirating').getText()
        pattern = re.compile('(?P<rating>[0-9]+\.[0-9]+) avg rating — (?P<voters_count>(\d+)+?,?(\d+)?)')
        match = pattern.search(rating_str_full)
        rating_str = match.group('rating')
        voters_count_str = match.group('voters_count').replace(',', '')
        base_url = 'https://goodreads.com'
        books.append(
            Book(
                title=book_elem.select_one('a.bookTitle').getText().strip(),
                author=book_elem.select_one('a.authorName span').getText(),
                author_url=book_elem.select_one('a.authorName').attrs['href'],
                url=base_url + book_elem.select_one('a.bookTitle[itemprop="url"]').attrs['href'],
                rating=float(rating_str),
                voters_count=int(voters_count_str),
                isbn=book_elem.select_one("ISBN")
            )
        )
    books_without_duplicates = set(books)
    return books_without_duplicates
