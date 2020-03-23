from dataclasses import dataclass


@dataclass
class Book:
    title: str
    author: str
    author_url: str
    rating: float
    voters_count: int
    url: str
    isbn: str

    def __eq__(self, other) -> bool:
        return self.title == other.title

    def __hash__(self) -> int:
        return hash(('title', self.title))
