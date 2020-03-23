goodreads_api_client
====================

A lightweight wrapper around the Goodreads API

.. image:: https://img.shields.io/pypi/v/goodreads-api-client.svg
    :target: https://pypi.python.org/pypi/goodreads-api-client
    :alt: PyPi page link -- version

.. image:: https://travis-ci.org/mdzhang/goodreads-api-client-python.svg?branch=master
    :target: https://travis-ci.org/mdzhang/goodreads-api-client-python

.. image:: https://img.shields.io/pypi/l/goodreads-api-client.svg
    :target: https://pypi.python.org/pypi/goodreads-api-client
    :alt: PyPi page link -- MIT license

.. image:: https://img.shields.io/pypi/pyversions/goodreads-api-client.svg
    :target: https://pypi.python.org/pypi/goodreads-api-client
    :alt: PyPi page link -- Python versions

.. image:: https://codeclimate.com/github/mdzhang/goodreads-api-client-python/badges/gpa.svg
    :target: https://codeclimate.com/github/mdzhang/goodreads-api-client-python
    :alt: Code Climate

.. image:: https://readthedocs.org/projects/goodreads-api-client/badge/?version=latest
    :target: http://goodreads-api-client.readthedocs.io/en/latest/
    :alt: RTD Docs

Installation
------------

.. code-block:: bash

    $ pip install goodreads_api_client

Usage
-----

.. code-block:: python

    >>> import goodreads_api_client as gr
    >>> client = gr.Client(developer_key='<YOUR_DEVELOPER_KEY>')
    >>> book = client.Book.show('1128434')
    >>> keys_wanted = ['id', 'title', 'isbn']
    >>> reduced_book = {k:v for k, v in book.items() if k in keys_wanted}
    >>> reduced_book
    {'id': '1128434', 'title': 'The Last Wish (The Witcher, #1)', 'isbn': '0575077832'}

Resources
---------

* `Goodreads API Docs`_

.. _Goodreads API Docs: https://www.goodreads.com/api/index

Rationale
---------

There are a number of Goodreads API wrapper libraries out there, but most are
either abandoned or the code is some combination of odd, undocumented,
untested, or incomplete in its API coverage.

Contributing
------------

To install locally

.. code-block:: bash

    $ make install

And to test

.. code-block:: bash

    $ make test
