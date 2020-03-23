.. _contributing:

Contributor's Guide
===================

goodreads_api_client is under active development, and contributions are more than welcome!

#. See open issues `on Github <https://github.com/mdzhang/goodreads-api-client-python/issues>`_
#. Create a pull request against the `master` branch and tag `@mdzhang` for review
#. Once you get a `:+1:` and all CI checks are passing, go ahead and merge

Updating Tests
--------------

This library uses `VCR <https://github.com/kevin1024/vcrpy>`_ for recording Goodreads API responses to use during tests. When adding new tests, you should add new API response recordings, aka cassettes. To record new cassettes, you'll need

#. a `Goodreads developer key <https://www.goodreads.com/api/keys>`_
#. to put your keys into `GOODREADS_API_KEY` and `GOODREADS_API_SECRET` environment variables, using something like `direnv <https://direnv.net/>`_
#. to blow away old cassettes (if necessary)
#. to re-run `make test` to re-record cassettes

For Oauth tests, you also want to

#. Run `client.authorize()` locally (e.g. from a Python REPL) to generate a `client.json` file
#. to uncomment the test patch blocking triggering the Oauth setup flow in :file:`<goodreads_api_client/tests/resources/conftest.py>`, specifically the `patch_transport.start()` line
# to re-run `make test` to re-record cassettes


And that should be it.

Code Linting
------------

The CI is currently setup to lint all files, but you can also use
`Yelp's pre-commit  <http://pre-commit.com/>`_ to run lint checks as pre-commit hooks
