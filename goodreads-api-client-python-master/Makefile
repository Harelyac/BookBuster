all: test

clean:
	rm -rf .eggs/ build/ dist/ docs/_build/ htmlcov/ *.egg-info/ .coverage
	-find . -name '__pycache__' -prune -exec rm -rf "{}" \;
	-find . -name '*.pyc' -delete

coverage:
	coverage run setup.py test
	coverage html

coverage-report:
	codeclimate-test-reporter

.PHONY: docs
docs:
	cd docs && make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at docs/_build/html/index.html.\n\033[0m"

install:
	pip install .[docs,publish,test]

lint-py:
	flake8 setup.py goodreads_api_client

lint-rst:
	doc8 README.rst docs/*.rst

lint: lint-py lint-rst

publish: clean
	python setup.py sdist bdist_wheel
	twine upload dist/*

publish-test: clean
	python setup.py sdist bdist_wheel
	twine upload dist/* -r pypitest

test:
	python setup.py test

##################################
# Docker container management
##################################

docker-build:
	docker build -t goodreads-api-client-python .

docker-docs:
	docker run --rm -v `pwd`:/src goodreads-api-client-python make docs

docker-lint:
	docker run --rm goodreads-api-client-python make lint

docker-publish:
	docker run --rm goodreads-api-client-python make publish

docker-run:
	docker run -it --rm goodreads-api-client-python /bin/bash

docker-test:
	docker run --rm goodreads-api-client-python make test