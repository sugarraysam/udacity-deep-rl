export IPYTHONDIR = $(PWD)/.ipython
export PIPENV_VENV_IN_PROJECT=true

.PHONY: pipenv
pipenv:
	@pipenv install --dev

.PHONY: ipython
ipython:
	@pipenv run ipython

.PHONY:
jupyter:
	@pipenv run jupyter notebook

.PHONY: help
help:
	@grep '^.*:$$' Makefile
