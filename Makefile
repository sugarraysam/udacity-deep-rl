export IPYTHONDIR = $(PWD)/.ipython
export PIPENV_VENV_IN_PROJECT=true

.PHONY: pipenv
pipenv:
	@pipenv install --dev
	@pipenv clean

.PHONY: ipython
ipython:
	@pipenv run ipython

.PHONY: help
help:
	@grep '^.*:$$' Makefile
