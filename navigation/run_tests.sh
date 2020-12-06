#!/bin/bash

export PYTHONPATH=..
export PYTHONDONTWRITEBYTECODE=1

pytest -v tests
