#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

pytest --cache-clear -v control/tests
