ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TEST_MODULES := test
PYTHONPATH := .
VENV := .venv
PYTHON := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SITE_PACKAGES = $(VENV)/lib/python2.7/site-packages # This is not crossplatform. Should fix

DEFAULT_PYTHON := /usr/bin/python2
VIRTUALENV := /usr/bin/virtualenv

REQUIREMENTS := -r requirements.txt

DEPENDENCIES := ggcnn simulator core utils

init:
	$(VIRTUALENV) $(VENV)
	$(PIP) install $(REQUIREMENTS)
	for dep in $(DEPENDENCIES); do\
		ln -rfs $$dep $(SITE_PACKAGES); \
	done
cpu: init
	$(PIP) install tensorflow
gpu: init
	$(PIP) install tensorflow-gpu
test:
	$(PYTHON) -m unittest discover

.PHONY: init gpu cpu test
