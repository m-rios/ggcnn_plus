ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TEST_MODULES := test
PYTHONPATH := .
VENV := .venv
PYTHON := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SITE_PACKAGES = $(VENV)/lib/python2.7/site-packages # This is not crossplatform. Should fix

DEFAULT_PYTHON := python2
VIRTUALENV := virtualenv

SHELL_TYPE := zsh

REQUIREMENTS := -r requirements.txt

DEPENDENCIES := ggcnn simulator core utils

init:
	$(VIRTUALENV) $(VENV)
	$(PIP) install $(REQUIREMENTS)
	cd $(SITE_PACKAGES)
	for dep in $(DEPENDENCIES); do\
		ln -sf $(ROOT_DIR)/$$dep $(SITE_PACKAGES);\
	done
	echo "source $(ROOT_DIR)/environment_variables.sh" >> ~/.$(SHELL_TYPE)rc
cpu: init
	$(PIP) install tensorflow
gpu: init
	$(PIP) install tensorflow-gpu
test:
	$(PYTHON) -m unittest discover

.PHONY: init gpu cpu test
