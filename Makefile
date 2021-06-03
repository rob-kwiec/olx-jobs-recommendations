#################################################################################
# GLOBALS                                                                       #
#################################################################################
UNAME := $(shell uname)

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := jobs-research
SHELL=/bin/bash
TESTS_DIR=./tests

VENV = $(PROJECT_DIR)/.venv
PIP = $(VENV)/bin/pip
IPYTHON = $(VENV)/bin/ipython
PYTHON ?= python3.8
VIRTUALENV = $(PYTHON) -m venv

CENV = $(PROJECT_DIR)/.cenv
CONDAROOT=$(HOME)/anaconda3
ifeq ("$(UNAME)", "Darwin")
CONDAROOT=$(HOME)/opt/anaconda3
endif
CONDAPYTHON=$(CONDAROOT)/envs/$(PROJECT_NAME)/bin/python
CONDAPIP=$(CONDAROOT)/envs/$(PROJECT_NAME)/bin/pip


#################################################################################
# virtual environment and dependencies                                          #
#################################################################################

.PHONY: venv
## create virtual environment with requirements
venv: ./.venv/.requirements

.venv:
	$(VIRTUALENV) $(VENV)
	$(PIP) install -U pip setuptools wheel

.venv/.requirements: .venv
	$(PIP) install -r $(PROJECT_DIR)/requirements.txt
	$(PIP) install -r $(PROJECT_DIR)/requirements-dev.txt
	touch $(VENV)/.requirements

## create kernel in venv with requirements
vkernel: ./.venv/.requirements
	$(PIP) install -U jupyter ipywidgets
	$(IPYTHON) kernel install --name "$(PROJECT_NAME)" --user
	touch $(VENV)/.kernel

.PHONY: venv-clean
## clean virtual environment
venv-clean:
	rm -rf $(VENV)

.PHONY: cenv
## create conda environment with requirements
cenv: ./.cenv/.requirements

.cenv:
	conda create --name $(PROJECT_NAME) python=3.8 -y

.cenv/.requirements: .cenv
	$(CONDAPIP) install -r $(PROJECT_DIR)/requirements.txt
	mkdir $(CENV)
	touch $(CENV)/.requirements
	
.PHONY: cenv-clean
## clean conda environment
cenv-clean:
	rm -rf $(CENV)
	conda remove --name $(PROJECT_NAME) --all -y

.PHONY: ckernel
## create kernel in conda with requirements
ckernel: ./.cenv/.requirements
	# $(CONDA) install ipykernel -y
	# python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME)"
	$(CONDAPIP) install -U jupyter ipywidgets
	$(CONDAPYTHON) -c 'import IPython; IPython.terminal.ipapp.launch_new_instance()' kernel install --name "$(PROJECT_NAME)" --user
	touch $(CENV)/.kernel

.PHONY: kernel-clean
## delete kernel and venv/cenv
kernel-clean: venv-clean cenv-clean
	jupyter kernelspec uninstall "$(PROJECT_NAME)" -y

.PHONY: check-env
## checks that local environment is synced with requirements.txt file
check-env: ./.venv/.requirements
	$(VENV)/bin/pip-sync -n

.PHONY: sync-env
## syncs local environment according to the requirements.txt file (note: it will remove dependencies not present in the file)
sync-env: ./.venv/.requirements
	$(VENV)/bin/pip-sync

.PHONY: compile-reqs
## compiles the requirements from requirements.in and updates the requirements.txt file accordingly
compile-reqs: ./.venv/.requirements
	$(VENV)/bin/pip-compile requirements.in
	$(VENV)/bin/pip-compile requirements-dev.in

#################################################################################
# code format / code style                                                      #
#################################################################################

EXCLUDE_DIR = $(PROJECT_DIR)/.venv-runbooks

.PHONY: format-check
## check compliance with code style (via 'black')
format-check: .venv/.requirements
	$(VENV)/bin/black --check $(PROJECT_DIR)/

.PHONY: format-apply
## reformat code for compliance with code style (via 'black')
format-apply: venv
	$(VENV)/bin/black $(PROJECT_DIR)/

#################################################################################
# Local	                                                                        #
#################################################################################

.PHONY: clean
## clean pytest cache
clean:
	rm -rf .pytest_cache

.PHONY: clean-files
## clean tmp local files
clean-files:
	rm -rf *.npz
	rm -rf *.npy
	rm -rf *.pkl
	rm -rf *.csv
	rm -rf *.gz
	rm -rf *.tsv
#################################################################################
# Tests                                                                         #
#################################################################################

.PHONY: test
## run pytest
test: venv
	@PYTHONPATH=$(PYTHONPATH):$(PROJECT_DIR) $(VENV)/bin/pytest $(PROJECT_DIR) -v
    
.PHONY: lint
## run lint
lint: venv
	@PYTHONPATH=$(PYTHONPATH):$(PROJECT_DIR) $(VENV)/bin/pylint --rcfile=setup.cfg $(PROJECT_DIR)/src run.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
	