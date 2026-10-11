.PHONY: data features train eval plots all help

help:
	@echo "Targets: data features train eval plots all"

data:
	python -m epl.data

features:
	python -m epl.features

train:
	python -m epl.models

eval:
	python -m epl.evaluate

plots:
	python -m epl.plots

all: data features train eval plots
