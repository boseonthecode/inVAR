.PHONY: data features train eval plots demo table all help

help:
	@echo "Targets: data features train eval plots demo table all"

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

demo:
	python -m epl.demo

table:
	python -m epl.evaluate
	python -m epl.models --models logreg --sets PT

all: data features train eval plots
