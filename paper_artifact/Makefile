# parked-paper -- one entry point.
SHELL := /bin/bash
JOBS ?= 1
.PHONY: help data sim figs paper check all clean

help:
	@echo "make data   download the Azure traces (CC-BY, ~1 MB)"
	@echo "make sim    run E1-E10           (~30 min, CPU only, no GPU)"
	@echo "            set JOBS=N to parallelize E8/E9 workload points"
	@echo "make figs   results/ -> figs/"
	@echo "make paper  compile parked.pdf"
	@echo "make check  validate result completeness and run regression tests"
	@echo "make all    everything, from nothing"

data:
	bash data/get_data.sh

sim: data
	cd src && for e in 1 2 3 4 5 6; do python3 run_experiments.py $$e; done
	cd src && python3 exp7_sensitivity.py
	cd src && python3 exp8_high_load.py --jobs $(JOBS)
	cd src && python3 exp9_admission_sensitivity.py --jobs $(JOBS)
	cd src && python3 exp10_queue_tradeoff.py

figs:
	cd src && python3 make_figs.py
	cp figs/*.pdf paper/figs/

paper: figs
	cd paper && pdflatex -interaction=nonstopmode parked.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode parked.tex >/dev/null \
	  && echo "-> paper/parked.pdf"

check:
	python3 src/validate_results.py
	python3 -m pytest -q tests

all: sim paper

clean:
	rm -f results/*.csv results/*.json figs/* paper/*.aux paper/*.log paper/*.out
