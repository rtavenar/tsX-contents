PY_SCRIPTS := $(wildcard *.py)
SOL_NOTEBOOKS := $(PY_SCRIPTS:.py=_sol.ipynb)
NOTEBOOKS := $(PY_SCRIPTS:.py=.ipynb)

.PHONY: all
all: $(SOL_NOTEBOOKS) $(NOTEBOOKS)

# Rule: from foo.py -> foo_sol.ipynb (solutions notebook)
%_sol.ipynb: %.py
	jupytext $< --to ipynb -o $@

# Rule: from foo_sol.ipynb -> foo.ipynb (student notebook, solutions stripped)
%.ipynb: %_sol.ipynb
	jupyter nbconvert $< --to ipynb \
    	--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_cell_tags solution \
		--output $@

