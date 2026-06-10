.PHONY: setup test test-layer1 test-layer2 test-layer3 test-layer3-streaming test-layer3-full \
        run-re10 run-re100 run-sweep build-report bundle clean

PYTHON  ?= python
PYTEST  ?= pytest
SCRIPTS  = scripts

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	pip install -e .

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

test-layer1:
	$(PYTEST) tests/test_analytical.py -v

test-layer2:
	$(PYTEST) tests/test_spectral.py -v

test-layer3:
	$(PYTEST) tests/test_encoding.py tests/test_streaming.py tests/test_collision.py tests/test_qlbm_full.py -v

test-layer3-streaming:
	$(PYTEST) tests/test_encoding.py tests/test_streaming.py -v

test-layer3-full:
	$(PYTEST) tests/test_collision.py tests/test_qlbm_full.py -v

coverage:
	$(PYTEST) tests/ --cov=tgv --cov-report=term-missing

# ── Run single cases ──────────────────────────────────────────────────────────
run-re10:
	$(PYTHON) $(SCRIPTS)/run_case.py re=10

run-re100:
	$(PYTHON) $(SCRIPTS)/run_case.py re=100

# ── Full scaling study (all 3 required plots) ─────────────────────────────────
run-sweep:
	$(PYTHON) $(SCRIPTS)/run_sweep.py

# ── Reporting ─────────────────────────────────────────────────────────────────
build-report:
	$(PYTHON) $(SCRIPTS)/build_report.py

# ── Submission bundle ─────────────────────────────────────────────────────────
bundle:
	$(PYTHON) -c "import shutil, os; \
	  shutil.make_archive('airbus-quantum-submission', 'zip', \
	    root_dir=os.getcwd(), \
	    base_dir='.', \
	  )"
	@echo "Created airbus-quantum-submission.zip"

clean:
	rm -rf results/figures/*.png results/figures/*.pdf results/data/*.h5
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
