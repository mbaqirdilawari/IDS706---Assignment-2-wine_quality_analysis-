# Creates the virtual environment and installs required packages
setup:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt

# Runs the analysis script
run:
	python analysis.py

# Deletes cached files and generated charts for a fresh run
clean:
	rm -rf __pycache__
	rm -f graphs/*.png