init:
	virtualenv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	echo $(pwd):'$PYTHONPATH' >> .venv/bin/activate

initcpu: init
	pip install tensorflow

initgpu: init
	pip install tensorflow-gpu

