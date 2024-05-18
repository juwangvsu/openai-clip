------------- 5/18/24 run notebook locally ----------------
hpzbook: openmmlab

python -m ipykernel install --user --name=openmmlab
jupyter notebook
	then open ipynb file in browser

----------- start guide train step ----------
hpzbook (openmmlab), alien3(voice) cona

./download_data.sh
python dataconv.py

train:
	python main.py --mode train

inference
	python main.py --mode inference --paramfn best.pt
