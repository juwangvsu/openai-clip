
-----------5/30/29 ------------
add some stuff to generate data with isaac sim
see readme in VisionRobot
	docker-compose-isaacsim.yaml
	hospital.usd
test run:
	start docker compose,
	load webrtc client to open isaac ui, and load hospital.usd
	docker exec and and run isaac headless
	docker exec and run rviz2 to see camera
------------- 5/29/24 retest on navy lap ----------------
/data/clipdata
~/Document/openai-clip

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
