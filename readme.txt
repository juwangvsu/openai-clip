----clip main idea ----------------------
train a network:
	input: text, image
	that generate similar text embedding and image embedding
inference:
	find images that match text description:
		forward network to produce text embedding
		generate all image embedding from valid_df (or entire dataset)
		calculate cosine similarity btw text embedding and the image embeddings
		pick the top 9 matches

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
	modify image_path and caption_path in config.py

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
