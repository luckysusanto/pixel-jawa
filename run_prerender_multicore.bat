@echo off

set DATASET_FILE=scripts\data\prerendering\article_merged_wordlevel-jawa.csv
set RENDERER_PATH=configs\renderers\java_renderer

python scripts\data\prerendering\prerender_wikipedia_multiprocessing.py ^
  --renderer_name_or_path=%RENDERER_PATH% ^
  --data_path=%DATASET_FILE% ^
  --chunk_size=150000 ^
  --repo_id="Exqrch/pixel-dataset-prerender-java-wikipedia" ^
  --split="train" ^
  --auth_token="ADD TOKEN" ^
  --num_workers 12
