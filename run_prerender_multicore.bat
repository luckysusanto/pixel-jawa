@echo off

set DATASET_FILE=scripts\to_render\ban_combined.csv
set RENDERER_PATH=configs\renderers\bali_renderer

python scripts\data\prerendering\prerender_wikipedia_multiprocessing.py ^
  --renderer_name_or_path=%RENDERER_PATH% ^
  --data_path=%DATASET_FILE% ^
  --chunk_size=200000 ^
  --repo_id="Exqrch/pixel-prerender-bali-article" ^
  --split="train" ^
  --auth_token="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" ^
  --num_workers 12
