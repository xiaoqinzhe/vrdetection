#!/bin/bash

python vg_to_imdb.py \
    --image_dir /hdd/visualgenome/images/ \
    --metadata_input /hdd/visualgenome/image_data.json \
    --image_size 1024 \
    --imh5_dir /hdd/xqz/temp_data/scene_graph/ \
    --num_workers 10 \
