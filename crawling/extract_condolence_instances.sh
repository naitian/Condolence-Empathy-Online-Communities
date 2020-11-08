#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Must supply timeframe in format YYYY-MM"
    exit 1
fi

DATE=$1
GPU=$2
echo $DATE

mkdir -p parsed/

time xzcat /shared/2/datasets/reddit-dump-all/RC/RC_${DATE}.xz |\
     jq -r '[.created_utc, .link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit] | @tsv' | \
     python3 grab_comments.py $DATE $GPU --sample
