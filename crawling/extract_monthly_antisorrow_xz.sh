#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Must supply timeframe in format YYYY-MM"
    exit 1
fi

DATE=$1
echo $DATE

mkdir -p parsed/

if [[ "$2" == "--filtered" ]]; then
    echo "Running script on prefetched neg comments"
    time cat ./parsed/neg_children_comments_${DATE}.tsv ./parsed/neg_parent_comments_${DATE}.tsv | python get_non_condolence.py $DATE --filtered
else
    time xzcat /shared-1/datasets/reddit-dump-all/RC/RC_${DATE}.xz |\
         jq -r '[.link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit] | @tsv' |\
         python get_non_condolence.py $DATE
fi
