#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Must supply timeframe in format YYYY-MM"
    exit 1
fi

DATE=$1
echo $DATE

time python get_trees.py <(cut -f2 ./parsed/classified_comments/sample_comments_${DATE}.tsv) <(xzcat /shared/0/datasets/reddit/post-thread-trees/${DATE}.json.xz | jq -r .id) <(xzcat /shared/0/datasets/reddit/post-thread-trees/${DATE}.json.xz) ${DATE}
