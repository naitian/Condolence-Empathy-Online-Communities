#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Must supply timeframe in format YYYY-MM"
    exit 1
fi

DATE=$1
echo $DATE

time python parse_trees.py parsed/classified_comments/comment_trees_${DATE}.tsv <(cut -f4 parsed/classified_comments/sample_comments_${DATE}.tsv) ${DATE}
