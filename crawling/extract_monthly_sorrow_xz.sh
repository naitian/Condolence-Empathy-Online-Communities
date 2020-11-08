#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Must supply timeframe in format YYYY-MM"
    exit 1
fi

DATE=$1
echo $DATE

mkdir -p parsed/

# generates 2 files: parent_ids.csv and submission_links.csv
# takes approximately 75 minutes to run
echo "Getting parent references from /shared-1/datasets/RC/RC_${DATE}.xz"
time xzcat /shared-1/datasets/reddit-dump-all/RC/RC_${DATE}.xz |\
     jq -r '[.link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score] | @tsv' |\
     python get_parent_references.py $DATE

cat ./parsed/parent_ids_$DATE.csv | sort | uniq > ./parsed/unique_parent_ids_$DATE.csv
cat ./parsed/submission_links_$DATE.csv | sort | uniq > ./parsed/unique_submission_links_$DATE.csv

# generates 2 files: parent_comments.tsv and children_comments.tsv
echo "Getting parent and child comments from /shared-1/datasets/RC/RC_${DATE}.xz"
time xzcat /shared-1/datasets/reddit-dump-all/RC/RC_${DATE}.xz |\
     jq -r '[.link_id,.parent_id,.id,.permalink,.body,.gilded,.controversiality,.score,.author,.subreddit] | @tsv' |\
     python get_comments.py $DATE

# generates 1 file: parent_submissions.tsv
# takes approximately 22 minutes to run
echo "Getting parent submissions from /shared-1/datasets/RS/RS_${DATE}.xz"
time xzcat /shared-1/datasets/reddit-dump-all/RS/RS_${DATE}.xz |\
     jq -r '[.id,.permalink,.title,.selftext,.url,.subreddit,.score,.num_comments] | @tsv' |\
     python get_submissions.py $DATE
