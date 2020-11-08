#!/bin/bash

# Takes optional date argument in YYYY-MM-DD format

if [ -z "$1" ];
then
    DATE="$(date --date='3 days ago' +'%Y-%m-%d')"
else
    DATE="$1"
fi

echo "Labeling and grabbing replies for" $DATE
/opt/anaconda/bin/python get_condolence_tweets.py $DATE cuda:2 && /opt/anaconda/bin/python get_replies.py $DATE
