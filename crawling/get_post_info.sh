DATE=$1

echo $DATE
cat ./parsed/classified_comments/comment_trees_${DATE}.tsv | tqdm --bytes | jq -r '[.id,.author,.is_self,.created_utc,.score] | @tsv' > ./parsed/classified_comments/post_metadata_${DATE}.tsv    
