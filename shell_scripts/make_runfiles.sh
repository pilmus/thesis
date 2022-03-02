cd /mnt/c/Users/maaik/Documents/thesis/evaluation || return

for file in 2020/rawruns/submission_lambdamart_r-training-sequence-full.tsv-rev-*; do
  [ -e "$file" ] || continue
  filename="$(basename -- $file)"
  filename="${filename/.json/""}"
  echo eval2020/trec/json2runfile.py -I "$file" \> "2020/runfiles/$filename.tsv"
  python eval2020/trec/json2runfile.py -I "$file" >"2020/runfiles/$filename.tsv"
done

#./trec/json2runfile.py -I $RUNFILE_JSON > runfile.tsv
