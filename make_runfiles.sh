for filename in resources/evaluation/2020/rawruns/trec_run.*.json
do
  [ -e "$filename" ] || continue
  IFS='.' read -r -a array <<< "$filename"
#  echo "resources/evaluation/2020/runfiles/trec_run.${array[1]}.tsv"
  python src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$filename" > "resources/evaluation/2020/runfiles/trec_run.${array[1]}.tsv"
done



#./trec/json2runfile.py -I $RUNFILE_JSON > runfile.tsv