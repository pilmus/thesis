for file in resources/evaluation/2020/rawruns/deltr_gammas/*
do
#  echo "$filename"
  [ -e "$file" ] || continue
  filename="$(basename -- $file)"
#  IFS='.' read -r -a array <<< "$filename"
#  echo "resources/evaluation/2020/runfiles/deltr_gammas/${array[1]}.tsv"
  echo 'src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$file" > "resources/evaluation/2020/runfiles/deltr_gammas/$filename.tsv"'
  python src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$file" > "resources/evaluation/2020/runfiles/deltr_gammas/$filename.tsv"
done



#./trec/json2runfile.py -I $RUNFILE_JSON > runfile.tsv