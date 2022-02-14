#for file in $(find resources/evaluation/2020/rawruns/lambdamart_ferraro -maxdepth 1 -mindepth 1  -newermt '2/08/2022 22:30:00')
for file in resources/evaluation/2020/rawruns/lambdamart_ferraro/*_[12].json
#for file in resources/evaluation/2020/rawruns/deltr_gammas-*
do
#  echo "$file"
  [ -e "$file" ] || continue
  filename="$(basename -- $file)"
  filename="${filename/.json/""}"
#  IFS='.' read -r -a array <<< "$filename"
#  echo "resources/evaluation/2020/runfiles/deltr_gammas/${array[1]}.tsv"
  echo src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$file" \> "resources/evaluation/2020/runfiles/$filename.tsv"
#  echo 'src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$file" > "resources/evaluation/2020/runfiles/deltr_gammas/$filename.tsv"'
  python src/evaluation/twenty_twenty/eval/trec/json2runfile.py -I "$file" > "resources/evaluation/2020/runfiles/$filename.tsv"
done



#./trec/json2runfile.py -I $RUNFILE_JSON > runfile.tsv