for grouping in mix_down mix_up mixed_group
do
  for filename in resources/evaluation/2020/runfiles/trec_run.*.tsv
  do
    [ -e "$filename" ] || continue
    IFS='.' read -r -a array <<< "$filename"
    python src/evaluation/twenty_twenty/eval/expeval.py "resources/evaluation/2020/merged-annotations-groups-$grouping-qrels.tsv" "$filename" -G -C -U > "resources/evaluation/2020/eval_output/trec_run.${array[1]}-$grouping-unnormalized.tsv" -G -C -U
  done
done
