#for grouping in mix_down mix_up mixed_group
#do
for file in resources/evaluation/2020/runfiles/deltr_gammas/*
  do
    [ -e "$file" ] || continue
    filename="$(basename -- $file)"
#    IFS='.' read -r -a array <<< "$file"
#    echo src/evaluation/twenty_twenty/eval/expeval.py "resources/evaluation/2020/merged-annotations-groups-mixed_group-qrels.tsv" "$file" -G -C -U "resources/evaluation/2020/eval_output/deltr_gammas/$filename.tsv"
    python src/evaluation/twenty_twenty/eval/expeval.py "resources/evaluation/2020/merged-annotations-groups-mixed_group-qrels.tsv" "$file" -G -C -U -S> "resources/evaluation/2020/eval_output/deltr_gammas/$filename.squared.tsv"
done
#done
