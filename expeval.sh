for grouping in basic mix_down mix_up mixed_group nomixed
do
  python src/evaluation/twenty_twenty/eval/expeval.py "resources/evaluation/2020/merged-annotations-groups-$grouping-qrels.tsv" resources/evaluation/2020/runfiles/Deltr-gammas-ferraro.tsv -G -C -U > "resources/evaluation/2020/eval_output/Deltr-gammas-ferraro-$grouping-unnormalized.tsv"
  python src/evaluation/twenty_twenty/eval/expeval.py "resources/evaluation/2020/merged-annotations-groups-$grouping-qrels.tsv" resources/evaluation/2020/runfiles/Deltr-gammas-ferraro.tsv -G -C > "resources/evaluation/2020/eval_output/Deltr-gammas-ferraro-$grouping.tsv"
done
