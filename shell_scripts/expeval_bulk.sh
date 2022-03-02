cd /mnt/c/Users/maaik/Documents/thesis/evaluation || return


for file in 2020/runfiles/submission_lambdamart_r-training-sequence-full.tsv-rev-*
  do
    [ -e "$file" ] || continue
    filename="$(basename -- $file)"
    filename="${filename/.tsv/""}"
    echo python 2020/eval/expeval.py 2020/qrels/merged-annotations-groups-mixed_group-qrels.tsv "$file" -G -C -U -S \> 2020/eval_output/"$filename"_s.tsv
    python 2020/eval/expeval.py "2020/qrels/merged-annotations-groups-mixed_group-qrels.tsv" "$file" -G -C -U -S > 2020/eval_output/"$filename"_s.tsv
    python 2020/eval/expeval.py "2020/qrels/merged-annotations-groups-mixed_group-qrels.tsv" "$file" -G -C -U > 2020/eval_output/"$filename".tsv
done
#done
