for corpus in semanticscholar2020 semanticscholar2020subset; do
  for grouping in all_low all_high majorityL majorityH; do
#  for grouping in majorityL majorityH; do
    for alpha in 0.0 0.25 0.50 0.75 1.0; do
      echo $corpus $grouping $alpha
      python src/bonart/runs/deltr_run.py -c $corpus --training-group-file "resources/training/2020/doc-annotations-hclass-groups-$grouping.csv" -r $alpha -t
    done
  done
done
