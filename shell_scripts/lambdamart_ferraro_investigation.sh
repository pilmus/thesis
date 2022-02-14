for corpus in semanticscholar2020; do
#for corpus in semanticscholar2020subset semanticscholar2020; do
  for rev in "--sort-reverse" ""; do
#  for rev in ""; do
      echo $corpus $rev
      python src/bonart/runs/lambdamart_run.py -c $corpus --lambdamart-version ferraro --sequence-train resources/training/2020/training-sequence-10000.tsv -t $rev
  done
done

