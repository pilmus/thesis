
PROTECTED_ATTR=( "level" "h_index_4" )


for G in "${PROTECTED_ATTR[@]}";
do
	python3 trec-fair-ranking-evaluator.py  \
            --groundtruth_file TREC-Competition-eval-sample-with-rel.json  \
            --query_sequence_file TREC-Competition-eval-seq-5-25000.csv \
            --group_annotations_file group_annotations/article-$G.csv \
            --group_definition $G
done
