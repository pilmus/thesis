To run the evaluator:

1. Add your run files to the fairRuns directory.

2. In the trec-fair-ranking-evaluator.py script find the definition of the run_files variable. 
   Add the names of the runs you want to evaluate. These names needs to match the names of the run files.

3. Run ./run_eval.sh to produce the evaluation results. 
   (You might need to grant the execute permission to the file: chmod +x run_eval.sh ) 

4. The results are written to the eval_results and plots directories.