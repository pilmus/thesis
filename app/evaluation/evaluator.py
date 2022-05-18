import glob
import os.path

import pandas as pd

from app.evaluation.src.y2020.compare_run_means import compare_run_means
from app.evaluation.src.y2020.eval.trec.json2qrels import json_to_group_qrels

from app.evaluation.src.y2020.eval.expeval import expeval
from app.evaluation.src.y2020.eval.trec.json2runfile import json2runfile

from app.post_processing.post_processor import get_postprocessor

import app.evaluation.src.y2019.trec_fair_ranking_evaluator as eval2019
from app.utils.src.utils import valid_file_with_none


def evaluate(app_entry):
    runfile = get_postprocessor().outfile
    valid_runfile = valid_file_with_none(runfile)
    while not valid_runfile: #todo: replace with method
        print("Enter a runfile to evaluate.")
        runfile = str(input("$ "))
        valid_runfile = valid_file_with_none(runfile)
    runfile = os.path.basename(runfile)

    year = get_year(app_entry)

    if year == 2019:
        ref_run = app_entry.get_argument("ref_run")

        outdir = app_entry.get_argument('outdir')
        qseq_file = app_entry.get_argument('qseq_file')
        gt_file = app_entry.get_argument('gt_file')
        group_annot_dir = app_entry.get_argument('group_annot_dir')

        level_annot_file = os.path.join(group_annot_dir, "article-level.csv")
        h_index_4_annot_file = os.path.join(group_annot_dir, "article-h_index_4.csv")

        eval2019.evaluate(qseq_file, gt_file, level_annot_file, "level", outdir, run_files=[runfile, ref_run])
        eval2019.evaluate(qseq_file, gt_file, h_index_4_annot_file, "h_index_4", outdir, run_files=[runfile, ref_run])
    elif year == 2020: #todo: add preproc config name to outfiles
        jsonruns_dir = (app_entry.get_argument("outdir") or "evaluation/resources/2020/jsonruns")
        trecruns_dir = (app_entry.get_argument("trecruns_dir") or "evaluation/resources/2020/trecruns")

        qrels = app_entry.get_argument("qrels")
        valid_qrels = valid_file_with_none(qrels)
        while not valid_qrels:
            print("Which qrels file?")
            qrels = str(input(
                    "$ (default: evaluation/resources/2020/qrels/train-DocHLevel-mixed_group-qrels.tsv)") or "evaluation/resources/2020/qrels/train-DocHLevel-mixed_group-qrels.tsv")
            valid_qrels = valid_file_with_none(qrels)

        # qrels = app_entry.get_argument("qrels")

        tsv_name = f"{os.path.splitext(os.path.basename(runfile))[0]}.tsv"
        trec_format_runfile = os.path.join(trecruns_dir, tsv_name)
        json2runfile(os.path.join(jsonruns_dir, runfile), trec_format_runfile, non_verbose=True)

        outname = f"{os.path.splitext(tsv_name)[0]}_{os.path.basename(os.path.splitext(qrels)[0])}.tsv"

        outdir = os.path.join(os.path.dirname(jsonruns_dir), 'eval_results')
        outfile = os.path.join(outdir, outname)
        expeval(qrels, trec_format_runfile, outfile,
                complete=True,
                groupEvaluation=True,
                normalize=False,
                square=app_entry.get_argument('square'))

    else:
        raise ValueError(f"Invalid year: {year}.")


def compare_means(app_entry):
    # reranker = app_entry.reranker_name
    year = get_year(app_entry)

    if year == 2020:
        # outdir = app_entry.get_argument("outdir")
        # outdir = app_entry.get_argument("outdir")
        # eval_results = os.path.join(os.path.dirname(outdir), 'eval_results')
        eval_results = 'evaluation/resources/2020/eval_results' #todo: un-hardcode

        print("Enter glob pattern to select files.")

        runfiles = []
        if not runfiles:
            globstr = str(input("$ "))
            runfiles = glob.glob(os.path.join(eval_results, globstr))
            if not runfiles:
                print("Glob pattern returned empty list, try again.")
        # globfile = f"{reranker}_{app_entry.basename}*.tsv"

        print("Enter reference run name, if any:")
        refpath = str(input("$ ") or "")

        # ref_run = app_entry.get_argument("ref_run")


        print("Comparing these files:", runfiles)



        if refpath == "":
            refpath = os.path.basename(runfiles[0])

        # runfile = os.path.basename(get_postprocessor().outfile)


        # runfile_result = os.path.join(eval_results, f"{os.path.splitext(runfile)[0]}.tsv")
        refrun_result = os.path.join(eval_results, f"{os.path.splitext(refpath)[0]}.tsv")
        print("Round by how many digits?")
        round_by = int(input("$ (default: 3)") or 3)

        compare_run_means(refrun_result, runfiles, round_by=round_by)
    elif year == 2019:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid year: {year}.")

def get_year(app_entry):
    year = app_entry.get_argument('year')
    valid_year = year == "2019" or year == "2020"
    while not valid_year:
        print("Which year's method? (2019/2020)")
        year = str(input("$ (default: 2020)") or 2020)
        valid_year = year == "2019" or year == "2020"
    year = int(year)
    return year


def summarize(app_entry):
    reranker = app_entry.reranker_name
    config = app_entry.config_name
    year = int(app_entry.get_argument('year'))

    if year == 2019:
        outdir = app_entry.get_argument('outdir')
        eval_results_dir = os.path.join(os.path.dirname(outdir), 'eval_results')

        outdf = pd.DataFrame(columns=['run', 'mean_delta_hindex', 'mean_delta_level', 'mean_util'])

        globfile = f"{reranker}_{config}*.csv"

        h_index_outfiles = glob.glob(os.path.join(eval_results_dir, 'h_index_4', globfile))

        print(h_index_outfiles)

        for file in h_index_outfiles:
            filename = os.path.splitext(os.path.basename(file))[0]
            filedf = pd.read_csv(file, sep='\t')
            outdf = outdf.append({'run': filename,
                                  'mean_delta_hindex': filedf['unfairness-run'].mean(),
                                  'mean_util': filedf['util-run'].mean()},
                                 ignore_index=True)

        level_outfiles = glob.glob(os.path.join(eval_results_dir, 'level', globfile))
        for file in level_outfiles:
            filename = os.path.splitext(os.path.basename(file))[0]
            filedf = pd.read_csv(file, sep='\t')

            outdf['mean_delta_level'][outdf['run'] == filename] = filedf['unfairness-run'].mean()

        outdf = outdf.reset_index(drop=True)
        outdf.to_csv(os.path.join(eval_results_dir, "summary", f"{reranker}_{config}_summary.csv"))

    elif year == 2020:
        pass
    else:
        raise ValueError(f"Invalid year: {year}.")




def kendall_tau(app_entry):
    pass
