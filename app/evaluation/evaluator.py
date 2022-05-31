import glob
import os.path
import re

import pandas as pd
from tqdm import tqdm

from app.evaluation.src.y2020.compare_run_means import compare_run_means


from app.evaluation.src.y2020.eval.expeval import expeval, utility
from app.evaluation.src.y2020.eval.trec.json2runfile import json2runfile

from app.post_processing.post_processor import get_postprocessor

import app.evaluation.src.y2019.trec_fair_ranking_evaluator as eval2019
from app.utils.src.utils import valid_file_with_none, valid_path_from_user_input


def evaluate(app_entry):
    runfile = get_postprocessor().outfile
    # valid_runfile = valid_file_with_none(runfile)

    if not runfile:
        runfile = valid_path_from_user_input('runfile to evaluate', 'no default', 'file')

    # while not valid_runfile: #todo: replace with method
    #     print("Enter a runfile to evaluate.")
    #     runfile = str(input("$ "))
    #     valid_runfile = valid_file_with_none(runfile)
    runfile = os.path.basename(runfile)

    year = app_entry.get_argument('year')
    if not year:
        year = get_year_from_user_input()
    year = int(year)

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
    elif year == 2020:  # todo: add preproc config name to outfiles
        jsonruns_dir = (app_entry.get_argument("outdir") or "evaluation/resources/2020/jsonruns")
        trecruns_dir = (app_entry.get_argument("trecruns_dir") or "evaluation/resources/2020/trecruns")

        eval_metric = app_entry.get_argument('eval_metric')
        while not (eval_metric == "EEL" or eval_metric == "EEL_ind"):
            print("Which eval metric do you want to use? ([EEL/EEL_ind])")
            eval_metric = str(input("$ (default: EEL)") or "EEL")
            if not (eval_metric == "EEL" or eval_metric == "EEL_ind"):
                print("Invalid eval_metric:", eval_metric, "Try again.")

        square = False  # todo: make user inputtable

        qrels = app_entry.get_argument("qrels")
        if not qrels:
            qrels = valid_path_from_user_input('qrels file',
                                               'evaluation/resources/2020/qrels/train-DocHLevel-mixed_group-qrels.tsv',
                                               'file')

        tsv_name = f"{os.path.splitext(os.path.basename(runfile))[0]}.tsv"
        trec_format_runfile = os.path.join(trecruns_dir, tsv_name)
        json2runfile(os.path.join(jsonruns_dir, runfile), trec_format_runfile, non_verbose=True)

        outname = f"{os.path.splitext(tsv_name)[0]}_{os.path.basename(os.path.splitext(qrels)[0])}_{eval_metric}.tsv"

        outdir = os.path.join(os.path.dirname(jsonruns_dir), 'eval_results')
        outfile = os.path.join(outdir, outname)
        if eval_metric == "EEL":
            expeval(qrels, trec_format_runfile, outfile,
                    complete=True,
                    groupEvaluation=True,
                    normalize=False,
                    square=square)
        else:
            expeval(qrels, trec_format_runfile, outfile,
                    complete=True,
                    groupEvaluation=False,
                    normalize=False,
                    square=square)
        # delete intermediate file because they take up a lot of space :\
        os.remove(trec_format_runfile)


    else:
        raise ValueError(f"Invalid year: {year}.")


def evaluate_multiple():
    # todo: add 2019 eval path

    jsonruns_dir = "evaluation/resources/2020/jsonruns"
    trecruns_dir = "evaluation/resources/2020/trecruns"
    outdir = "evaluation/resources/2020/eval_results"

    eval_metric = False
    while not (eval_metric == "EEL" or eval_metric == "EEL_ind" or eval_metric == "util"):
        print("Which eval metric do you want to use? ([EEL/EEL_ind/util])")
        eval_metric = str(input("$ (default: EEL)") or "EEL")
        if not (eval_metric == "EEL" or eval_metric == "EEL_ind" or eval_metric == "util"):
            print("Invalid eval_metric:", eval_metric, "Try again.")

    qrels = valid_path_from_user_input('qrels file',
                                       'evaluation/resources/2020/qrels/train-DocHLevel-mixed_group-qrels.tsv', 'file')

    print("These are the available files:")
    allruns = glob.glob(os.path.join(f"{jsonruns_dir}", '*'))
    allruns = [os.path.splitext(os.path.basename(run))[0] for run in allruns]
    for i, file in enumerate(allruns):
        print(f"{i}:", file)

    # runfile = valid_path_from_user_input('runfile to evaluate', 'no default', 'file')
    # runfile = os.path.basename(runfile)

    print(
        "Select files regex pattern or list slice. Note: if you want to select many files running list slice multiple times is faster!")

    filtered_list = []
    while not filtered_list:
        pattern = str(input("$ "))

        listr = re.compile('\[([0-9]*):([0-9]*)\]')
        listm = listr.match(pattern)
        if listm:
            low = int(listm.group(1))
            hi = int(listm.group(2))
            if low <= hi:
                filtered_list = allruns[low:hi]
        else:
            r = re.compile(pattern)
            filtered_list = list(filter(r.match, allruns))
        if not filtered_list:
            print("Empty result list, enter new list or regex pattern.")

    print("Evaluating following runs: ")
    for i, run in enumerate(filtered_list):
        print(i, ": ", run)

    square = False  # todo: write in eval chapter somewhere

    for run in tqdm(filtered_list):

        tsv_name = f"{run}.tsv"
        trec_format_runfile = os.path.join(trecruns_dir, tsv_name)
        json2runfile(os.path.join(jsonruns_dir, f"{run}.json"), trec_format_runfile, non_verbose=True)

        outname = f"{run}_{os.path.basename(os.path.splitext(qrels)[0])}_{eval_metric}.tsv"

        outfile = os.path.join(outdir, outname)
        if eval_metric == "EEL":
            expeval(qrels, trec_format_runfile, outfile,
                    complete=True,
                    groupEvaluation=True,
                    normalize=False,
                    square=square)
        elif eval_metric == "EEL_ind":
            expeval(qrels, trec_format_runfile, outfile,
                    complete=True,
                    groupEvaluation=False,
                    normalize=False,
                    square=square)
        elif eval_metric == "util":
            utility(qrels, trec_format_runfile, outfile)
        # delete intermediate file because they take up a lot of space :\
        os.remove(trec_format_runfile)


def compare_means(app_entry):
    # reranker = app_entry.reranker_name
    year = app_entry.get_argument('year')
    if not year:
        year = get_year_from_user_input()

    if year == 2020:
        # outdir = app_entry.get_argument("outdir")
        # outdir = app_entry.get_argument("outdir")
        # eval_results = os.path.join(os.path.dirname(outdir), 'eval_results')
        eval_results = 'evaluation/resources/2020/eval_results'  # todo: un-hardcode

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


def get_year_from_user_input():
    valid_year = False
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
