import glob
import os.path

import pandas as pd

from app.post_processing.post_processor import get_postprocessor

import app.evaluation.src.y2019.trec_fair_ranking_evaluator as eval2019


def evaluate(app_entry):
    runfile = os.path.basename(get_postprocessor().outfile)
    year = int(app_entry.get_argument('year'))

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
    elif year == 2020:
        pass
    else:
        raise ValueError(f"Invalid year: {year}.")


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
            filedf = pd.read_csv(file,sep='\t')
            outdf = outdf.append({'run': filename,
                          'mean_delta_hindex': filedf['unfairness-run'].mean(),
                          'mean_util': filedf['util-run'].mean()},
                         ignore_index=True)

        level_outfiles = glob.glob(os.path.join(eval_results_dir, 'level', globfile))
        for file in level_outfiles:
            filename = os.path.splitext(os.path.basename(file))[0]
            filedf = pd.read_csv(file,sep='\t')

            outdf['mean_delta_level'][outdf['run'] == filename] = filedf['unfairness-run'].mean()

        outdf = outdf.reset_index(drop=True)
        outdf.to_csv(os.path.join(eval_results_dir,"summary",f"{reranker}_{config}_summary.csv"))

    elif year == 2020:
        pass
    else:
        raise ValueError(f"Invalid year: {year}.")
