import os.path

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
