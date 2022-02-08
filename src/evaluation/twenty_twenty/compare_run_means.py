import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='compare the mean difference of different runs')
    parser.add_argument('--eval-output-runs', nargs='*', default=[], type=str, help='run evaluations to compare')
    parser.add_argument('--reference-run', help='evaluation on the uploaded run, used as reference')
    parser.add_argument('--save', help='store the output in this location')

    args = parser.parse_args()

    runs = args.eval_output_runs
    ref_run = args.reference_run

    df_ref = pd.read_csv(ref_run, sep='\t', names=['key', 'qid', 'value'])
    df_ref = df_ref.pivot(index='qid', columns='key', values='value')
    ref_mean = df_ref.difference.mean()

    print("\t".join(['mean', 'meandiff', 'refmean', 'file']))

    comp_dicts = []
    comp_dicts.append(
        {'mean': str(round(ref_mean, 3)),
         'meandiff': str(round(ref_mean - ref_mean, 3)),
         'refmean': str(round(ref_mean, 3)),
         'file': os.path.basename(ref_run)})

    print("\t".join(comp_dicts[-1].values()))

    for run in runs:
        df = pd.read_csv(run, sep='\t', names=['key', 'qid', 'value'])
        df = df.pivot(index='qid', columns='key', values='value')
        df_mean = df.difference.mean()
        comp_dicts.append(
            {'mean': str(round(df_mean, 3)),
             'meandiff': str(round(df.difference.mean() - ref_mean, 3)),
             'refmean': str(round(ref_mean, 3)),
             'file': os.path.basename(run)})
        print("\t".join(comp_dicts[-1].values()))

    if args.save:
        df_comp = pd.DataFrame(comp_dicts)
        df_comp.to_csv(args.save)


if __name__ == '__main__':
    main()
