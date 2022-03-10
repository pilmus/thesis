import pandas as pd
from tqdm import tqdm

from interface.iohandler import IOHandlerKR
from klettirenders.controller import get_doc_to_author_mapping, mus_vs_matrix, author_doc_mapping, naive_controller


def main():
# def test_theta_1_same_as_relevance_based_ranking():
    seq = 'evaluation/2020/TREC-Fair-Ranking-eval-full-seq.tsv'
    q = 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json'
    ioh = IOHandlerKR(seq, q)

    estimated_relevances = pd.read_csv('klettirenders/relevances/Evaluation_rel_scores_model_A.csv')
    qseq_with_relevances = pd.merge(ioh.get_query_seq(), estimated_relevances, on=['qid', 'doc_id', ],
                                    how='left').sort_values(by=['sid', 'q_num']).reset_index(drop=True)

    # set the est relevance of each item that doesn't have an estimated relevance to 0
    qseq_with_relevances = qseq_with_relevances.fillna(0)

    docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()
    doc_to_author_mapping = get_doc_to_author_mapping(docids, 'klettirenders/mappings/evaluation_doc_to_author.json')

    assert len(doc_to_author_mapping) == len(ioh.get_query_seq().doc_id.drop_duplicates())

    outdf = pd.DataFrame(columns=['sid', 'q_num', 'document', 'score', 'rank'])

    Nmin = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().min()
    Nmax = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().max()

    mus, vs = mus_vs_matrix(Nmin, Nmax)

    subdf = qseq_with_relevances[qseq_with_relevances.sid == 0]

    subdocids = subdf.doc_id.drop_duplicates().to_list()
    sub_doc_to_author_mapping = {k: v for k, v in doc_to_author_mapping.items() if k in subdocids}
    sub_author_to_doc_mapping = author_doc_mapping(sub_doc_to_author_mapping)

    rhos_df = subdf[['doc_id', 'est_relevance']].drop_duplicates()
    rhos = dict(zip(rhos_df.doc_id, rhos_df.est_relevance))

    seq_df = naive_controller(rhos, sub_doc_to_author_mapping, sub_author_to_doc_mapping, mus, vs, theta=1,verbose=False)
    seq_df['sid'] = 0

    outdf = outdf.append(seq_df[['sid', 'q_num', 'document', 'score', 'rank']])


if __name__ == '__main__':
    main()
