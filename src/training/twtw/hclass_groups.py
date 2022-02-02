import pandas as pd

paper_authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.paper_authors.csv'
authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.authors.csv'

paper_authors = pd.read_csv(paper_authors_file, dtype={'paper_sha': str, 'corpus_author_id': str, 'position': int})
authors = pd.read_csv(authors_file, dtype={'corpus_author_id': str, 'h_class': str})


paper_author_hclass = pd.merge(paper_authors, authors, how='left', on='corpus_author_id')


def merge_hclass(df):
    h_classes = df.h_class.to_list()
    if 'L' in h_classes:
        df['group'] = '0'
    else:
        df['group'] = '1'
    return df


paper_author_hclass = paper_author_hclass.groupby('paper_sha').apply(merge_hclass)
paper_author_hclass = paper_author_hclass[['paper_sha', 'group']]
paper_author_hclass = paper_author_hclass.drop_duplicates()
paper_author_hclass = paper_author_hclass.rename({'paper_sha':'doc_id'},axis=1)
paper_author_hclass.to_csv('resources/training/2020/doc-annotations-hclass-groups.csv', index=False)
