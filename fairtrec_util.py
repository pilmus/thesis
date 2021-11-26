import pandas as pd


def sample_to_df(filename, deduplicate=True):
    df = pd.read_json(filename, lines=True)
    df = df[['documents']]
    df = df.explode('documents')
    df['documents'] = df['documents'].transform(lambda x: x['doc_id'])
    print(f"{filename} contains {len(df)} documents.")
    if deduplicate:
        df = df.drop_duplicates()
        print(f"{filename} contains {len(df)} unique documents.")
    return df
