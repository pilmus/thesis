import json

import numpy as np
import pandas as pd
from fairsearchdeltr import Deltr

import bonart.src.reranker.model as model


class DeltrWrapper(model.RankerInterface):
    """
    Wrapper arround DELTR
    """

    COLUMN_ORDER = ["q_num", "doc_id", "protected",
                    "abstract_score", "authors_score", "entities_score",
                    "inCitations", "journal_score", "outCitations", "title_score",
                    "venue_score", "qlength"]

    def __init__(self, featureengineer, protected_feature_mapping, gamma, standardize=False):
        super().__init__(featureengineer)
        # setup the DELTR object
        self.protected_feature_name = protected_feature_mapping['feature_name']
        self.protected_feature_mapping = protected_feature_mapping
        self.gamma = gamma
        self.standardize = standardize

        # create the Deltr object
        self.dtr = Deltr("protected", self.gamma, number_of_iterations=1, standardize=self.standardize)
        self.weights = None
        self.mus = None
        self.sigmas = None

    def load_model(self, model_path):
        with open(model_path) as mp:
            model_dict = json.load(mp)
        self.weights = np.asarray(list(model_dict['omega'].values()))
        self.mus = model_dict['mus']
        self.sigmas = model_dict['sigmas']

        self.dtr._omega = self.weights
        self.dtr._mus = self.mus
        self.dtr._sigmas = self.sigmas

        if self.standardize and not (self.mus and self.sigmas):
            print("You want to standardize but you don't have the required values stored.")

        return self.dtr

    def __protected_feature_grouping(self, df):

        doc_annotations = pd.read_csv('../../resources/2020/doc-annotations.csv')

        # todo: warning if not two groups
        doc_annotations['protected'] = doc_annotations.DocHLevel.map(self.protected_feature_mapping['value_mapping'])


        df['protected'] = doc_annotations['protected']
        return df

    def __prepare_data(self, inputhandler, has_judgment=True, mode='train'):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature, ...
        """

        features = self.fe.get_feature_mat(inputhandler)

        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]

        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])

        data = data.groupby('qid', as_index=False).apply(self.__protected_feature_grouping)

        col_order = self.COLUMN_ORDER
        if has_judgment:
            col_order = self.COLUMN_ORDER + ['relevance']
        else:
            data = data.drop('relevance', axis=1)

        if mode == 'train':
            col_order[0] = 'qid'
        if mode == 'eval':
            data.q_num = data.sid.astype(str) + '.' + data.q_num.astype(str)

        data = data.dropna()  # drop missing values as some doc_ids are not in the corpus and not all docs have
        # Hlevel annotations

        data = data.reindex(columns=col_order)  # protected variable has to be at third position for DELTR

        # data = data.sort_values('relevance', ascending=False)
        data = data.drop_duplicates()
        return data

    def train(self, inputhandler, save=True):
        data = self.__prepare_data(inputhandler)
        self.weights = self.dtr.train(data)
        if self.standardize:
            self.mus = self.dtr._mus
            self.sigmas = self.dtr._sigmas

        if save:
            self.save()

        return self.weights

    def save(self):
        feature_names = self.COLUMN_ORDER[2:-1]
        feature_names[0] = self.protected_feature_name

        model_dict = {}
        model_dict['omega'] = dict(zip(feature_names, self.weights))
        model_dict['mus'] = self.mus
        model_dict['sigmas'] = self.sigmas

        with(open(f'models/deltr_gamma_{self.gamma}_prot_{self.protected_feature_name}.model.json', 'w')) as f:  # todo:
            # versioning
            json.dump(model_dict, f)

    def __predict_apply(self, df):
        print(df.columns)
        print(df)

        df_copy = df.copy(deep=True)
        df_copy.q_num_combi = df.sid + "." + df.q_num
        df_copy = df_copy.drop(['sid', 'q_num'], axis=1)

        predictions = self.dtr.rank(df_copy, has_judgment=False)
        predictions[['sid', 'q_num']] = predictions['q_num'].str.split('.')
        # return predictions
        return df

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self.__prepare_data(inputhandler, has_judgment=False, mode='eval')

        data = data.groupby('q_num').apply(self.dtr.rank, has_judgment=False)
        data = data.reset_index(level=0)

        data['rank'] = data.groupby('q_num')['judgement'].apply(pd.Series.rank, ascending=False,
                                                                method='first')

        data[['sid', 'q_num']] = data['q_num'].str.split('.', expand=True)

        data = data.astype({"sid": int, "q_num": int})

        data = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], data, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        # pred_df['rank'] = pred_df.groupby(['sid', 'q_num'])['judgement'].apply(pd.Series.rank, ascending=False,
        #                                                                        method='first')
        #
        # pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], pred_df,
        #                 how='left', on=['sid', 'q_num', 'doc_id'])

        return data
