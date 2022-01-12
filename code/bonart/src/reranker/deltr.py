import json
import sys

import numpy as np
import pandas as pd
from fairsearchdeltr import Deltr

import src.reranker.model as model


class DeltrWrapper(model.RankerInterface):
    """
    Wrapper arround DELTR
    """

    def __init__(self, featureengineer, protected_feature, gamma, standardize=False):
        super().__init__(featureengineer)
        # setup the DELTR object
        self.protected_feature = protected_feature  # generic name so we can easily swap features around
        self.gamma = gamma
        self.standardize = standardize

        # create the Deltr object
        self.dtr = Deltr("protected", self.gamma, number_of_iterations=1, standardize=self.standardize)
        self.weights = None
        self.mus = None
        self.sigmas = None

    def __first_group(self, group, inputhandler):
        authors = inputhandler.get_authors()
        doc_id = group['doc_id'].iloc[0]
        groups = authors.gid[authors.doc_id == doc_id]
        group['in_first'] = int(any(groups.isin(['1'])))
        return group

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

    def dummy_grouping(self, df):
        df['protected'] = 0
        halfsize = int(len(df) / 2)
        df['protected'][0:halfsize] = 1
        return df

    def __prepare_data(self, inputhandler, has_judgment=True):
        """
        DELTR requires the data to be in a specific order: qid, docid, protected feature, ...
        """
        column_order = ["q_num", "doc_id", "protected",
                        "abstract_score", "authors_score", "entities_score",
                        "inCitations", "journal_score", "outCitations", "title_score",
                        "venue_score", "qlength"]

        features = self.fe.get_feature_mat(inputhandler)

        data = inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id', 'relevance']]

        data = pd.merge(data, features, how='left', on=['qid', 'doc_id'])
        # rename protected feature
        data = data.rename(columns={self.protected_feature: "protected"})

        # todo; dummy solution!
        # data['protected'] = data.apply(lambda row: 1 if row['protected'] > 0 else 0, axis=1)
        data = data.groupby('qid', as_index=False).apply(self.dummy_grouping)

        # data = data.groupby('doc_id', as_index=False).apply(self.__first_group,inputhandler=inputhandler)

        if not has_judgment:
            data = data.drop('relevance', axis=1)
            column_order.append("sid")
            column_order.append("qid")
        else:
            column_order.append("relevance")

        data = data.dropna()  # drop missing values as some doc_ids are not in the corpus

        data = data.reindex(columns=column_order)  # protected variable has to be at third position for DELTR

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
        feature_names = [self.protected_feature, "abstract_score", "authors_score", "entities_score",
                         "inCitations", "journal_score", "outCitations", "title_score",
                         "venue_score", "year", "qlength"]
        model_dict = {}
        model_dict['omega'] = dict(zip(feature_names, self.weights))
        model_dict['mus'] = self.mus
        model_dict['sigmas'] = self.sigmas

        with(open(f'models/deltr_gamma_{self.gamma}.model.json', 'w')) as f: #todo: versioning
            json.dump(model_dict, f)

    def __predict_apply(self, df):
        print(df.columns)
        print(df)

        df_copy = df.copy(deep=True)
        df_copy.q_num_combi = df.sid + "." + df.q_num
        df_copy = df_copy.drop(['sid','q_num'],axis=1)

        predictions = self.dtr.rank(df_copy,has_judgment=False)
        predictions[['sid', 'q_num']] = predictions['q_num'].str.split('.')
        # return predictions
        return df

    def _predict(self, inputhandler):
        """
        requires first column to contain the query ids, second column the document ids
                                    and (optionally) last column to contain initial judgements in descending order
                                    i.e. higher scores are better
        """

        data = self.__prepare_data(inputhandler, has_judgment=False)

        print(data)

        grouped_by_qnum = data.drop(['sid','qid'],axis=1).groupby('q_num')
        pred_df = pd.DataFrame(columns=['doc_id','protected','judgement','q_num','sid'])
        for q_num, df in grouped_by_qnum:
            print(q_num)
            prediction = self.dtr.rank(df,has_judgment = False)
            print(prediction)
            prediction['q_num'] = q_num
            prediction['sid'] = 1
            pred_df = pred_df.append(prediction)




        pred_df['rank'] = pred_df.groupby(['sid', 'q_num'])['judgement'].apply(pd.Series.rank, ascending=False,
                                                                         method='first')

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], pred_df,
                        how='left', on=['sid', 'q_num', 'doc_id'])

        return pred
