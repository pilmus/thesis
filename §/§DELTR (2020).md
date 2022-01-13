# [[§DELTR (2020)]]

[[ferraroBalancingExposureRelevance]] build upon the approach by [[bonartFairRankingAcademic]], but instead of LambdaMart they use Deltr ([[zehlikeReducingDisparateExposure2020]]). Deltr is a listwise ranking algorithm based on ListNet ([[caoLearningRankPairwise2007]]).  [[zehlikeReducingDisparateExposure2020]] alter the objective function of ListNet to include a fairness component. This makes Deltr an in-processing method.

❗somethign about how it is a top-1 based listwise method❗

Deltr assumes that all items belong to either of two groups, a protected group and a non-protected group. In context of the Fair Trec task, we could delineate groups for example based on country of origin of authors: we could say that authors from a developing area are protected and authors from mixed or advanced locations are not. The idea is that members of the protected group are historically underprivileged and as such need to be boosted in the rankings to achieve fairness.

The objective function is augmented with an "unfairness" term that measures the disparity in exposure between two groups of items. The definition of exposure used by 
[[zehlikeReducingDisparateExposure2020]] in turn is based on that given by [[singhFairnessExposureRankings2018]]. The unfairness $U$ is defined as

$$❗insert def here❗$$
and the full loss function is 
$$❗insert def here❗$$
The $\gamma$  parameter determines the trade-off between fairness and relevance. 
[[ferraroBalancingExposureRelevance]] trains two models, one with $\gamma=0$ and one with $\gamma=1$ and combines the scores documents get from each model.

## Specific technologies and hyperparameters
We use the implementation of Deltr as created for the FairSearch tool ([[zehlikeFairSearchToolFairness2020]]).  We use the features as described in [[ferraroBalancingExposureRelevance]] described in the table below. These are the same as the features used by [[bonartFairRankingAcademic]] with the exception of the year.


| level                 | feature        |
 | --------------------- | -------------- |
 | query-document (BM25) | title          |
 |                       | abstract       |
 |                       | entities       |
 |                       | venue          |
 |                       | journal        |
 |                       | author's names |
 | document              | number of out-citations  |
 |                       | number of in-citations   |
 | query                 | query length   |
 Table: Features used to train Deltr 

 We used the following hyperparameters: ❗fill in values when they are available❗

 We unified the scores for the $\gamma=0$ and the $\gamma =1$ models by taking the mean of the two scores. 
 ❗is this what Ferraro did? why train two sep models and then weight them again when Deltr has the trade-off in its objective function?❗



 In the FairTrec task the queries are grouped together in training sequences. The idea is that fairness can be amortized across the results in a sequence of queries. However, Deltr is trained per-query. Therefore, during training we treat each query as a separate training point.