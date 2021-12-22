 # [[Pairwse LtR (2019)]]
 [@bonartFairRankingAcademic] use a pairwise LtR algorithm to predict the order of documents in the re-ranking. The algorithm they use is LambdaMart ([@wuRankingBoostingModel]) and it is used through the Elasticsearch LtR plugin [^1]. The features are listed in the table below. 
 
 
 | level                 | feature        |
 | --------------------- | -------------- |
 | query-document (BM25) | title          |
 |                       | abstract       |
 |                       | entities       |
 |                       | venue          |
 |                       | journal        |
 |                       | author's names |
 | document              | year           |
 |                       | out-citations  |
 |                       | in-citations   |
 | query                 | query length   |

Table: Features used for the system (table copied from [@bonartFairRankingAcademic])


The source code is available online [^2]. 




We reproduced this approach because it had open source code available.



 
 ## Implementation detail
 - Remove `missing: null` from featureset because `null` throws an error.
 
 ## Related
 [[A note on the features for LambdaMART]] - Changes made to the missing value of year.
 
          
[^1]: elasticsearch-learning-to-rank.readthedocs.io
[^2]: https://github.com/irgroup/fair-trec