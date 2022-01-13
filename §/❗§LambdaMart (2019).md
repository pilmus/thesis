 # [[❗§LambdaMart (2019)]]
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
 |                       | number of out-citations  |
 |                       | number of in-citations   |
 | query                 | query length   |

Table: Features used for the system (table copied from [@bonartFairRankingAcademic])


The source code is available online [^2]. 




We reproduced this approach because it had open source code available.



 
 ## Implementation detail
 - Remove `missing: null` from featureset because `null` throws an error.


In og code: missing: null for year. Is not possible anymore (throws error), so instead: missing 0? [[QUESTION CLAUDIA]] if that is appropriate.



`b = {'aggs':{'miss':{'filter': {'ids':{'values': ll}},'aggs':{'miss':{'missing':{'field':'year'}}}}}}`
--> ` 'aggregations': {'miss': {'doc_count': 3571, 'miss': {'doc_count': 31}}}}`

31 van de training documents hebben geen year 




WE STICK WITH 0 for now then i guess

 
 ## Related
[[¿Questions about approach of Malte Bonart]]

 ## Footnotes
  
          
[^1]: elasticsearch-learning-to-rank.readthedocs.io
[^2]: https://github.com/irgroup/fair-trec