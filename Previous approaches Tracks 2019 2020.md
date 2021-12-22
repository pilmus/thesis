# Description of tracks
The FairTrec track was first run in 2019. It was also organized in 2020 and 2021. The approaches from 2019 and 2020 have been published. We now give a short description of the tasks.

FairTrec is concerned with group fairness: providing a fair amount of exposure to groups of documents or document providers.

![[Pasted image 20211201140357.png]]


- Something about the browsing model

## 2019
### Task
In 2019 the task was re-ranking of papers retrieved by an academic search engine. 

### Corpus

### Evaluation metrics
In 2019 there were two evaluation metrics that were used: exposure and utility. Exposure is amortized across responses to multiple queries. This is akin to the approach by [[biegaEquityAttentionAmortizing2018]].
Utility is measured as.... ?

Group definitions were NOT known before hand.
Evaluated on two group definitions.

### Approaches
In 2019, the following runs were submitted:

| Group/Organization                | Paper                                    | Run name        | Utility | Fairness (H) | Fairness (IMF) | Total (H) | Total (IMF) | Approach category | Engine            | Approach                                                                                                                                                                       |
| --------------------------------- | ---------------------------------------- | --------------- | ------- | ------------ | -------------- | --------- | ----------- | ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| University of Glasgow/Naver Labs  | [[mcdonaldUniversityGlasgowTerrier2019]] | uognleDivAAsp   | 0.5612  | 0.0585       | 0.0059         | 1.5027    | 1.5553      | Diversification   | Terrier 5.2       | DPH (Divergence from Randomness) + diversification with xQuaD                                                                                                                  |
| Institute of Computing Technology | [[resources/lit_notes/wangICTTREC2019]]  | first           | 0.5507  | 0.0456       | 0.0428         | 1.5051    | 1.5079      | Greedy            | MongoDB           | Sort by relevance with BERT, greedy swapping for fairness                                                                                                                      |
| Technische Hochschule Köln        | [[bonartFairRankingAcademic]]            | fair_random     | 0.5476  | 0.0405       | 0.0326         | 1.5071    | 1.515       | Random            | ES                | Random shuffle                                                                                                                                                                 |
| University of Glasgow/Naver Labs  | [[mcdonaldUniversityGlasgowTerrier2019]] | uognleDivAJc    | 0.5544  | 0.0449       | 0.0352         | 1.5095    | 1.5192      | Diversification   | Terrier 5.2       | DPH (Divergence from Randomness) + diversification with xQuaD                                                                                                                  |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.00500 | 0.6239  | 0.1112       | 0.0191         | 1.5127    | 1.6048      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.00200 | 0.6230  | 0.1071       | 0.0348         | 1.5159    | 1.5882      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.00100 | 0.6228  | 0.1068       | 0.0347         | 1.516     | 1.5881      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.01000 | 0.6273  | 0.1097       | 0.0198         | 1.5176    | 1.6075      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.00010 | 0.6241  | 0.1059       | 0.0330         | 1.5182    | 1.5911      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Padua               | [[melucciTestingFairnessUsing]]          | QUARTZ-e0.00001 | 0.6247  | 0.1036       | 0.0332         | 1.5211    | 1.5915      | Term weighting    | ES                | BM25 scores on abstract, title, entities + logarithm over distribution of relevant/irrelevant docs                                                                             |
| University of Glasgow/Naver Labs  | [[mcdonaldUniversityGlasgowTerrier2019]] | ugonleSgbrUtil  | 0.6151  | 0.0482       | 0.0649         | 1.5669    | 1.5502      | Greedy            | Terrier 5.2       | uognleMaxUtil + pre-ordering based on previously received exposure + brute force re-ranking based on utility and exposure discrepancy                                          |
| University of Glasgow/Naver Labs  | [[mcdonaldUniversityGlasgowTerrier2019]] | uognleSgbrFair  | 0.6151  | 0.0482       | 0.0649         | 1.5669    | 1.5502      | Greedy            | Terrier 5.2       | uognleMaxUtil + pre-ordering based on previously received exposure + brute force re-ranking based on utility and exposure discrepancy                                          |
| MacEwan University                | X                                        | MacEwanBase     | 0.6194  | 0.0476       | 0.0770         | 1.5718    | 1.5424      | Term weighting    | SOLR (presumably) | "an approach where the final ranking is a weighted merge of search results for different fields; weights are adjusted throughout the sequence" ([[biegaOverviewTREC20192020]]) |
| Technische Hochschule Köln        | [[bonartFairRankingAcademic]]            | fair_LambdaMART | 0.6599  | 0.0855       | 0.0741         | 1.5744    | 1.5858      | Pairwise          | ES                | Lambdamart trained on 10 features, no specific fairness                                                                                                                        |
| University of Glasgow/Naver Labs  | [[mcdonaldUniversityGlasgowTerrier2019]] | uognleMaxUtil   | 0.6741  | 0.0656       | 0.0799         | 1.6085    | 1.5942      | Term weighting    | Terrier 5.2       | DPH (Divergence from Randomness) with query expansion + LM with Dirichlet smoothing on TITLE, no fairness                                                                      |

Of these, we reproduce the results of 

#### Random
[[Random shuffle (2019)]]

#### Pairwise LtR
Also from [@bonartFairRankingAcademic] is a pairwise LtR approach. They use the [LTR plugin for Elasticsearch](https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/) to learn LambdaMART based on the features in table X. Their code is open source and is available [here](). We made a number of small changes to the source code during the replication process, these are recorded in the appendix.

year missing null -> 0 [[QUESTION CLAUDIA]]



Table X contains our reproduced results.

## 2020
### Task
In 2020 the task was either retrieval or re-ranking of papers from an academic search engine.

### Evaluation metric
In 2020 the runs were evaluated with a combined exposure/utility metric. This metric is based on [[diazEvaluatingStochasticRankings2020]].

### Approaches

| Group/Organization       | Paper                                    | Run name        | Disparity | Approach category     | Engine      | Approach                                                                                                                                                                             |
| ------------------------ | ---------------------------------------- | --------------- | --------- | --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Naver Labs               | [[klettiNaverLabsEurope]]                | NLE_META_9_1    | 0.428     | Term weighting        | ❓          | Linear combo BM25/word embedding combined with metadata (recency, citations) with Gradient-Boosted Tree Classifier + controller correcting for too little/too much exposure thus far |
| Naver Labs               | [[klettiNaverLabsEurope]]                | NLE_META_99_1   | 0.429     | Term weighting        | ❓          | Linear combo BM25/word embedding combined with metadata (recency, citations) with Gradient-Boosted Tree Classifier + controller correcting for too little/too much exposure thus far |
| Naver Labs               | [[klettiNaverLabsEurope]]                | NLE_META_PKL    | 0.433     | Plackett-Luce sampler | ❓          | Plackett-Luce sampler                                                                                                                                                                |
| Naver Labs               | [[klettiNaverLabsEurope]]                | NLE_TEXT_9_1    | 0.438     | Controller            | ❓          | Linear combo BM25/word embedding + controller correcting for too little/too much exposure thus far                                                                                   |
| Naver Labs               | [[klettiNaverLabsEurope]]                | NLE_TEXT_99_1   | 0.442     | Controller            | ❓          | Linear combo BM25/word embedding + controller correcting for too little/too much exposure thus far                                                                                   |
| University of Glasgow    | [[mcdonaldUniversityGlasgowTerrier2020]] | UoGTrBComFu     | 0.475     | Data fusion           | Terrier 5.2 | Groups based on citation links, data fusion approach                                                                                                                                 |
| Universitat Pompeu Fabra | [[ferraroBalancingExposureRelevance]]    | LM-rel-groups   | 0.580     | Pairwise              | ES          | Lambdamart + randomization and groups based on collaboration graph                                                                                                                   |
| Universitat Pompeu Fabra | [[ferraroBalancingExposureRelevance]]    | LM-relevance    | 0.601     | Pairwise              | ES          | Lambdamart + randomization                                                                                                                                                           |
| MacEwan University       | [[almquistMacEwanUniversityTREC]]        | MacEwan-base    | 0.722     | Term weighting        | SOLR        | BM25 of abstract and title combined with SMART ann weighting scheme [[resources/lit_notes/shawCombinationMultipleSearches]]                                                          |
| University of Glasgow    | [[mcdonaldUniversityGlasgowTerrier2020]] | UoGTrComRel     | 0.798     | Dissimilarity         | Terrier 5.2 | Groups based on citation links, linear combination of DPH (no COLBERT), representativeness, dissimilarity                                                                            |
| Universitat Pompeu Fabra | [[ferraroBalancingExposureRelevance]]    | LM-relev-year   | 0.811     |                       | ES          | ❓                                                                                                                                                                                   |
| University of Glasgow    | [[mcdonaldUniversityGlasgowTerrier2020]] | UoGTrBComRel    | 0.832     | Dissimilarity         | Terrier 5.2 | Groups based on citation links, include documents based on linear combo relevance, representativeness and dissimilarity                                                              |
| MacEwan University       | [[almquistMacEwanUniversityTREC]]        | MacEwan-norm    | 0.850     | Term weighting        | SOLR        | Normalized BM25 of abstract and title combined with SMART ann weighting scheme [[resources/lit_notes/shawCombinationMultipleSearches]]                                               |
| University of Glasgow    | [[mcdonaldUniversityGlasgowTerrier2020]] | UoGTrBComPro    | 0.851     | Dissimilarity         | Terrier 5.2 | Groups based on citation links, include documents based on linear combo relevance and representativeness time dissimilarity                                                          |
| University of Washington | [[fengUNIVERSITYWASHINGTONTREC]]         | UW_bm25         | 0.875     | Term weighting        | DynamoDB    | BM25 for relevance                                                                                                                                                                   |
| University of Glasgow    | [[mcdonaldUniversityGlasgowTerrier2020]] | UoGTrBRel       | 0.886     | Term weighting        | Terrier 5.2 | DPH and COLBERT + no fairness                                                                                                                                                        |
| University of Washington | [[fengUNIVERSITYWASHINGTONTREC]]         | UW_Kr_r60g20c20 | 0.895     | Cost function         | DynamoDB    | Extract author gender with genderize, location with multi-step process, select documents based on cost function                                                                      |
| University of Maryland   | [[sayedUniversityMarylandTREC]]          | umd_relfair_ltr | 0.907     | Listwise              | ES          | Objective function combined rel + entropy-based fairness, use to train listwise ltr (coordinate ascent)                                                                              |
| University of Washington | [[fengUNIVERSITYWASHINGTONTREC]]         | UW_Kr_r25g25c50 | 0.916     | Cost function         | DynamoDB    | Relevance, gender, country                                                                                                                                                           |
| University of Washington | [[fengUNIVERSITYWASHINGTONTREC]]         | UW_Kr_r0g0c100  | 0.948     | Cost function         | DynamoDB    | Cost function only looks at country                                                                                                                                                  |
| University of Washington | [[fengUNIVERSITYWASHINGTONTREC]]         | UW_Kr_r0g100c0  | 0.999     | Cost function         | DynamoDB    | Cost function only looks at gender                                                                                                                                                   |
| Universitat Pompeu Fabra | [[ferraroBalancingExposureRelevance]]    | LM-rel-year-100 | 1.046     |                       | ES          | ❓                                                                                                                                                                                   |
| Universitat Pompeu Fabra | [[ferraroBalancingExposureRelevance]]    | Deltr-gammas    | 1.067     | Listwise              | ES          | DELTR trained on H-class with $\gamma=0,1$, linear combination of scores                                                                                                             |