# 2020
## Results
| run             | code?                                     | $\Delta_{\mathcal{G}}$ | paper                                                     |
| --------------- | ----------------------------------------- | ---------------------- | --------------------------------------------------------- |
| NLE_META_9_1    | ❌                                        | 0.428                  | [[klettiNaverLabsEurope]]                                 |
| NLE_META_99_1   | ❌                                        | 0.429                  | [[klettiNaverLabsEurope]]                                 |
| NLE_META_PKL    | ❌                                        | 0.433                  | [[klettiNaverLabsEurope]]                                 |
| NLE_TEXT_9_1    | ❌                                        | 0.438                  | [[klettiNaverLabsEurope]]                                 |
| NLE_TEXT_99_1   | ❌                                        | 0.442                  | [[klettiNaverLabsEurope]]                                 |
| UoGTrBComFu     | ❌                                        | 0.475                  | [[mcdonaldUniversityGlasgowTerrier2020]]                  |
| LM-rel-groups   | [✔](https://github.com/irgroup/fair-trec) | 0.580                  | [[resources/lit_notes/ferraroBalancingExposureRelevance]] |
| LM-relevance    | [✔](https://github.com/irgroup/fair-trec) | 0.601                  | [[resources/lit_notes/ferraroBalancingExposureRelevance]] |
| MacEwan-base    | ❌                                        | 0.722                  | [[almquistMacEwanUniversityTREC]]                         |
| UoGTrComRel     | ❌                                        | 0.798                  | [[mcdonaldUniversityGlasgowTerrier2020]]                  |
| LM-relev-year   | [✔](https://github.com/irgroup/fair-trec) | 0.811                  | [[resources/lit_notes/ferraroBalancingExposureRelevance]] |
| UoGTrBComRel    | ❌                                        | 0.832                  | [[mcdonaldUniversityGlasgowTerrier2020]]                  |
| MacEwan-norm    | ❌                                        | 0.850                  | [[almquistMacEwanUniversityTREC]]                         |
| UoGTrBComPro    | ❌                                        | 0.851                  | [[mcdonaldUniversityGlasgowTerrier2020]]                  |
| UW_bm25         | ❌                                        | 0.875                  | [[fengUNIVERSITYWASHINGTONTREC]]                          |
| UoGTrBRel       | ❌                                        | 0.886                  | [[mcdonaldUniversityGlasgowTerrier2020]]                  |
| UW_Kr_r60g20c20 | ❌                                        | 0.895                  | [[fengUNIVERSITYWASHINGTONTREC]]                          |
| umd_relfair_ltr | ❌                                        | 0.907                  | [[sayedUniversityMarylandTREC]]                           |
| UW_Kr_r25g25c50 | ❌                                        | 0.916                  | [[fengUNIVERSITYWASHINGTONTREC]]                          |
| UW_Kr_r0g0c100  | ❌                                        | 0.948                  | [[fengUNIVERSITYWASHINGTONTREC]]                          |
| UW_Kr_r0g100c0  | ❌                                        | 0.999                  | [[fengUNIVERSITYWASHINGTONTREC]]                          |
| LM-rel-year-100 | [✔](https://github.com/irgroup/fair-trec) | 1.046                  | [[resources/lit_notes/ferraroBalancingExposureRelevance]] |
| Deltr-gammas    | [✔](https://github.com/irgroup/fair-trec) | 1.067                  | [[resources/lit_notes/ferraroBalancingExposureRelevance]] |

# 2019
## IMF level with 2 groups
| run             | utility | unfairness | paper                                                        | Approach | code                                   |
| --------------- | ------- | ---------- | ------------------------------------------------------------ | -------- | -------------------------------------- |
| uognleDivAAsp   | 0.5612  | 0.0059     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| QUARTZ-e0.00500 | 0.6239  | 0.0191     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.01000 | 0.6273  | 0.0198     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| fair_random     | 0.5476  | 0.0326     | [[resources/lit_notes/bonartFairRankingAcademica]]           |          | [✔](https://zenodo.org/record/3514668) |
| QUARTZ-e0.00010 | 0.6241  | 0.0330     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00001 | 0.6247  | 0.0332     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00100 | 0.6228  | 0.0347     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00200 | 0.6230  | 0.0348     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| uognleDivAJc    | 0.5544  | 0.0352     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| first           | 0.5507  | 0.0428     | [[resources/lit_notes/wangICTTREC2019]]                      |          | ❌                                     |
| uognleSgbrFair  | 0.6151  | 0.0649     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| uognleSgbrUtil  | 0.6151  | 0.0649     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| fair_LambdaMART | 0.6599  | 0.0741     | [[resources/lit_notes/bonartFairRankingAcademica]]           |          | [✔](https://zenodo.org/record/3514668) |
| MacEwanBase     | 0.6194  | 0.0770     | ❌                                                           |          | ❌                                     |
| uognleMaxUtil   | 0.6741  | 0.0799     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |

## h-index with 4 groups
| run             | utility | unfairness | paper                                                        | Approach | code                                   |
| --------------- | ------- | ---------- | ------------------------------------------------------------ | -------- | -------------------------------------- |
| fair_random     | 0.5476  | 0.0405     | [[resources/lit_notes/bonartFairRankingAcademica]]           |          | [✔](https://zenodo.org/record/3514668) |
| uognleDivAJc    | 0.5544  | 0.0449     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| first           | 0.5507  | 0.0456     | [[resources/lit_notes/wangICTTREC2019]]                      |          | ❌                                     |
| MacEwanBase     | 0.6194  | 0.0476     | ❌                                                           |          | ❌                                     |
| uognleSgbrFair  | 0.6151  | 0.0482     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| uognleSgbrUtil  | 0.6151  | 0.0482     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| uognleDivAAsp   | 0.5612  | 0.0585     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| uognleMaxUtil   | 0.6741  | 0.0656     | [[resources/lit_notes/mcdonaldUniversityGlasgowTerrier2019]] |          | ❌                                     |
| fair_LambdaMART | 0.6599  | 0.0855     | [[resources/lit_notes/bonartFairRankingAcademica]]           |          | [✔](https://zenodo.org/record/3514668) |
| QUARTZ-e0.00001 | 0.6247  | 0.1036     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00010 | 0.6241  | 0.1059     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00100 | 0.6228  | 0.1068     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00200 | 0.6230  | 0.1071     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.01000 | 0.6273  | 0.1097     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |
| QUARTZ-e0.00500 | 0.6239  | 0.1112     | [[resources/lit_notes/melucciTestingFairnessUsing]]          |          | ❌                                     |