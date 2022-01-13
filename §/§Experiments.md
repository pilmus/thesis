# Experiments

Our experiments are based on the datasets for FairTrec 2019 and FairTrec 2020. These are both snapshots of the S2O (❗insert correct name❗) dataset at different points in time.

We use this track to compare our results because there is an available dataset and because there are two years of runs already available to perform error analysis on.

We replicate the following approaches:
❗insert approaches❗

These were chosen for the following reasons: they are based on an elasticsearch database and the only paper for which source code is available online ([[bonartFairRankingAcademic]]) also uses an ES database so it is easier to extend. Also, ES has a learning to rank plugin. Further, we selected a variety of method and looked at the scores each approach achieved on their submitted runs. The latter is not a fool proof method, since the track is in such an early stadium that there is still a wide variance on the scores and not that many participants. ❗Also talk about the criticisms of the track designers/coordinators❗


## 2019
See the table below for our results.

| Run name        | Reported $U$ | Reported $\Delta_{hindex}$ | Reported $\Delta_{level}$ | Reproduced $U$ | Reproduced $\Delta_{hindex}$ | Reproduced $\Delta_{level}$ |
| --------------- | ------------ | -------------------------- | ------------------------- | -------------- | ---------------------------- | --------------------------- |
| fair_random     | 0.5840       | 0.0405                     | 0.0377                    | 0.5472         | 0.0377                       | 0.0312                      |
| fair_lambdaMART | 0.6600       | 0.0855                     | 0.0741                    | 0.6277         | 0.0514                       | 0.0779                      |
Table: Reproduction results.