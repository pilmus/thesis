# Ranking/RecSys/NLP
## Causal Intersectionality and/for Fair Ranking
[[yangCausalIntersectionalityFair2021]]
[[resources/lit_notes/yangCausalIntersectionalityFair2020]]

### Approach
Using causal models, compute a new score for items to make the results fairer.

### Datasets
- Subset of COMPAS
- MEPS
- Synthetic data set ("Moving Company") (3 variations)

### Result
- Comes close to DP on COMPAS but only because of the prior choice of causal model.
	- Can you not always choose your model so that it seems as if you have achieved fairness?
	- This method requires someone to *decide* what variables cause discrimination.


- Counterfactual ranking has a way higher rKL but that's actually good? B/c counteracting historical discrimination? This is vague...

- Utility as measured by AP is reduced to... 77% around ish


### Oracle result

### Remarks
- Not all metrics shown for all scenarios (EO for score-based)
- Focuses specifically on intersectionality
- Code available!

---


## On Discrimination Discovery and Removal in Ranked Data using Causal Graph
[[wuDiscriminationDiscoveryRemoval2018]]

### Approach
- Discover direct and indirect discriminatory influence of some variables on rankings.
- Discrimination == a protected attribute influences the outcome somehow.
- Discrimination criterion is a ratio between... ?
- Solve quadratic program subject to discriminatory effect constraints.

- Infer scores based on observed rankings

1. Modify causal graph to be discrimination-free.
2. Construct fair ranking based on modified causal graph.

### Dataset
- Ranked datasets based on German Credit 
	- Ordered by...
		1. Sum of all attributes ([[yangMeasuringFairnessRanked2017]])
		2. Sum of all except one attributes ([[yangMeasuringFairnessRanked2017]])
		3. Original credit score
- Available through [here](https://www.yongkaiwu.com/publication/wu-2018-discrimination/wu-2018-discrimination.zip).

### Result
- Theoretically optimal - no more discrimination
- Unclear how it performs wrt other fairness metrics

### Remarks
- $SE^E_{\pi}(c^+,c^-)$ is the specific effect of protected attribute $C$ on "binary decision attribute" $E$.

---

## Towards Personalized Fairness based on Causal Notion
[[liPersonalizedFairnessBased2021]]

### Approach
- Not all users find the same protected attributes equally important
- Goal: counterfactual user fairness
- Use adversarial learning to remove the causal link between user embedding and recommendation outcome

### Metrics
- AUC for "fairness" --> AUC of 0.5 means the attacker cannot guess the sensitive feature

### Data

- MovieLens
- Insurance (Kaggle)


---


## Deconfounded Recommendation for Alleviating Bias Amplification
[[wangDeconfoundedRecommendationAlleviating2021]]

### Approach
- Use causal model to find the source of bias in data
- Use an approximation of "backdoor adjustment" to remove the bias during training

### Datasets
- MovieLens 1M
- Amazon-Book

### Results
- Lower calibration drift than the other compared methods

### Remarks
- ❗ Recommendation rather than ranking
- Targets "Bias Amplification" rather than fairness per se
- Unclear how it performs for fairness metrics


---

# Counterfactual intervention NLP

## Gender Bias in Neural Natural Language Processing
[[luGenderBiasNeural2020]]

### Approach
- Detect bias in word embeddings
- Use counterfactual data augmentation to counteract the bias

✔ Counterfactual, ❌ *causal modeling*



---

# Causal language model

## Intersectional Bias in Causal Language Models
[[mageeIntersectionalBiasCausal2021]]

### Approach
- Use causal language models to generate sentences about different intersectional groups.
- Sentiment analysis.
- Is the sentiment more negative towards some groups?

### Datasets

---

# Algorithmic fairness

## Causal Multi-Level Fairness
[[mhasawadeCausalMultilevelFairness2021]]

### Approach
1. Learn parameters of causal model.
2. 


### Datasets
- UCI Adult dataset