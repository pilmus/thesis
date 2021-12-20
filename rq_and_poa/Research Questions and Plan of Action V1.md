# Research Questions
- RQ1: How do we translate a specific notion of causal fairness to the web search domain?
	- RQ1.1: Which notion of causal fairness is most suitable for the web search domain?
	- RQ1.2: Which assumptions and conditions must hold for the notion of causal fairness to apply to the web search domain?
	
- RQ2: How can we mitigate causal unfairness in our chosen dataset?
	- RQ2.1: What is the origin of the bias in our dataset?
	- RQ2.2: What is the most appropriate point of intervention for mitigating the causal unfairness in our dataset?
	- RQ2.3: To what extent can we mitigate the causal unfairness in our dataset?
	- RQ2.4: What is the impact of our mitigation strategy on the accuracy?
	

# Plan of Action
- How do we define fairness in the context of web search?
	- How do we define web search for the purposes of this thesis?
	- What kind of fairness is most relevant for the web search domain? E.g. equal opportunity, statistical parity, etc.

- [[makhloufSurveyCausalbasedMachine2021]] present an exhaustive list of causal fairness notions up to 18 Jan 2021. Which of these is most suitable for translating to the web search domain?
	- What is the intuitive interpretation of each causal fairness notion; what kind of unfairness does it seek to mitigate? Of these, which fairness notion matches with the kind of unfairness that is most relevant for web search?
	- What associative fairness measures are suitable for web search?

- How do we translate our chosen causal fairness notion to the web search domain?
	- What are the assumptions we need to make about our data?
	- How can we find the causal relationships between the variables in our dataset?
	- How can we measure causal unfairness in our dataset?
	- How can we mitigate the causal unfairness in our dataset?
		- What is the appropriate point of intervention to mitigate the causal unfairness in our dataset? (May be different for a score-based or an LTR approach.)
		- Look at [[yangCausalIntersectionalityFair2021]], [[wuDiscriminationDiscoveryRemoval2018]] (and maybe [[liPersonalizedFairnessBased2021]], [[wangDeconfoundedRecommendationAlleviating2021]]) for inspiration.

- How do we evaluate the effectiveness of our proposed mitigation procedure?
	- Which web search/IR dataset is suitable for evaluating fairness?
		- FAIRTREC 2019 & 2020 (Semantic Scholar): Authors of academic articles split into groups.
		- FAIRTREC 2021 (WikiProject): Documents with specific protected attributes.
		- (Yahoo! Answers)
		- (StackExchange)
		- (YowNews)
	- To which extent are we able to remove the observed causal unfairness?
	- What is the influence of our mitigation procedure on relevant associative fairness metrics?
	- What is the influence of our mitigation procedure on accuracy metrics?

- Once the above questions are answered, what improvements could be made to our procedure?

# Raw
[[rq_ideas_2021_11_12.pdf]]