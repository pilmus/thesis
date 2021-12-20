# Agenda 
1. Daily supervisor --> PhD student?
2. Talk through the different possible topics
	1. Main challenge: how decide which topic is worth pursuing? No result is also a result I guess...

- Use methods from other fields such as resource scheduling for fair ranking
	- Has it already been done?
	- *Can* it be done (vertaalslag)

	
- Causal fairness
	- [[wuDiscriminationDiscoveryRemoval2018]] use certain models to model the causal graph
		- What happens if we use a different model?
		- Can be done with multiple protected attributes?
	- [[yangCausalIntersectionalityFair2021]] train ListNet with a causal graph (somehow)
		- Train a different LtR algorithm?
		- Can use a cyclic causal graph instead?
		- Find out what the normative judgment is encoded in the choice of causal model?
	- Combine counterfactual fairness and merit-based fairness somehow?
	- Is there a causal post-processing method? Or only in-processing?
	- Individual fairness under causal ranking? Currently all approaches target group fairness, also [[yangCausalIntersectionalityFair2021]] also look at intersectional fairness.


- Compositional fairness
	- [[wangPracticalCompositionalFairness2021]]
		- Apply the methods from this paper to ranking specifically
		- Look at different composition models
	- [[wangUnderstandingImprovingFairnessAccuracy2021]]
		- Can apply fair multi-task learning to ranking?
	- [[pitouraFairnessRankingsRecommendations2021]]
		- Investigate the fairness of a full ranking pipeline, so pre, in, and post processing. Does optimizing individual components automatically lead to a fairer ranking process overall?



3. Meet again in three weeks.


# Notes
1. Look at papers on causal fairness in IR. Which datasets do they use?
	1. Is the data publicly available?
	2. Is the dataset large enough?
2. If there is a good dataset:
	1. What is SOTA?
	2. How far is it from perfect?
	3. Reimplement
	4. Error analysis
	5. Theoretical improvements
	6. Implement improvements
	7. Re-evaluate


- Is the data publicly available?
- Causal
	- What exactly is the research question?
		- Is there an ERROR in a model?
		- People used model X, what is SOTA? But there's still a big gap to the Oracle result. 
			- Reimplement
			- Error analysis
			- Propose improvement
			- Write improvement
			- , and try to improve

	- The big issue is data: you can't just give people a BAD experience.
		- 2-3 days to find out whether there is data and if I can use it
		- Natural setups: in one market Netflix doesn't have the rights so they can't play it. Then look at what the effects are.
		- What datasets in NLP, IR, RS, look at KDD (conference).

- Compositional
	- Proprietary? Or academic? If proprietary not that great

- From other fields?
	- No matter the outcome, you have to have some finding. 
	- Just "nope" is not very good.



- Fairtrec: what did people do, how can we improve them?


- Next meeting: next Monday

	