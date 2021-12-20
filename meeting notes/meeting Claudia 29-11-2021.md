
# Summary
- Downloaded the dataset for 2019 and I think 2020.
- E-mailed everyone who participated in 2019 or 2020 to ask for their code.
- Downloaded the open source code I could find [[bonartFairRankingAcademic]].
	- Started trying to make it work.
	- Indexing all the files is a challenge --> fiddle with the ES settings.

# Questions & Answers
- Q: Do we only need the documents with relevance judgments?
- A: No, we need the full collection for e.g. collection frequency statistics.

- Q: What is a good way to choose papers to replicate?
- A: Go for a variety of approaches.

- Q: There are two published tasks for this track, should I focus on only one?
- A: No, look at both. 
	- Consider: Why did the task change between 2019, 2020 ( and 2021)?

- Q: How to interpret this formula from [[diazEvaluatingStochasticRankings2020]]:
	- ![[Pasted image 20211129084422.png]]
	- $\epsilon^*$ and $\epsilon$ are target and actual exposure respectively.
	- The difference between them can be measured by square loss:
		$$\begin{align}
		loss(\epsilon^*,\epsilon) &= ||\epsilon-\epsilon^*||^2_2\\
		&= ||\epsilon||^2_2 - 2\epsilon^T\epsilon^* + ||\epsilon^*||^2_2
		\end{align}$$
	- According to the paper...
		-  $||\epsilon||^2_2$ represents the disparity in distribution of exposure.
		-  $2\epsilon^T\epsilon^*$ represents how much of the exposure is on relevant documents.
		-  $||\epsilon^*||^2_2$ is a constant fixed for each information need.
	-  How do we arrive at this specific interpretation of those terms?
-  A: People often give names to parts of formulas to make it easier to refer to them.
	-  For more insight: construct toy example and see how it influences the metric.




# Other

- Could participate in FairTrec 2022

- I e-mailed x people and y got back with code <-- can put that in thesis