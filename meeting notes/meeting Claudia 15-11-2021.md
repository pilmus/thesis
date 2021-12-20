# Meeting
- FairTrec is a good choice for the dataset: peer reviewed, error runs available, constructed for a search task

- Start with an error analysis: gives insight into where the current approaches fail
- Plan of action:
	1. Analyze current top runs for FairTrec (Chapter: Error Analysis)
		- What approaches do people use? Are there distinct categories? (e.g. neural, greedy algorithms)
			- Does anyone use a causal approach already?
		- Which queries do the top approaches get most wrong? Can we link this to their approach?
		- Are there any runs with open source implementations? Repeat the runs yourself as a sanity check.
			- If no open source: are there any we can easily re-implement?
		 
	2. Think: what approach could remedy the main errors we see? Is there a causal approach that could work? Or anything else?

	3. Implement our own algorithm, make some runs, do a new error analysis to see whether our approach helped.

