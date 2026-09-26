\# Amazon ML Challenge 2026 - Handoff



\## Current Status



Working entity-resolution pipeline completed through Phase 5.



Pipeline:



Normalization

→ Candidate Generation

→ 27 Pairwise Features

→ LightGBM Matcher

→ Entity-level Matching



\## 10K Validation Results



\- Candidate Recall@20: 93.28%

\- Macro F0.5: 96.39%

\- Precision: 98.10%

\- Recall: 90.14%

\- Selected threshold: 0.65

\- Runtime: \~4.5 minutes

\- Peak RAM: \~1.66 GB

\- Entity-level train/validation split used

\- No entity leakage



\## Important Implementation Details



\- Candidate generation currently uses efficient blocking/indexing.

\- Candidate generation was validated at K=20.

\- 27 pairwise features are generated for candidate pairs.

\- LightGBM is used for pair classification.

\- Matching produces variable-length match lists, including empty lists.

\- Candidate generation, features, model and matching have been tested independently and end-to-end.



\## Known Limitation



The original Python dictionary indexing approach became too slow when indexing the full \~10M S2/S3 records.



Do NOT blindly rewrite the current pipeline.



Validate changes on the 10K experiment before attempting full-scale execution.



\## Next Steps



1\. Verify the current GitHub checkpoint.

2\. Run full training using the complete training data.

3\. Generate candidates/features for the test data.

4\. Train/finalize the matcher using the training data.

5\. Generate:

&#x20;  - output/matching\_results.tsv

&#x20;  - output/candidate\_pairs.tsv

6\. Run the official submission validator.

7\. Package the final submission.



\## Development Rules



\- Make incremental changes.

\- Benchmark before and after changes.

\- Do not modify working modules without testing.

\- Do not commit datasets, generated smoke data, caches, or large artifacts.

\- Keep candidate\_pairs.tsv consistent with the exact candidates supplied to the final model.

\- Every test S1 must appear exactly once in matching\_results.tsv.

\- Never predict a match that is not present in candidate\_pairs.tsv.

