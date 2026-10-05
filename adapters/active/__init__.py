"""Active pair querying + target-domain prompt (the method of the Active_Pair_Prompt branch).

Setting: a source-trained ReID model with a deep visual prompt (the "base", e.g. a DG-trained VPT model) is
deployed in a new camera network. The target pool (its train split) is unlabeled; camera ids are known.
Each round:

  1. features of the pool under the current prompt                     image_store.features
  2. the model proposes cross-camera pairs it believes are one person   candidates.candidate_pairs
  3. a strategy ranks them; a human answers same / different           pair_selection, oracle.PairOracle
     (pairs whose answer follows by transitivity are skipped for free)  constraints.ConstraintStore
  4. positives grow identity clusters, negatives become hard negatives  constraints.ConstraintStore
  5. a domain prompt is tuned on the clusters, the base model frozen    prompt_tuning.tune_domain_prompt
     (append: extra tokens next to the frozen base prompt; replace: a tuned copy of the base prompt)

After the last round the domain prompt is frozen and inserted for retrieval (a constant per-domain
parameter). loop.ActiveRun runs the rounds; scripts/eval_active.py runs strategies x domains x splits.
"""
