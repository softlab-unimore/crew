# CREW: Clusters of RElated Words to explain Entity Matching

CREW is a cluster-based explainer for Entity Matching, that generates faithful and easily interpretable explanations to matching decisions. 
Given a pair of entity descriptions, classified as "matching" or "not matching" by any model, CREW:
- Clusters the words according to the relatedness heuristics detailed in the paper;
- Measures the importance of each cluster on the model behaviour with a local, post-hoc explainer.

An explanation cosists of each word cluster pairs with its importance coefficient. 

```
@article{}
```
