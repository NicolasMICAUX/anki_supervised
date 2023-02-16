

### Languages
XLM-Roberta [supports 100 languages](https://huggingface.co/xlm-roberta-large).  
They are listed in [Appendix A of the paper](https://arxiv.org/pdf/1911.02116.pdf).  
All the languages in the dataset are supported:
_en, de, pt, ar, vi, ja, es, fr, ko, it, ro, ceb, zh-hant, zh-hans, fa, hr, ca, nl, id, hy, cs, et, ml, sl, eu_.

## Use Fairness to improve results
I think the model is learning the individual Anki packages themselves, and not a generalizable model.

#### Tensorflow Model Remediation
**https://www.tensorflow.org/responsible_ai/model_remediation**
##### MinDiff
https://www.tensorflow.org/responsible_ai/model_remediation/min_diff/guide/mindiff_overview
We need an output layer that is a single value. Here, it will just ensure that all anki packages
receive the same probability of a word being masked. Not what we want.
##### CLP
https://www.tensorflow.org/responsible_ai/model_remediation/counterfactual/guide/counterfactual_overview
Mainly use on classification models. We do not have paired data, so we cannot use this.
##### DP-SGD
https://www.tensorflow.org/responsible_ai/privacy/guide
Would enable to train without being able to recover Anki package data. Not what we want.

##### TFCO
**https://github.com/google-research/tensorflow_constrained_optimization/blob/master/README.md**
**https://github.com/google-research/tensorflow_constrained_optimization/blob/master/examples/jupyter/Generalization_communities.ipynb**

#### AIF360
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.ExponentiatedGradientReduction.html#aif360.algorithms.inprocessing.ExponentiatedGradientReduction
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html#aif360.algorithms.inprocessing.PrejudiceRemover
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.MetaFairClassifier.html#aif360.algorithms.inprocessing.MetaFairClassifier
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.GerryFairClassifier.html#aif360.algorithms.inprocessing.GerryFairClassifier
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.ARTClassifier.html#aif360.algorithms.inprocessing.ARTClassifier
https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html#aif360.algorithms.inprocessing.AdversarialDebiasing
Only for classification.

#### FairLearn
https://fairlearn.org/v0.8/api_reference/fairlearn.adversarial.html
Only for classification.

## Dependency parsing as features
A lot of models on the web. Seems that the task has not seen many progress since 2018-2019, so I take the most easy-to-use library.
**Shortlist**
- https://github.com/yzhangcs/parser (uses stanza internally)
- https://github.com/stanfordnlp/stanza/
- https://github.com/idiap/g2g-transformer (moins entretenu)
- https://spacy.io/api/dependencyparser (utilise un algo de 2014)
https://malaya.readthedocs.io/en/stable/load-dependency-huggingface.html (malais seulement)

### Supar (yzhangcs)
'biaffine-dep-xlmr' is a multi-lingual dependency parser
it's the only dep parser XLMR model: https://github.com/yzhangcs/parser/releases/tag/v1.1.0

#### constituency-vs-dependency-parsing
2 different tasks, both may be relevant to P2A
https://www.baeldung.com/cs/constituency-vs-dependency-parsing

maybe test crf-con-xlmr one day
