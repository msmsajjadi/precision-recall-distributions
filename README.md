# Assessing Generative Models via Precision and Recall

Official code for [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035) by [Mehdi S. M. Sajjadi](http://msajjadi.com), [Olivier Bachem](http://olivierbachem.ch/), [Mario Lucic](https://ai.google/research/people/MarioLucic), [Olivier Bousquet](https://ai.google/research/people/OlivierBousquet), and [Sylvain Gelly](https://ai.google/research/people/SylvainGelly), presented at [NIPS 2018](https://nips.cc/).

## Usage
### Manually: Compute PRD from any embedding
Example: you want to compare the precision and recall of a pair of generative models in some feature embedding to your liking (e.g., Inception activations).

1. Take your test dataset and generate the same number of data points from each of your generative models to be evaluated.
2. Compute feature embeddings of both real and generated datasets, e.g. `feats_real`, `feats_gen_1` and `feats_gen_2` as numpy arrays each of shape `[number_of_data_points, feature_dimensions]`.
3. Run the following code:
```python
import prd
prd_data_1 = prd.compute_prd_from_embedding(feats_real, feats_gen_1)
prd_data_2 = prd.compute_prd_from_embedding(feats_real, feats_gen_2)
prd.plot([prd_data_1, prd_data_2], ['model_1', 'model_2'])
```

### Automatically: Compute PRD for 2 folders of images on disk
_(Wrapper for automatically loading images from disk and computing Inception embeddings to be added to the repository.)_

## BibTex citation
```
@inproceedings{precision_recall_distributions,
  title={{Assessing Generative Models via Precision and Recall}},
  author={Sajjadi, Mehdi~S.~M. and Bachem, Olivier and Lu{\v c}i{\'c}, Mario and Bousquet, Olivier and Gelly, Sylvain},
  booktitle = {{Advances in Neural Information Processing Systems (NIPS)}},
  year={2018},
}
```

## Further information
See also the Google repository:
<https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score.py>

For any questions, comments or help to get it to run, please don't hesitate to mail us: <msajjadi@tue.mpg.de>
