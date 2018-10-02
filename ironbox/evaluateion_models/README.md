# evaluation models

Need to support 2 functions

## learn(train\_corpus)

train corpus can be as simple as (X,Y) or a DataHolder if we want to support
non-uniform sampling, which require different weights be associated with each
(x,y) data point

this function should completely train the evaluation model, I defined this as
training over and over on the training corpus until convergeance / saturation

## evaluate(test\_corpus)

call this after the evaluation model is trained, returns either classification
error (% misclassified) or MSE (for regression problems)
