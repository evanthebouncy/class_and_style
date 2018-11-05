# What To Run

At the root directory of ironbox

## artificial classification problem

    python -m experiments.artificial_classify fcnet 400

fcnet is a evaluation model name, we can make more

400 is the size of the subset to be selected, the size of X\_train is 1000, so
I think we can run 10 20 ... 500 would be fine There is a some kind of "bug" in
subset\_select\_knn code that would cause infinite loop if going all the way to
1000 and slow if subset size is large

## MNIST classification problem

    python -m experiments.mnist_classify fcnet 200

same as artificial classification. It will train an auto-encoder if the saved model dont exist
you should try with different parameters, this take awhile to run . . . 
