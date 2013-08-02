random_forest_compare
=====================

Comparison of 4 random forest packages

## Depedencies
Python 2.7
scikit-learn (depends on numpy)
memory_profiler
R
WiseRF

## Configuration
Make your environment aware of your BigML credentials per [this procedure](http://bigml.readthedocs.org/en/latest/#authentication).

Edit wiserf_tester.py to specify the path to your WiseRF executable file.

## Usage
Launch the tester with:

    {python|python2} forest_compare.py [method] [path to data] [regression?]

Method can be "sklearn", "wiserf", or "bigml". Data path points to a folder containing the CSV source files. To specify a regression
task, append the word "regression" as the final argument. Otherwise, the script
assumes a classification task.

The R tester can be run with

    Rscript r-test.r
