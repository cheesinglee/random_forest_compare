#/bin/bash

set -vx
CWD=`pwd`
WINTERMUTE_DIR=/home/ubuntu/wintermute
TARGET_DIR=/data/csv-data/synth/
CLASSIFICATION_DIR=$TARGET_DIR/classification
REGRESSION_DIR=$TARGET_DIR/regression
NUM_CLASSES=3
cd $WINTERMUTE_DIR

mkdir -p {$CLASSIFICATION_DIR/cat,$REGRESSION_DIR/cat}

# CLASSIFICATION

# increasing rows
for i in 10000000
do
 FILENAME=$CLASSIFICATION_DIR/synthdata_${i}_10.csv
 lein generate-csv $FILENAME 10 :rows $i :frac-cat 0 :noise 0.15 :num-classes $NUM_CLASSES
done

#increasing fields
for i in 10 100 1000 10000
do
 lein generate-csv $CLASSIFICATION_DIR/synthdata_20000_${i}.csv ${i} :rows 20000 :frac-cat 0 :noise 0.15 :num-classes $NUM_CLASSES
done

#categorical
for i in 8 16 32 64 128 256
do
 lein generate-csv $CLASSIFICATION_DIR/cat/synthdata_1000000_10_cat${i}.csv 10 :rows 1000000 :frac-cat 1 :noise 0.15 :num-cats ${i} :num-classes $NUM_CLASSES
done

# REGRESSION

#increasing rows
for i in 1000 10000 100000 1000000 10000000
do
 FILENAME=$REGRESSION_DIR/synthdata_${i}_10.csv
 lein generate-csv $FILENAME 10 :rows $i :frac-cat 0 :noise 0.15 :num-classes -$NUM_CLASSES
done

#increasing fields
for i in 10 100 1000 10000
do
 lein generate-csv $REGRESSION_DIR/synthdata_20000_${i}.csv ${i} :rows 20000 :frac-cat 0 :noise 0.15 :num-classes -$NUM_CLASSES
done

#categorical
for i in 8 16 32 64 128 256
do
 lein generate-csv $REGRESSION_DIR/cat/synthdata_1000000_10_cat${i}.csv 10 :rows 1000000 :frac-cat 1 :noise 0.15 :num-cats ${i} :num-classes -$NUM_CLASSES
done
