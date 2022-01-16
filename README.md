# Intro

This is the official repository for DoubleMatch.

The code is build on the FixMatch repo. For more details on how to use this code we refer to their instructions.

Code for the paper: "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by 
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.


## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3 python3-dev python3-tk imagemagick python3-pip
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the DoubleMatch code"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```


### Reproducing results in paper


```bash

DOUBLEMATCH_PATH="path to the DoubleMatch code"

export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:$DOUBLEMATCH_PATH

TRAIN_KIMG=22000
EXPERIMENT_PATH="path for storing results"

# CIFAR-10
LABELS=(40 250 4000)
WS=(0.5 1 5)
for SEED in 1 2 3 4 5; do
for i in ${!LABELS[@]}; do
    DATASET="cifar10.${SEED}@${LABELS[$i]}-1"
    python3 ${DOUBLEMATCH_PATH}/doublematch.py --dataset=${DATASET} --ws=${WS[$i]} --train_kimg=${TRAIN_KIMG}\
    --train_dir=${EXPERIMENT_PATH}
done; done;

# SVHN
LABELS=(40 250 1000)
WS=(0.001 0.05 0.05)
for SEED in 1 2 3 4 5; do
for i in ${!LABELS[@]}; do
    DATASET="svhn_noextra.${SEED}@${LABELS[$i]}-1"
    python3 ${DOUBLEMATCH_PATH}/doublematch.py --dataset=${DATASET} --ws=${WS[$i]} --train_kimg=${TRAIN_KIMG}\
    --train_dir=${EXPERIMENT_PATH}
done; done;

# CIFAR-100
LABELS=(400 2500 10000)
WS=(2 5 10)
COSINEDECAY=$(bc <<< "scale=4; 5/8")
for SEED in 1 2 3 4 5; do
for i in ${!LABELS[@]}; do
    DATASET="cifar100.${SEED}@${LABELS[$i]}-1"
    python3 ${DOUBLEMATCH_PATH}/doublematch.py --dataset=${DATASET} --ws=${WS[$i]} --train_kimg=${TRAIN_KIMG}\
    --train_dir=${EXPERIMENT_PATH} --wd=0.001 --filters=128 --cosinedececay=${COSINEDECAY}
done; done;

# STL-10
for SEED in 1 2 3 4 5; do
    DATASET="stl10.${SEED}@1000-1"
    python3 ${DOUBLEMATCH_PATH}/doublematch.py --dataset=${DATASET} --ws=1 --train_kimg=${TRAIN_KIMG}\
    --train_dir=${EXPERIMENT_PATH} --scales=4
done;

# EXTRACT RESULTS FROM EVENT-FILES
find "${EXPERIMENT_PATH}" -type d -name "*Match*" -exec \
python3 ${DOUBLEMATCH_PATH}/scripts/extract_accuracy_extended.py {} \;


# EVALUATE MEAN AND STANDARD DEVIATION OF ACCURACIES
genmeanstd () {
    DATASET=$1
    SIZE=$2

    echo "${DATASET}-${SIZE}"
    singularity exec --nv /proj/ssl_erik/fixmatch.simg \
		find ${EXPERIMENT_DIR} -path "*${DATASET}.d.d.d.*@${SIZE}-1/*/DoubleMatch*" \
		-type f -name "accuracy.json" \
		-exec python3 "${DOUBLEMATCH_PATH}/scripts/generate_meanstd.py {} +
}

genmeanstd "cifar10" 40
genmeanstd "cifar10" 250
genmeanstd "cifar10" 4000
genmeanstd "cifar100" 400
genmeanstd "cifar100" 2500
genmeanstd "cifar100" 10000
genmeanstd "stl10" 1000
genmeanstd "svhn_noextra" 40
genmeanstd "svhn_noextra" 250
genmeanstd "svhn_noextra" 1000

```

