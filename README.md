# Logistic Regression in C

Logistic regression written from scratch in C99, trained on the
[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
dataset to predict whether a customer will subscribe to a term deposit.
Matrix math and memory come from [Sorach](https://github.com/Kenlnw/sorach),
a small linear algebra library included as a git submodule.

## Features

- Batch gradient descent on the cross-entropy loss, with L2 regularization
- 5-fold cross-validation to pick the best regularization strength (lambda)
- Standardization of the numeric columns, with the mean and standard
  deviation saved alongside the weights
- Save a trained model to a binary file and load it back for prediction
- A model summary that lists every learned weight by feature name
- All memory taken from a single arena, freed in one call

## Project layout

```
.
├── train.c          cross-validate, train on the full dataset, save the model
├── test.c           load the model and predict for 5 example customers
├── src/
│   ├── model.c/.h   LogReg struct, fit, predict, standardize, save/load, summary
│   ├── cv.c/.h      k-fold cross-validation and lambda search
│   ├── read.c/.h    CSV reading
│   └── customers.h  the 5 example customers used by test.c
├── csv/
│   ├── bank-additional.csv         raw dataset (41,188 rows)
│   └── bank-additional-onehot.csv  same data with categories one-hot encoded
├── libs/sorach/     Sorach (git submodule)
└── Makefile
```

## Getting started

Clone with the submodule:

```sh
git clone --recurse-submodules <this-repo-url>
```

If you already cloned without it:

```sh
git submodule update --init
```

You need `gcc` and `make` (on Windows, MinGW's `mingw32-make` works too).

## Usage

```sh
make          # build build/train and build/test
make train    # build and run training; writes bank_model.bin
make test     # build and run prediction with bank_model.bin
make clean    # remove build files, including Sorach's
```

Add `DOUBLE=1` to any command to use `double` instead of `float`. Those
programs are built into `build_double/`. A model saved by a `float` build
cannot be loaded by a `double` build, and the other way round.

### Training

`make train`:

1. Reads `csv/bank-additional-onehot.csv` and adds a bias column
2. Runs 5-fold cross-validation for each lambda in
   `{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}` and keeps the one with the lowest mean
   validation loss
3. Standardizes the 10 numeric columns
4. Trains on the whole dataset with the best lambda
   (learning rate 0.1, 1000 epochs), printing loss and accuracy every 100 epochs
5. Prints the model summary and saves it to `bank_model.bin`

### Prediction

`make test` loads `bank_model.bin`, standardizes five example customers
from `src/customers.h` with the saved mean and standard deviation, and
prints each prediction with its probability. After the model summary,
the output looks like this (from the `bank_model.bin` in this repo; your
numbers will differ slightly after retraining, since the folds are
shuffled with a time-based seed):

```
Customer 1 : Kate
retired, previous campaign succeeded, called in October
age 67, call 420 s, contacts this campaign 1, last contacted 6 days ago, previous contacts 2
Subscribe : YES (86.93%)

Customer 2 : Jessie
student, long call in March
age 23, call 610 s, contacts this campaign 1, not contacted before, previous contacts 0
Subscribe : YES (82.03%)

Customer 3 : Lewis
blue-collar worker, short call in May, contacted many times
age 38, call 45 s, contacts this campaign 6, not contacted before, previous contacts 0
Subscribe : NO (0.32%)

Customer 4 : Steve
manager, medium-length call in August
age 45, call 280 s, contacts this campaign 2, not contacted before, previous contacts 1
Subscribe : NO (3.32%)

Customer 5 : Robert
technician, average customer in June
age 34, call 200 s, contacts this campaign 3, not contacted before, previous contacts 0
Subscribe : NO (2.17%)
```

To try your own customers, edit `src/customers.h`. Each row is a 1 for the
bias followed by the 62 features in the same column order as the header of
`csv/bank-additional-onehot.csv`.

## Dataset

S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the
Success of Bank Telemarketing.* Decision Support Systems, 2014.
Available from the UCI Machine Learning Repository.
