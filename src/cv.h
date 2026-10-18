#ifndef CV_H
#define CV_H

#include <stddef.h>
#include <sorach/sorach.h>
#include "model.h"

#define CV_FOLDS 5

void shuffle_data_idx(int *row_idx, int n);
void take_rows(SorMat dst, SorMat src, int *row_idx, int count);

SorScalar cv_loss(SorArena *arena,
                  LogReg *model,
                  SorMat X,
                  SorMat y,
                  const int *data_idx,
                  SorScalar lambda,
                  int folds);

int find_best_lambdas(SorArena *arena,
                     LogReg *model,
                     SorMat X,
                     SorMat y,
                     SorMat lambdas,
                     int folds,
                     unsigned int rand_seed);

#endif