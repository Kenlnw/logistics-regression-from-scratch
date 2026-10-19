#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include <stdio.h>
#include <sorach/sorach.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define DEFAULT_THRESHOLD 0.5

// Logistics Regression's data structure
typedef struct {
    SorScalar lr;
    SorScalar lambda;
    SorScalar threshold;
    SorScalar loss;
    size_t epochs;
    SorMat w;
    SorMat mean;
    SorMat std;
} LogReg;

LogReg model_create(SorScalar threshold, size_t epochs);
SorScalar sigmoid(SorScalar z);
SorScalar loss_func(SorMat O, SorMat y);
void model_predict_prob(LogReg *model, SorMat out, SorMat X);

SorScalar model_accuracy(SorArena *arena, LogReg *model, SorMat X, SorMat y);

void model_predict(LogReg *model, SorMat out, SorMat X);

void standardize_fit(SorMat mean, SorMat std, SorMat src);
void standardize_fit_range(SorMat mean, SorMat std, SorMat train, int start, int end);
void standardize_apply(SorMat dst, SorMat mean, SorMat std);

void model_fit(SorArena *arena,
               LogReg *model,
               SorMat X,
               SorMat y,
               int verbose);

int model_save(LogReg *model, const char *path);
int model_load(SorArena *arena, LogReg *model, const char *path);
void model_reset_params(LogReg *model);

// Print a readable summary of a trained model.
// feature_names may be NULL; otherwise it must hold w.rows - 1 names
// (one per feature, in column order, without the bias).
void model_summary(const LogReg *model, const char *const *feature_names);

#endif