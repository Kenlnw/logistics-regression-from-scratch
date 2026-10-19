#include "cv.h"
#include <sorach/sorach.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void shuffle_data_idx(int *data_idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = data_idx[i];
        data_idx[i] = data_idx[j];
        data_idx[j] = t;
    }
}

void take_rows(SorMat dst, SorMat src, int *data_idx, int count) {
    assert(dst.data != NULL && src.data != NULL);
    assert(dst.cols == src.cols && dst.rows == count);
    for (int i = 0; i < count; i++)
        sor_mat_set_row(dst, i, sor_mat_row_view(src, data_idx[i]));
}

SorScalar cv_loss(SorArena *arena, LogReg *model, SorMat X, SorMat y, const int *data_idx, SorScalar lambda, int folds) {
    int n = X.rows;
    double loss_total = 0.0;
    model->lambda = lambda;

    printf("%d-folds-cv. lambda: %f\n", folds, lambda);
    for (int k = 0; k < folds; k++) {
        size_t mark = sor_arena_mark(arena);

        int val_start = k * n / folds;
        int val_end = (k + 1) * n / folds;

        int n_val = val_end - val_start;
        int n_train = n - n_val;

        printf("Round %d/%d: %d training data, %d validation data\n", k+1, folds, n_train, n_val);

        int *train_idx = sor_arena_alloc(arena, n_train * sizeof(int), SOR_ARENA_DEFAULT_ALIGN);
        int *val_idx = sor_arena_alloc(arena, n_val * sizeof(int), SOR_ARENA_DEFAULT_ALIGN);
        int t = 0, v = 0;
        for (int i = 0; i < n; i++) {
            if (i >= val_start && i < val_end) val_idx[v++] = data_idx[i];
            else train_idx[t++] = data_idx[i];
        }

        SorMat X_train = sor_mat_new(arena, n_train, X.cols);
        SorMat X_val = sor_mat_new(arena, n_val, X.cols);
        SorMat y_train = sor_mat_new(arena, n_train, 1);
        SorMat y_val = sor_mat_new(arena, n_val, 1);

        take_rows(X_train, X, train_idx, n_train);
        take_rows(X_val, X, val_idx, n_val);
        take_rows(y_train, y, train_idx, n_train);
        take_rows(y_val, y, val_idx, n_val);

        int numeric_cols = 10;
        SorMat mean = sor_mat_new(arena, 1, X_train.cols);
        SorMat std = sor_mat_new(arena, 1, X_train.cols);
        standardize_fit_range(mean, std, X_train, 1, numeric_cols);
        standardize_apply(X_train, mean, std);
        standardize_apply(X_val, mean, std);

        model_reset_params(model);
        model_fit(arena, model, X_train, y_train, 0);

        SorMat val_prob = sor_mat_new(arena, n_val, 1);
        model_predict_prob(model, val_prob, X_val);

        SorScalar loss = loss_func(val_prob, y_val);

        SorScalar acc = model_accuracy(arena, model, X_val, y_val);

        printf("loss: %f,   accuracy = %f\n", loss, acc);

        loss_total += loss;
        sor_arena_rewind(arena, mark);
    }

    SorScalar loss_mean = (SorScalar)(loss_total / folds);
    printf("This lambda(= %f)'s loss mean: %f\n", lambda, loss_mean);
    return loss_mean;
}

int find_best_lambdas(SorArena *arena, LogReg *model, SorMat X, SorMat y, SorMat lambdas, int folds, unsigned int rand_seed) {
    if (model->w.data == NULL) model->w = sor_mat_new(arena, X.cols, 1);

    size_t mark = sor_arena_mark(arena);

    int *data_idx = sor_arena_alloc(arena, X.rows * sizeof(int), SOR_ARENA_DEFAULT_ALIGN);
    for (int i = 0; i < X.rows; i++) data_idx[i] = i;

    sor_mat_seed(rand_seed);
    shuffle_data_idx(data_idx, X.rows);

    printf("Find the best L2 regularization strength (lambda)\n");
    SorMat ld_loss = sor_mat_new(arena, lambdas.rows, 1);
    for (int l = 0; l < lambdas.rows; l++)
        SOR_MAT_AT(ld_loss, l, 0) = cv_loss(arena, model, X, y, data_idx, SOR_MAT_AT(lambdas, l, 0), folds);

    int best_lambda_idx = 0;
    sor_mat_argmin(&best_lambda_idx, 1, ld_loss, SOR_MAT_ROW);
    printf("The best lambda is %f (idx: %d)\n",
        SOR_MAT_AT(lambdas, best_lambda_idx, 0),
        best_lambda_idx);

    sor_arena_rewind(arena, mark);

    // Reset all params and lambda for using in the final training section
    model->lambda = SOR_MAT_AT(lambdas, best_lambda_idx, 0);
    model_reset_params(model);
    return best_lambda_idx;
}