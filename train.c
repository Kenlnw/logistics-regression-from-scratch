#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "src/model.h"
#include "src/read.h"
#include "src/cv.h"

#define MB(x) 1024 * 1024 * x

int main(void) {
    const char *train_path = "csv/bank-additional-onehot.csv";
    const char *model_path = "bank_model.bin";

    const SorScalar lambda_values[] = {0, 1e-4, 1e-3, 1e-2, 1e-1, 1};

    SorArena arena = sor_arena_create(MB(64));

    int train_rows = 0, train_cols = 0;

    count_data_size(&arena, &train_rows, &train_cols, train_path);

    SorMat X_train = sor_mat_new(&arena, train_rows, train_cols);
    SorMat y_train = sor_mat_new(&arena, train_rows, 1);
    sor_mat_fillall(X_train, 1.0);
    prepare_data(&arena, X_train, y_train, train_path);

    char **feature_names = sor_arena_alloc(&arena,
                (size_t)X_train.cols * sizeof(char*),
                SOR_ARENA_DEFAULT_ALIGN);

    int n_names = read_header(&arena, feature_names, X_train.cols, train_path);


    int n_lambdas = sizeof(lambda_values) / sizeof(lambda_values[0]);
    SorMat lambdas = sor_mat_new(&arena, n_lambdas, 1);
    sor_mat_from_array(lambdas, lambda_values);

    LogReg model = model_create(0.1, DEFAULT_THRESHOLD, 1000);
    model.w = sor_mat_new(&arena, X_train.cols, 1);
    int best_lambda_idx = find_best_lambdas(&arena, &model, X_train, y_train, lambdas, CV_FOLDS, (unsigned int)time(NULL));

    int numeric_cols = 10;
    if (model.mean.data == NULL && model.std.data == NULL) {
            model.mean = sor_mat_new(&arena, 1, X_train.cols);
            model.std = sor_mat_new(&arena, 1, X_train.cols);
    }

    standardize_fit_range(model.mean, model.std, X_train, 1, numeric_cols);
    standardize_apply(X_train, model.mean, model.std);

    printf("Final train with best lambda with X_train and y_train\n");
    model_fit(&arena, &model, X_train, y_train, 1);

    model_summary(&model, (const char *const *)feature_names);
    model_save(&model, model_path);
    printf("Saved the model in '%s' successfully.\n", model_path);

    sor_arena_destroy(&arena);
    return 0;
}

