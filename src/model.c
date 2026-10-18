#include "model.h"


LogReg model_create(SorScalar lr, SorScalar threshold, size_t epochs) {
    LogReg model = {
        .lr = lr,
        .loss = 0.0,
        .lambda = 0.0,
        .threshold = threshold,
        .epochs = epochs,
        .w = {0},
        .mean = {0},
        .std = {0}
    };
    return model;
}

SorScalar sigmoid(SorScalar z) {
    return (SorScalar)(1 / (1 + exp(-z)));
}

SorScalar loss_func(SorMat O, SorMat y) {
    const double epsilon = 1e-7;
    double cross_en = 0.0;

    for (int i = 0; i < O.rows; i++) {
        double o = SOR_MAT_AT(O, i, 0);
        if (o < epsilon) o = epsilon;
        if (o > 1 - epsilon) o = 1 - epsilon;

        cross_en += y.data[i] * log(o) + (1 - y.data[i]) * log(1 - o);
    }
    return (SorScalar)(-cross_en / y.rows);
}

void model_predict_prob(LogReg *model, SorMat out, SorMat X) {
    sor_mat_mul(out, X, model->w);
    sor_mat_apply(out, out, sigmoid);
}

void model_predict(LogReg *model, SorMat out, SorMat X) {
    model_predict_prob(model, out, X);
    for (int i = 0; i < out.rows * out.cols; i++)
        out.data[i] = out.data[i] >= model->threshold;
}

SorScalar model_accuracy(SorArena *arena, LogReg *model, SorMat X, SorMat y) {
    size_t mark = sor_arena_mark(arena);

    SorMat pred = sor_mat_new(arena, y.rows, 1);
    model_predict(model, pred, X);

    int correct = 0;
    for (int i = 0; i < pred.rows * pred.cols; i++)
        correct += (pred.data[i] == y.data[i]);

    sor_arena_rewind(arena, mark);
    return (SorScalar)correct / pred.rows;
}

void standardize_fit(SorMat mean, SorMat std, SorMat src) {
    assert(mean.rows == 1 && mean.cols == src.cols);
    assert(std.rows == 1 && std.cols == src.cols);

    sor_mat_mean(mean, src, SOR_MAT_ROW);
    sor_mat_std(std, src, SOR_MAT_ROW);

    // Handle the first col that is all 1s
    SOR_MAT_AT(mean, 0, 0) = 0;
    // guard the std when it very small (approximated 0)
    for (int j = 0; j < std.cols; j++)
        if (SOR_MAT_AT(std, 0, j) < 1e-8) SOR_MAT_AT(std, 0, j) = 1;
}

void standardize_fit_range(SorMat mean, SorMat std, SorMat train, int start, int end) {
    standardize_fit(mean, std, train);
    for (int j = 0; j < train.cols; j++) {
        if (j < start || j > end) {
            SOR_MAT_AT(mean, 0, j) = 0;
            SOR_MAT_AT(std, 0, j) = 1;
        }
    }
}

void standardize_apply(SorMat dst, SorMat mean, SorMat std) {
    sor_mat_sub_vec(dst, mean);
    sor_mat_div_vec(dst, std);
}

void model_fit(SorArena *arena, LogReg *model, SorMat X, SorMat y, int verbose) {
    if (model->w.data == NULL) model->w = sor_mat_new(arena, X.cols, 1);
    assert(model->w.rows == X.cols && model->w.cols == 1);
    assert(y.rows == X.rows && y.cols == 1);

    size_t mark = sor_arena_mark(arena);

    SorMat O = sor_mat_new(arena, X.rows, model->w.cols);
    SorMat pred = sor_mat_new(arena, X.rows, model->w.cols);
    SorMat M = sor_mat_new(arena, pred.rows, pred.cols);
    SorMat grad = sor_mat_new(arena, X.cols, M.cols);

    const SorScalar prob_eps = 1e-7;
    const SorScalar grad_tol = 1e-5;

    for (int epoch = 0; epoch < model->epochs; epoch++) {
        model_predict_prob(model, O, X);    // apply sigmoid to z = X @ model->w
        sor_mat_clamp(O, prob_eps, 1 - prob_eps);

        // Calculate gradiant of the loss function
        sor_mat_sub(M, O, y);   // SorMat M = O - y;
        sor_mat_mul_at(grad, X, M);  // SorMat grad = X.T @ M;
        sor_mat_scale(grad, (SorScalar)1 / y.rows);

        // Add a regularization l2 (ridge)
        for (int j = 1; j < model->w.rows; j++)
            SOR_MAT_AT(grad, j, 0) += model->lambda * SOR_MAT_AT(model->w, j, 0);

        if ((epoch % 100 == 0 || epoch == model->epochs - 1) && verbose == 1) {
            // Calculate loss function for every 100 epochs
            SorScalar w0 = SOR_MAT_AT(model->w, 0, 0);
            SorScalar penalty = sor_mat_dot(model->w, model->w) - w0*w0;
            model->loss = loss_func(O, y) + (0.5 * model->lambda * penalty);

            SorScalar train_acc = model_accuracy(arena, model, X, y);
            printf("EPOCH %d/%d   loss: %f   accuracy: %f\n",
                    epoch, model->epochs, model->loss, train_acc);
        }

        if (sor_mat_norm(grad) < grad_tol) break;

        // Adjust weight vecotor by gradient descent
        sor_mat_add_scaled(model->w, -model->lr, grad);
    }

    sor_arena_rewind(arena, mark);
}

int model_save(LogReg *model, const char *path) {
    assert(model->w.data != NULL);
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "model_save: cannot open '%s'\n", path);
        return 0;
    }

    const char code[4] = {'L', 'R', 'G', '1'};
    int version = 1;

    int is_completed = fwrite(code, sizeof(char), 4, fp) == 4
                    && fwrite(&version, sizeof(int), 1, fp) == 1
                    && fwrite(&model->threshold, sizeof(model->threshold), 1, fp) == 1
                    && fwrite(&model->lambda, sizeof(model->lambda), 1, fp) == 1
                    && fwrite(&model->lr, sizeof(model->lr), 1, fp) == 1
                    && fwrite(&model->epochs, sizeof(model->epochs), 1, fp) == 1
                    && sor_mat_write(model->w, fp) == 1
                    && sor_mat_write(model->mean, fp) == 1
                    && sor_mat_write(model->std, fp) == 1;

    fclose(fp);
    if (!is_completed) fprintf(stderr, "model_save: write failed for '%s'\n", path);
    return is_completed;
}

int model_load(SorArena *arena, LogReg *model, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "model_save: cannot open '%s'\n", path);
        return 0;
    }

    char code[4];
    int version;

    int is_completed = fread(code, sizeof(char), 4, fp) == 4
                    && memcmp(code, "LRG1", 4) == 0
                    && fread(&version, sizeof(int), 1, fp) == 1
                    && version == 1
                    && fread(&model->threshold, sizeof(model->threshold), 1, fp) == 1
                    && fread(&model->lambda, sizeof(model->lambda), 1, fp) == 1
                    && fread(&model->lr, sizeof(model->lr), 1, fp) == 1
                    && fread(&model->epochs, sizeof(model->epochs), 1, fp) == 1
                    && sor_mat_read(arena, &model->w, fp) == 1
                    && sor_mat_read(arena, &model->mean, fp) == 1
                    && sor_mat_read(arena, &model->std, fp) == 1;

    is_completed = is_completed && model->w.cols == 1
            && model->mean.rows == 1 && model->std.rows == 1
            && model->mean.cols == model->w.rows
            && model->std.cols == model->w.rows;

    fclose(fp);
    if (!is_completed)
        fprintf(stderr, "model_load: '%s' is not a valid model file\n", path);
    return is_completed;
}

void model_reset_params(LogReg *model) {
    sor_mat_fillall(model->w, 0.0);
    model->loss = 0.0;
}

void model_summary(const LogReg *model, const char *const *feature_names) {
    assert(model != NULL && model->w.data != NULL);

    int n_features = model->w.rows - 1;                 // row 0 is the bias
    SorScalar bias = SOR_MAT_AT(model->w, 0, 0);

    double norm = 0.0;
    for (int j = 1; j <= n_features; j++) {
        double w = SOR_MAT_AT(model->w, j, 0);
        norm += w * w;
    }
    norm = sqrt(norm);

    printf("==================== Model summary ====================\n");
    printf("Type             : logistic regression, L2 regularization\n");
    printf("Features         : %d (+ bias)\n", n_features);
    printf("Lambda           : %g\n", (double)model->lambda);
    printf("Learning rate    : %g\n", (double)model->lr);
    printf("Epochs           : %ld\n", (long)model->epochs);
    printf("Threshold        : %g\n", (double)model->threshold);
    printf("Standardization  : %s\n",
           (model->mean.data && model->std.data) ? "stored" : "MISSING");
    printf("Bias             : %+.4f\n", (double)bias);
    printf("Weight norm      : %.4f (bias excluded)\n", norm);
    printf("-------------------------------------------------------\n");
    printf("%-4s %-30s %10s\n", "#", "feature", "weight");

    for (int j = 1; j <= n_features; j++) {
        char name[16];
        const char *label = feature_names ? feature_names[j - 1] : name;
        if (!feature_names) snprintf(name, sizeof name, "x%d", j);

        printf("%-4d %-30.30s %+10.4f\n", j, label, (double)SOR_MAT_AT(model->w, j, 0));
    }
    printf("=======================================================\n");
}