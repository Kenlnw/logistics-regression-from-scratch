#include <sorach/sorach.h>
#include "src/model.h"
#include "src/read.h"
#include "src/customers.h"

#define MB(x) 1024 * 1024 * x

int main(void)
{
    const char *dataset_path = "csv/bank-additional-onehot.csv";
    const char *model_path = "bank_model.bin";
    const char *classes[2] = {"NO", "YES"};

    SorArena arena = sor_arena_create(MB(64));

    int rows = 0, cols = 0;
    count_data_size(&arena, &rows, &cols, dataset_path);

    char **feature_names = sor_arena_alloc(&arena, (size_t)cols * sizeof(char*), SOR_ARENA_DEFAULT_ALIGN);
    if (!read_header(&arena, feature_names, cols, dataset_path)) return 0;

    LogReg model = {0};
    if (!model_load(&arena, &model, model_path)) return 1;
    model_summary(&model, (const char *const *)feature_names);

    printf("Adding 5 new customers to the model\nfor predicting if these customers will subscribe a term deposit.\n");

    SorMat X_new = sor_mat_new(&arena, N_CUSTOMERS, N_FEATURES + 1);
    sor_mat_from_array(X_new, customers);
    standardize_apply(X_new, model.mean, model.std);
    printf("Added the data successfully.\n\n");

    SorMat y_pred = sor_mat_new(&arena, N_CUSTOMERS, 1);
    model_predict(&model, y_pred, X_new);

    SorMat y_prob = sor_mat_new(&arena, N_CUSTOMERS, 1);
    model_predict_prob(&model, y_prob, X_new);
    sor_mat_scale(y_prob, 100);

    printf("Here's is the result of the prediction.\n");

    for (int i = 0; i < y_pred.rows; i++) {
        printf("\nCustomer %d : %s\n", i + 1, customers_names[i]);
        printf("%s\n", customers_details[i]);
        printf("Subscribe : %s (%.2f%%)\n", classes[(int)y_pred.data[i]], y_prob.data[i]);
    }

    sor_arena_destroy(&arena);
    return 0;
}
