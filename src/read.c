#include "read.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sorach/sorach.h>

int count_data_size(SorArena *arena, int *rows, int *cols, const char *path) {
    size_t mark = sor_arena_mark(arena);

    FILE *fp = fopen(path, "r");
    if (fp == NULL) return 0;

    char *line = sor_arena_alloc(arena, MAX_LINE_SIZE, SOR_ARENA_DEFAULT_ALIGN);

    // Safeguard reset
    *rows = 0;
    *cols = 0;

    if (fgets(line, MAX_LINE_SIZE, fp) == NULL) {
        printf("Total rows: 0 (File is empty)\n");
        fclose(fp);
        return 0;
    }

    for (int i = 0; line[i] != '\0'; i++)
        if (line[i] == ',') (*cols)++;
    (*cols)++;

    while (fgets(line, MAX_LINE_SIZE, fp) != NULL)
        (*rows)++;

    sor_arena_rewind(arena, mark);
    fclose(fp);
    return 1;
}

int prepare_data(SorArena *arena, SorMat X, SorMat y, const char *path) {
    size_t mark = sor_arena_mark(arena);

    FILE *fp = fopen(path, "r");
    if (fp == NULL) return 0;

    char *line = sor_arena_alloc(arena, MAX_LINE_SIZE, SOR_ARENA_DEFAULT_ALIGN);

    if (fgets(line, MAX_LINE_SIZE, fp) == NULL) {
        printf("Total rows: 0 (File is empty)\n");
        fclose(fp);
        return 0;
    }

    int row_count = 0;
    while (fgets(line, MAX_LINE_SIZE, fp) != NULL) {
        line[strcspn(line, "\n")] = '\0';

        char *token = line;
        int col_count = 1;
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] == ',') {
                line[i] = '\0';

                if (col_count < X.cols) SOR_MAT_AT(X, row_count, col_count) = atof(token);
                else SOR_MAT_AT(y, row_count, 0) = atof(token);

                token = line + i + 1;
                col_count++;
            }
        }

        // for compute the last column value
        if (col_count < X.cols) SOR_MAT_AT(X, row_count, col_count) = atof(token);
        else SOR_MAT_AT(y, row_count, 0) = atof(token);

        row_count++;
    }

    sor_arena_rewind(arena, mark);
    return 1;
}

int read_header(SorArena *arena, char **headers, int max_count, const char *path) {
    assert(arena != NULL && headers != NULL && max_count > 0);

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "read_header: cannot open '%s'\n", path);
        return 0;
    }

    // 1. measure the first line, so its exact size can be allocated
    size_t len = 0;
    int c;
    while ((c = fgetc(fp)) != EOF && c != '\n') len++;
    if (len == 0) {
        fclose(fp);
        return 0;
    }

    // 2. go back to the start and read the line into the arena
    rewind(fp);
    char *line = sor_arena_alloc(arena, len + 1, 1);
    size_t got = fread(line, 1, len, fp);
    fclose(fp);
    if (got != len) return 0;
    line[len] = '\0';

    // 3. drop a Windows line ending
    if (len > 0 && line[len - 1] == '\r') line[--len] = '\0';

    // 4. split on commas in place: each comma becomes the end of a name
    int count = 0;
    char *start = line;
    for (char *p = line; ; p++) {
        if (*p == ',' || *p == '\0') {
            int at_end = (*p == '\0');
            *p = '\0';

            // trim spaces and surrounding quotes
            while (*start == ' ' || *start == '"') start++;
            char *end = p - 1;
            while (end >= start && (*end == ' ' || *end == '"')) *end-- = '\0';

            if (count == max_count) {
                fprintf(stderr, "read_header: '%s' has more than %d columns\n", path, max_count);
                return 0;
            }
            headers[count++] = start;

            if (at_end) break;
            start = p + 1;
        }
    }
    return count;
}