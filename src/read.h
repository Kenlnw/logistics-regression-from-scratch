#ifndef READ_H
#define READ_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sorach/sorach.h>

#define MAX_LINE_SIZE 1024

int count_data_size(SorArena *arena, int *rows, int *cols, const char *path);
int prepare_data(SorArena *arena, SorMat X, SorMat y, const char *path);
int read_header(SorArena *arena, char **headers, int max_count, const char *path);

#endif