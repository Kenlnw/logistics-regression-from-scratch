# Makefile for a project that uses Sorach as a git submodule in libs/sorach.
#
#   make          build both programs (build/train and build/test)
#   make train    build train.c, then run it
#   make test     build test.c, then run it
#   make clean    delete this project's build files and Sorach's
#
# Options: DOUBLE=1 builds Sorach and both programs with double, into
# build_double/ so they never mix with the float programs in build/.
# Files saved by a float program cannot be loaded by a double one, and
# the other way round: sor_mat_load refuses them with a message.
#
# Code shared by both programs (anything without a main) can go in src/;
# every .c file there is compiled into train and into test.

CC     := gcc
CFLAGS := -std=c99 -Wall -Wextra -O2 -Ilibs/sorach/include -Isrc
LDLIBS := -lm

SORACH       := libs/sorach
SORACH_FLAGS := RELEASE=1
SORACH_BUILD := build
BUILD        := build

ifdef DOUBLE
CFLAGS       += -DSOR_USE_DOUBLE
SORACH_FLAGS += DOUBLE=1
SORACH_BUILD := $(SORACH_BUILD)_double
BUILD        := build_double
endif

# Must match the folder Sorach's own Makefile picks for these options.
SORACH_LIB := $(SORACH)/$(SORACH_BUILD)_release/libsorach.a

ifeq ($(OS),Windows_NT)
EXE := .exe
endif

# cmd (mingw32-make from PowerShell) has no rm; see Sorach's Makefile.
ifeq ($(shell echo "x"),"x")
RMDIR := rmdir /s /q
else
RMDIR := rm -rf
endif

SHARED    := $(wildcard src/*.c)
TRAIN_BIN := $(BUILD)/train$(EXE)
TEST_BIN  := $(BUILD)/test$(EXE)

# These names are commands, not files, so make always runs them.
.PHONY: all train test clean FORCE

all: $(TRAIN_BIN) $(TEST_BIN)

train: $(TRAIN_BIN)
	./$(TRAIN_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

# Always ask Sorach's own Makefile to bring the library up to date, with its
# own settings. It only recompiles what changed, and the programs below are
# relinked only when the library file actually changed.
$(SORACH_LIB): FORCE
	$(MAKE) -C $(SORACH) $(SORACH_FLAGS)

FORCE:

$(TRAIN_BIN): train.c $(SHARED) $(SORACH_LIB) | $(BUILD)
	$(CC) $(CFLAGS) train.c $(SHARED) $(SORACH_LIB) -o $@ $(LDLIBS)

$(TEST_BIN): test.c $(SHARED) $(SORACH_LIB) | $(BUILD)
	$(CC) $(CFLAGS) test.c $(SHARED) $(SORACH_LIB) -o $@ $(LDLIBS)

$(BUILD):
	mkdir $(BUILD)

clean:
	$(MAKE) -C $(SORACH) clean
	$(foreach d,$(wildcard build build_double),$(RMDIR) $(d) &&) echo cleaned