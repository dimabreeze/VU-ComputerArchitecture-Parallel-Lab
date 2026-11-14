# Parallel matrix multiplication: seq / OpenMP / MPI / OpenCL
#
# Every file src/<tech>/<variant>.cpp becomes a program build/<tech>-<variant>.
#
#   make                          build everything
#   make openmp                   build all OpenMP variants
#   make openmp-base              build one variant
#   make run-openmp-base ARGS="-s 1000 -t 8 --verify"
#   make run-mpi-base NP=4 ARGS="-s 1000"
#   make run-opencl-base ARGS="-s 2000 --kernel src/opencl/base/naive.cl"
#   make run-opencl-opt ARGS="-s 2000 --kernel src/opencl/opt/tiled.cl"   (default: regblock.cl)
#   make bench                    benchmark sweep -> rows appended to results/csv/*.csv
#   make memory                   memory microbenchmarks (build/tools/memlat) -> results/memory/*.csv
#                                 and plots docs/img/memory/*.png (docs/memory.md)
#   make compdb                   compile_commands.json for the IDE
#
# NOTE: the project path contains spaces, so only relative paths are used here.
# Written for GNU Make 3.81 (macOS default).

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS += -Iinclude

# OpenMP via Homebrew libomp (Apple clang has no bundled OpenMP runtime)
OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
OMP_LIBS   := -L$(OMP_PREFIX)/lib -lomp

MPICXX := mpicxx
MPIRUN := mpirun
NP     ?= 4

CL_CFLAGS := -DCL_KERNEL_DIR='"src/opencl"'
CL_LIBS   := -framework OpenCL

ARGS ?=

TECHS   := seq openmp mpi opencl
COMMON  := src/common/common.cpp
HEADERS := $(wildcard include/*.hpp)
SRCS    := $(foreach t,$(TECHS),$(wildcard src/$(t)/*.cpp))
# src/openmp/base.cpp -> openmp-base
NAMES   := $(foreach s,$(SRCS),$(word 2,$(subst /, ,$(s)))-$(basename $(notdir $(s))))

.PHONY: all clean bench plot reports memory compdb list $(TECHS) $(NAMES)

all: $(NAMES)

list:
	@echo $(NAMES)

# make openmp -> all openmp-* variants
$(foreach t,$(TECHS),$(eval $(t): $(filter $(t)-%,$(NAMES))))

# make openmp-base -> build/openmp-base
$(NAMES): %: build/%

build:
	@mkdir -p build

# ---- build rules (one per technology) ----
build/seq-%: src/seq/%.cpp $(COMMON) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

build/openmp-%: src/openmp/%.cpp $(COMMON) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMP_CFLAGS) $(filter %.cpp,$^) -o $@ $(OMP_LIBS)

build/mpi-%: src/mpi/%.cpp $(COMMON) $(HEADERS) | build
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

build/opencl-%: src/opencl/%.cpp $(COMMON) $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CL_CFLAGS) $(filter %.cpp,$^) -o $@ $(CL_LIBS)

# per-variant extra flags (target-specific variables)
build/openmp-opt2 build/seq-opt build/mpi-opt: CXXFLAGS += -O3 -mcpu=native

# ---- run rules (specific patterns first: make 3.81 picks the first match) ----
run-mpi-%: build/mpi-%
	$(MPIRUN) -np $(NP) ./$< $(ARGS)

run-%: build/%
	./$< $(ARGS)

bench: all
	scripts/bench.sh

# ---- memory microbenchmarks (not part of the bench sweep) ----
build/tools/memlat: src/tools/memlat.cpp
	@mkdir -p build/tools
	$(CXX) -std=c++17 -O2 -Wall -Wextra $< -o $@

# run on an otherwise idle machine: takes about 3 minutes
memory: build/tools/memlat .venv
	scripts/memory.sh

# ---- IDE support: compile_commands.json for clangd ----
MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
flags_seq    := $(CPPFLAGS) $(CXXFLAGS)
flags_openmp := $(CPPFLAGS) $(CXXFLAGS) $(OMP_CFLAGS)
flags_mpi    := $(CPPFLAGS) $(CXXFLAGS) $(MPI_CFLAGS)
flags_opencl := $(CPPFLAGS) $(CXXFLAGS)  # CL_KERNEL_DIR has a default in code
flags_common := $(flags_openmp) $(MPI_CFLAGS)

compdb:
	@{ echo '['; \
	  $(foreach s,$(SRCS),printf '  {"directory": "%s", "file": "%s", "command": "%s %s -c %s"},\n' "$$PWD" "$(s)" "$(CXX)" "$(flags_$(word 2,$(subst /, ,$(s))))" "$(s)";) \
	  printf '  {"directory": "%s", "file": "%s", "command": "%s %s -c %s"}\n' "$$PWD" "$(COMMON)" "$(CXX)" "$(flags_common)" "$(COMMON)"; \
	  echo ']'; } > compile_commands.json
	@echo "compile_commands.json written"

clean:
	rm -rf build

# ---- plots (matplotlib in a local virtualenv) ----
.venv:
	python3 -m venv .venv && .venv/bin/pip install -q matplotlib

# make plot                              results/csv/*.csv -> results/plots/
# make plot IN=<dir> OUT=<report-dir>    all *.csv in <dir> -> <report-dir>/plots/
# make plot IN="a.csv b-*.csv" OUT=...   only the given files (globs are expanded)
# Missing directories are created; files with the same names are overwritten.
IN  ?= results/csv
OUT ?= results
plot: .venv
	.venv/bin/python scripts/plot.py --input $(IN) --output "$(OUT)"

# make reports                 all stage reports -> reports/<stage>/plots/ (scripts/reports.sh)
# make reports STAGES="02 05"  only these stages
reports: .venv
	scripts/reports.sh $(STAGES)
