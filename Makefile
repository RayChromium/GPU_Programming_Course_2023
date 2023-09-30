# compiler
NVCC = nvcc
# archetechture
ARCH = sm_70
#
# source file
SRC = template.cu
# output excutable file
EXE = resultexe
# Specify the libary for linking, in this case math libs
LIBS = -lm

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) $(SRC) -o $(EXE) $(LIBS)

clean:
	rm -f $(EXE)
