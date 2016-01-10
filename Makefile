CC = cc
ifeq ($(shell uname -s), Darwin)
	CFLAGS = -O3 -framework Accelerate -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/
else
	CFLAGS = -g --compiler-options "-std=c99"
endif
LDFLAGS = -lblas -lm
FLOAT_IS_DOUBLE = -Dfloat=double -Dcblas_sger=cblas_dger -Dcblas_sgemv=cblas_dgemv -DcublasSgemv=cublasDgemv

all: grad_check main

grad_check:
	mkdir -p bin/
	$(CC) $(CFLAGS) $(FLOAT_IS_DOUBLE) -o bin/grad_check src/grad_check.c src/lstm.c src/util.c src/softmax.c src/matrix.c src/embedding.c src/seq2seq.c src/dense.c $(LDFLAGS)

main:
	$(CC) $(CFLAGS) -o bin/seq2seq src/main.c src/lstm.c src/util.c src/softmax.c src/matrix.c src/embedding.c src/seq2seq.c src/parser.c src/cmdline.c src/dense.c $(LDFLAGS)
