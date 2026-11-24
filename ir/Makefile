# O0 host IR smoke — без ICD; ocl LL — тонкий OpenCL
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
QNN_INC ?= $(HOME)/QuantumNN
CFLAGS += -I$(QNN_INC) -I.
OPENCL_LIBS ?= -lOpenCL

.PHONY: ir ir-ocl clean
mcp_ir.o: mcp_ir.c mcp_ir.h
	$(CC) $(CFLAGS) -c -o $@ mcp_ir.c

mcp_ir_ocl.o: mcp_ir_ocl.c mcp_ir.h
	$(CC) $(CFLAGS) -c -o $@ mcp_ir_ocl.c

mcp_ir_smoke: mcp_ir_smoke.c mcp_ir.o
	$(CC) $(CFLAGS) -o $@ mcp_ir_smoke.c mcp_ir.o -lm

mcp_ir_say: mcp_ir_say.c mcp_ir.o
	$(CC) $(CFLAGS) -o $@ mcp_ir_say.c mcp_ir.o -lm

mcp_ir_say_ocl: mcp_ir_say_ocl.c mcp_ir.o mcp_ir_ocl.o
	$(CC) $(CFLAGS) -o $@ mcp_ir_say_ocl.c mcp_ir.o mcp_ir_ocl.o -lm $(OPENCL_LIBS)

mcp_ir_say_stdin: mcp_ir_say_stdin.c mcp_ir.o
	$(CC) $(CFLAGS) -o $@ mcp_ir_say_stdin.c mcp_ir.o -lm

ir: mcp_ir_smoke mcp_ir_say
	./mcp_ir_smoke
	./mcp_ir_say

ir-ocl: mcp_ir_say_ocl
	./mcp_ir_say_ocl

clean:
	rm -f mcp_ir.o mcp_ir_ocl.o mcp_ir_smoke mcp_ir_say mcp_ir_say_ocl mcp_ir_say_stdin
