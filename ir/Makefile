# O0 host IR smoke — без ICD
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11

.PHONY: ir clean
mcp_ir.o: mcp_ir.c mcp_ir.h
	$(CC) $(CFLAGS) -c -o $@ mcp_ir.c

mcp_ir_smoke: mcp_ir_smoke.c mcp_ir.o
	$(CC) $(CFLAGS) -o $@ mcp_ir_smoke.c mcp_ir.o -lm

ir: mcp_ir_smoke
	./mcp_ir_smoke

clean:
	rm -f mcp_ir.o mcp_ir_smoke
