/**
 * mcp_ir.c — разбор имён ops / одной строки ассемблера (без ICD, без GPU).
 */
#include "mcp_ir.h"
#include <string.h>
#include <stdlib.h>

static int eq(const char *a, size_t n, const char *lit)
{
	size_t m = strlen(lit);
	return n == m && memcmp(a, lit, n) == 0;
}

uint8_t mcp_ir_op_parse(const char *s, size_t n)
{
	if (!s || !n)
		return MCP_IR_OP_COUNT;
	if (eq(s, n, "NOP")) return MCP_IR_NOP;
	if (eq(s, n, "VIEW")) return MCP_IR_VIEW;
	if (eq(s, n, "CONT")) return MCP_IR_CONT;
	if (eq(s, n, "PERMUTE")) return MCP_IR_PERMUTE;
	if (eq(s, n, "MUL_MAT")) return MCP_IR_MUL_MAT;
	if (eq(s, n, "ADD")) return MCP_IR_ADD;
	if (eq(s, n, "MUL")) return MCP_IR_MUL;
	if (eq(s, n, "RMS_NORM")) return MCP_IR_RMS_NORM;
	if (eq(s, n, "SILU")) return MCP_IR_SILU;
	if (eq(s, n, "BARRIER")) return MCP_IR_BARRIER;
	return MCP_IR_OP_COUNT;
}

uint8_t mcp_ir_dtype_parse(const char *s, size_t n)
{
	if (!s || !n) return MCP_IR_DTYPE_COUNT;
	if (eq(s, n, "f32") || eq(s, n, "F32")) return MCP_IR_F32;
	if (eq(s, n, "bf16") || eq(s, n, "BF16")) return MCP_IR_BF16;
	if (eq(s, n, "f16") || eq(s, n, "F16")) return MCP_IR_F16;
	if (eq(s, n, "q4_K") || eq(s, n, "Q4_K")) return MCP_IR_Q4_K;
	if (eq(s, n, "q6_K") || eq(s, n, "Q6_K")) return MCP_IR_Q6_K;
	return MCP_IR_DTYPE_COUNT;
}

int mcp_ir_resolve_src(int16_t src, uint16_t ring_head)
{
	int k;
	if (src >= 0)
		return src;
	k = -src; /* $k */
	if (k < 1 || k > MCP_IR_RING)
		return -1;
	/* $1 = последний результат = ring_head-1 */
	return (int)((ring_head + MCP_IR_RING - (unsigned)k) % MCP_IR_RING);
}

/* грубый разбор: OP ... — только код op; операнды позже */
int mcp_ir_node_from_line(const char *line, struct mcp_ir_node *out)
{
	const char *p, *e;
	size_t n;

	if (!line || !out)
		return -1;
	memset(out, 0, sizeof *out);
	while (*line == ' ' || *line == '\t')
		line++;
	if (*line == '#' || *line == '\0')
		return -1;
	p = line;
	while (*p && *p != ' ' && *p != '\t')
		p++;
	n = (size_t)(p - line);
	out->op = mcp_ir_op_parse(line, n);
	if (out->op == MCP_IR_OP_COUNT)
		return -2;
	/* dtype после первой точки в токене вида r1.f32(…); back-ref $n */
	for (e = p; *e; e++) {
		if (e[0] == '.' && out->dtype == 0) {
			const char *d = e + 1;
			const char *q = d;
			while (*q && *q != '(' && *q != ' ' && *q != '\t')
				q++;
			{
				uint8_t dt = mcp_ir_dtype_parse(d, (size_t)(q - d));
				if (dt != MCP_IR_DTYPE_COUNT)
					out->dtype = dt;
			}
			/* shape parse: (ne0xne1xne2) */
			if (*q == '(') {
				unsigned a=0,b=0,c=0;
				const char *p2 = q + 1;
				a = (unsigned)strtoul(p2, (char **)&p2, 10);
				if (*p2 == 'x' || *p2 == 'X') {
					b = (unsigned)strtoul(p2 + 1, (char **)&p2, 10);
					if (*p2 == 'x' || *p2 == 'X')
						c = (unsigned)strtoul(p2 + 1, (char **)&p2, 10);
				}
				out->out_shape[0]=a; out->out_shape[1]=b; out->out_shape[2]=c;
			}
		}
		if (e[0] == '$' && e[1] >= '1' && e[1] <= '9' && out->n_src < 4) {
			out->src[out->n_src++] = (int16_t)(-(e[1] - '0'));
		}
	}
	return 0;
}

#include <sys/mman.h>
#include <unistd.h>

int mcp_ir_slot_alloc(struct mcp_ir_slot *s, size_t bytes, uint8_t dtype,
		      const uint32_t shape[3])
{
	void *p;
	size_t page;

	if (!s || !bytes)
		return -1;
	page = (size_t)sysconf(_SC_PAGESIZE);
	if (page < 4096)
		page = 4096;
	bytes = (bytes + page - 1) & ~(page - 1);
	p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
		 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED)
		return -2;
	memset(s, 0, sizeof *s);
	s->base = p;
	s->bytes = bytes;
	s->dtype = dtype;
	s->flags = MCP_IR_SLOT_HOST_PINNED | MCP_IR_SLOT_MMAP_SHARED;
	if (shape) {
		s->shape[0] = shape[0];
		s->shape[1] = shape[1];
		s->shape[2] = shape[2];
	}
	return 0;
}

void mcp_ir_slot_free(struct mcp_ir_slot *s)
{
	if (!s || !s->base || !s->bytes)
		return;
	munmap(s->base, s->bytes);
	s->base = NULL;
	s->bytes = 0;
}

int mcp_ir_host_binop(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	size_t n, i;
	float *od, *oa, *ob;

	if (!dst || !a || !b || !dst->base || !a->base || !b->base)
		return -1;
	if (dst->dtype != MCP_IR_F32 || a->dtype != MCP_IR_F32 || b->dtype != MCP_IR_F32)
		return -2;
	n = dst->bytes;
	if (a->bytes < n) n = a->bytes;
	if (b->bytes < n) n = b->bytes;
	n /= sizeof(float);
	od = (float *)dst->base;
	oa = (float *)a->base;
	ob = (float *)b->base;
	if (op == MCP_IR_ADD) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] + ob[i];
		return 0;
	}
	if (op == MCP_IR_MUL) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] * ob[i];
		return 0;
	}
	return -3;
}


#include <math.h>

int mcp_ir_host_unary(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a)
{
	size_t n, i;
	float *od, *oa;
	double sum;

	if (!dst || !a || !dst->base || !a->base)
		return -1;
	if (dst->dtype != MCP_IR_F32 || a->dtype != MCP_IR_F32)
		return -2;
	n = dst->bytes < a->bytes ? dst->bytes : a->bytes;
	n /= sizeof(float);
	od = (float *)dst->base;
	oa = (float *)a->base;
	if (op == MCP_IR_SILU) {
		for (i = 0; i < n; i++) {
			float x = oa[i];
			od[i] = x / (1.f + expf(-x));
		}
		return 0;
	}
	if (op == MCP_IR_RMS_NORM) {
		sum = 0;
		for (i = 0; i < n; i++)
			sum += (double)oa[i] * (double)oa[i];
		{
			float scale = (float)(1.0 / sqrt(sum / (double)n + 1e-5));
			for (i = 0; i < n; i++)
				od[i] = oa[i] * scale;
		}
		return 0;
	}
	return -3;
}
