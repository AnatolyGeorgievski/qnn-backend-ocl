/* дым MCP+IR: разбор строк ассемблера без ICD */
#include "mcp_ir.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fail;

static void check(const char *line, uint8_t want_op, int want_srcs)
{
	struct mcp_ir_node n;
	int rc = mcp_ir_node_from_line(line, &n);
	if (rc != 0 || n.op != want_op || n.n_src != want_srcs) {
		printf("FAIL  %s  rc=%d op=%u n_src=%u\n", line, rc, n.op, n.n_src);
		fail++;
	} else {
		printf("OK    %s\n", line);
	}
}

int main(void)
{
	check("MUL_MAT   r5.bf16(384x128x4)   r1.f32(384x365x28)", GGML_OP_MUL_MAT, 0);
	check("SILU      $2", GGML_OP_UNARY, 1);
	check("ADD       $1  $3", GGML_OP_ADD, 2);
	check("BARRIER", GGML_OP_BARRIER, 0);
	{
		struct mcp_ir_node n;
		if (mcp_ir_node_from_line("MUL_MAT   r5.bf16(384x128x4)  $1", &n) ||
		    n.op != GGML_OP_MUL_MAT || n.dtype != GGML_TYPE_BF16 || n.n_src != 1) {
			printf("FAIL  dtype from line op=%u dt=%u ns=%u\n", n.op, n.dtype, n.n_src);
			fail++;
		} else printf("OK    dtype from line\n");
		if (mcp_ir_resolve_src(-1, 3) != 2) { printf("FAIL  resolve $1\n"); fail++; }
		else printf("OK    resolve $1→ring\n");
	}
	{
		struct mcp_ir_node n;
		int rc = mcp_ir_node_from_line("NOPE", &n);
		if (rc != -2) { printf("FAIL  NOPE should be -2 got %d\n", rc); fail++; }
		else printf("OK    NOPE rejected\n");
	}
	/* Zero-Copy слот + host ADD */
	{
		struct mcp_ir_slot a, b, d;
		size_t ne[GGML_MAX_DIMS];
		float *fa, *fb, *fd;
		mcp_ir_ne_set(ne, 4, 1, 1, 1);
		if (mcp_ir_slot_alloc(&a, 16 * sizeof(float), GGML_TYPE_F32, ne) ||
		    mcp_ir_slot_alloc(&b, 16 * sizeof(float), GGML_TYPE_F32, ne) ||
		    mcp_ir_slot_alloc(&d, 16 * sizeof(float), GGML_TYPE_F32, ne)) {
			printf("FAIL  slot_alloc\n"); fail++;
		} else {
			fa = a.base; fb = b.base; fd = d.base;
			fa[0] = 2.f; fb[0] = 3.f;
			if (mcp_ir_host_binop(GGML_OP_ADD, &d, &a, &b) || fd[0] != 5.f) {
				printf("FAIL  host ADD got %g\n", (double)fd[0]); fail++;
			} else
				printf("OK    slot mmap + host ADD\n");
			fa[0]=2.f; fb[0]=4.f;
			if (mcp_ir_host_binop(GGML_OP_MUL, &d, &a, &b) || fd[0] != 8.f) {
				printf("FAIL  host MUL\n"); fail++;
			} else printf("OK    host MUL\n");
			fa[0]=0.f;
			if (mcp_ir_host_unary(GGML_OP_UNARY, &d, &a) || fabsf(fd[0]-0.f)>1e-6f) {
				printf("FAIL  host SILU\n"); fail++;
			} else printf("OK    host SILU\n");
			fa[0]=3.f; fa[1]=4.f;
			if (mcp_ir_host_unary(GGML_OP_RMS_NORM, &d, &a)) {
				printf("FAIL  host RMS\n"); fail++;
			} else printf("OK    host RMS\n");
			mcp_ir_slot_free(&a); mcp_ir_slot_free(&b); mcp_ir_slot_free(&d);
		}
	}
	/* фраза: 3 * 2 + 1, затем silu */
	{
		struct mcp_ir_slot ring[6];
		size_t ne[GGML_MAX_DIMS];
		uint16_t head = 3;
		mcp_ir_ne_set(ne, 1, 1, 1, 1);
		const char *phrase[] = {
			"MUL $1 $2",
			"ADD $1 $4",
			"SILU $1",
		};
		float *last;
		float want;
		int k;
		for (k = 0; k < 6; k++) {
			if (mcp_ir_slot_alloc(&ring[k], sizeof(float), GGML_TYPE_F32, ne)) {
				printf("FAIL  phrase alloc\n");
				fail++;
				break;
			}
		}
		if (k == 6) {
			((float *)ring[0].base)[0] = 1.f;
			((float *)ring[1].base)[0] = 2.f;
			((float *)ring[2].base)[0] = 3.f;
			if (mcp_ir_host_phrase(ring, 6, &head, phrase, 3) || head != 6) {
				printf("FAIL  phrase rc head=%u\n", head);
				fail++;
			} else {
				last = ring[5].base;
				want = 7.f / (1.f + expf(-7.f));
				if (fabsf(last[0] - want) > 1e-4f) {
					printf("FAIL  phrase got %g want %g\n",
					       (double)last[0], (double)want);
					fail++;
				} else
					printf("OK    phrase 3*2+1 silu = %g\n", (double)last[0]);
			}
			for (k = 0; k < 6; k++)
				mcp_ir_slot_free(&ring[k]);
		}
	}
	printf("=== mcp_ir smoke fail=%d ===\n", fail);
	return fail ? 1 : 0;
}
