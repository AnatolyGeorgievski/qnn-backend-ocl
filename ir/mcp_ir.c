/**
 * mcp_ir.c — разбор имён ops / одной строки ассемблера (без ICD, без GPU).
 */
#include "mcp_ir.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static int eq(const char *a, size_t n, const char *lit)
{
	size_t m = strlen(lit);
	return n == m && memcmp(a, lit, n) == 0;
}

uint8_t mcp_ir_op_parse(const char *s, size_t n)
{
	if (!s || !n)
		return GGML_OP_COUNT;
	if (eq(s, n, "NOP")) return GGML_OP_NONE;
	if (eq(s, n, "VIEW")) return GGML_OP_VIEW;
	if (eq(s, n, "CONT")) return GGML_OP_CONT;
	if (eq(s, n, "PERMUTE")) return GGML_OP_PERMUTE;
	if (eq(s, n, "MUL_MAT")) return GGML_OP_MUL_MAT;
	if (eq(s, n, "ADD")) return GGML_OP_ADD;
	if (eq(s, n, "MUL")) return GGML_OP_MUL;
	if (eq(s, n, "RMS_NORM")) return GGML_OP_RMS_NORM;
	if (eq(s, n, "SILU")) return GGML_OP_UNARY;
	if (eq(s, n, "NEG")) return GGML_OP_IR_NEG;
	if (eq(s, n, "SUB")) return GGML_OP_SUB;
	if (eq(s, n, "DIV")) return GGML_OP_DIV;
	if (eq(s, n, "BARRIER")) return GGML_OP_BARRIER;
	return GGML_OP_COUNT;
}

uint8_t mcp_ir_dtype_parse(const char *s, size_t n)
{
	if (!s || !n) return GGML_TYPE_COUNT;
	if (eq(s, n, "f32") || eq(s, n, "F32")) return GGML_TYPE_F32;
	if (eq(s, n, "bf16") || eq(s, n, "BF16")) return GGML_TYPE_BF16;
	if (eq(s, n, "f16") || eq(s, n, "F16")) return GGML_TYPE_F16;
	if (eq(s, n, "q4_K") || eq(s, n, "Q4_K")) return GGML_TYPE_Q4_K;
	if (eq(s, n, "q6_K") || eq(s, n, "Q6_K")) return GGML_TYPE_Q6_K;
	return GGML_TYPE_COUNT;
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
	if (out->op == GGML_OP_COUNT)
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
				if (dt != GGML_TYPE_COUNT)
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
		      const size_t ne[GGML_MAX_DIMS])
{
	void *p;
	size_t page;
	int i;

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
	__builtin_bzero(s, sizeof *s);
	s->base = p;
	s->bytes = bytes;
	s->dtype = dtype;
	s->flags = MCP_IR_SLOT_HOST_PINNED | MCP_IR_SLOT_MMAP_SHARED;
	if (ne) {
		for (i = 0; i < GGML_MAX_DIMS; i++)
			s->ne[i] = ne[i];
	} else {
		mcp_ir_ne_set(s->ne, 1, 1, 1, 1);
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


/** ne уже задан через mcp_ir_ne_set. */
static size_t slot_n(const struct mcp_ir_slot *s)
{
	size_t n = s->ne[0];
	int i;

	for (i = 1; i < GGML_MAX_DIMS; i++)
		n *= s->ne[i];
	return n;
}


/**
 * mcp_ir_host_mul_mat: C = A·B на хосте, f32.
 * A (M×K), B (K×N), C (M×N); ne: M=a->ne[0], K=a->ne[1], N=b->ne[1].
 * Раскладка column-major: A[i + k·M], B[k + j·K], C[i + j·M].
 * Готово: 0; -1 нет слота/base, -2 не f32, -3 b->ne[0]≠K, -4 ne dst ≠ (M,N).
 */
int mcp_ir_host_mul_mat(struct mcp_ir_slot *dst,
			const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	size_t M, K, N, i, j, k;
	float *A, *B, *C;

	if (!dst || !a || !b || !dst->base || !a->base || !b->base)
		return -1;
	if (dst->dtype != GGML_TYPE_F32 || a->dtype != GGML_TYPE_F32 || b->dtype != GGML_TYPE_F32)
		return -2;
	M = a->ne[0];
	K = a->ne[1];
	if (b->ne[0] != K)
		return -3;
	N = b->ne[1];
	if (dst->ne[0] != M || dst->ne[1] != N)
		return -4;
	A = (float *)a->base;
	B = (float *)b->base;
	C = (float *)dst->base;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			float s = 0.f;
			for (k = 0; k < K; k++)
				s += A[i + k * M] * B[k + j * K];
			C[i + j * M] = s;
		}
	}
	return 0;
}

int mcp_ir_host_binop(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	size_t n, i;
	float *od, *oa, *ob;

	if (!dst || !a || !b || !dst->base || !a->base || !b->base)
		return -1;
	if (dst->dtype != GGML_TYPE_F32 || a->dtype != GGML_TYPE_F32 || b->dtype != GGML_TYPE_F32)
		return -2;
	n = slot_n(dst);
	{
		size_t na = slot_n(a), nb = slot_n(b);
		if (na < n) n = na;
		if (nb < n) n = nb;
	}
	od = (float *)dst->base;
	oa = (float *)a->base;
	ob = (float *)b->base;
	if (op == GGML_OP_ADD) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] + ob[i];
		return 0;
	}
	if (op == GGML_OP_MUL) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] * ob[i];
		return 0;
	}
	if (op == GGML_OP_SUB) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] - ob[i];
		return 0;
	}
	if (op == GGML_OP_DIV) {
		for (i = 0; i < n; i++)
			od[i] = oa[i] / ob[i];
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
	if (dst->dtype != GGML_TYPE_F32 || a->dtype != GGML_TYPE_F32)
		return -2;
	n = slot_n(dst);
	{
		size_t na = slot_n(a);
		if (na < n) n = na;
	}
	od = (float *)dst->base;
	oa = (float *)a->base;
	if (op == GGML_OP_UNARY) {
		for (i = 0; i < n; i++) {
			float x = oa[i];
			od[i] = x / (1.f + expf(-x));
		}
		return 0;
	}
	if (op == GGML_OP_IR_NEG) {
		for (i = 0; i < n; i++)
			od[i] = -oa[i];
		return 0;
	}
	if (op == GGML_OP_RMS_NORM) {
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

static mcp_ir_binop_fn ll_binop;
static mcp_ir_unary_fn ll_unary;
static mcp_ir_mul_mat_fn ll_mul_mat;

static void ll_ensure(void)
{
	if (!ll_binop)
		ll_binop = mcp_ir_host_binop;
	if (!ll_unary)
		ll_unary = mcp_ir_host_unary;
	if (!ll_mul_mat)
		ll_mul_mat = mcp_ir_host_mul_mat;
}

void mcp_ir_ll_set(mcp_ir_binop_fn b, mcp_ir_unary_fn u, mcp_ir_mul_mat_fn m)
{
	ll_binop = b ? b : mcp_ir_host_binop;
	ll_unary = u ? u : mcp_ir_host_unary;
	ll_mul_mat = m ? m : mcp_ir_host_mul_mat;
}

void mcp_ir_ll_host(void)
{
	ll_binop = mcp_ir_host_binop;
	ll_unary = mcp_ir_host_unary;
	ll_mul_mat = mcp_ir_host_mul_mat;
}

int mcp_ir_host_phrase(struct mcp_ir_slot *ring, uint16_t cap, uint16_t *head,
		       const char *const *lines, int nlines)
{
	int i;

	ll_ensure();

	if (!ring || !head || (!lines && nlines > 0) || nlines < 0)
		return -1;
	for (i = 0; i < nlines; i++) {
		struct mcp_ir_node n;
		int rc, sa, sb;

		rc = mcp_ir_node_from_line(lines[i], &n);
		if (rc == -1)
			continue;
		if (rc != 0)
			return rc;
		if (n.op == GGML_OP_NONE || n.op == GGML_OP_BARRIER)
			continue;
		if (*head >= cap)
			return -4;
		if (n.op == GGML_OP_ADD || n.op == GGML_OP_MUL || n.op == GGML_OP_SUB || n.op == GGML_OP_DIV) {
			if (n.n_src < 2)
				return -5;
			sa = mcp_ir_resolve_src(n.src[0], *head);
			sb = mcp_ir_resolve_src(n.src[1], *head);
			if (sa < 0 || sb < 0 || sa >= (int)*head || sb >= (int)*head)
				return -6;
			rc = ll_binop(n.op, &ring[*head], &ring[sa], &ring[sb]);
			if (rc)
				return rc;
			(*head)++;
			continue;
		}
		if (n.op == GGML_OP_UNARY || n.op == GGML_OP_RMS_NORM || n.op == GGML_OP_IR_NEG) {
			if (n.n_src < 1)
				return -5;
			sa = mcp_ir_resolve_src(n.src[0], *head);
			if (sa < 0 || sa >= (int)*head)
				return -6;
			rc = ll_unary(n.op, &ring[*head], &ring[sa]);
			if (rc)
				return rc;
			(*head)++;
			continue;
		}
		return -3;
	}
	return 0;
}

static int dim_sep(const char **q)
{
	const unsigned char *s = (const unsigned char *)*q;

	if (*s == 'x' || *s == 'X') {
		*q += 1;
		return 1;
	}
	/* · */
	if (s[0] == 0xC2 && s[1] == 0xB7) {
		*q += 2;
		return 1;
	}
	return 0;
}

/** \brief Разбор ссылки на слот
 *  \param[in] tok строка ссылки на слот
 *  \param[out] shape массив размености, всегда задан. По умолчанию (1, 1, 1, 1)
 *  \param[out] has_shape флаг наличия размеров слота
 *  \return идентификатор слота
 */
static int slot_ref(const char *tok, size_t *shape, int *has_shape)
{
	const char *q;
	int id;

	mcp_ir_ne_set(shape, 1, 1, 1, 1);
	if (has_shape)
		*has_shape = 0;
	if (!tok || tok[0] != '$' || tok[1] < '0' || tok[1] > '9')
		return -1;
	id = tok[1] - '0';
	if (tok[2] == '\0')
		return id;
	if (tok[2] != '(')
		return -1;
	q = tok + 3;
	shape[0] = strtoul(q, (char **)&q, 10);
	if (dim_sep(&q)) {
		shape[1] = strtoul(q, (char **)&q, 10);
		if (dim_sep(&q))
			shape[2] = strtoul(q, (char **)&q, 10);
	}
	if (*q != ')' || q[1] != '\0' || shape[0] == 0) {
		mcp_ir_ne_set(shape, 1, 1, 1, 1);
		return -1;
	}
	if (has_shape)
		*has_shape = 1;
	return id;
}

static int slot_id(const char *tok)
{
	return (int)strtoul(tok + 1, NULL, 10);
}

/** shape задан через mcp_ir_ne_set: лишние оси уже 1. */
static size_t shape_n(const size_t shape[GGML_MAX_DIMS])
{
	size_t n = 1;
	for (int i = 0; i < GGML_MAX_DIMS; i++)
		n *= shape[i];
	return n;
}

static void shape_txt(const size_t sh[GGML_MAX_DIMS], char *b, size_t n)
{
	int i, u;
	u = snprintf(b, n, "%zu", sh[0]);
	for (i = 1; i < GGML_MAX_DIMS && u > 0 && (size_t)u < n; i++) {
		if (sh[i] > 1)
			u += snprintf(b + u, n - (size_t)u, "·%zu", sh[i]);
	}
}

static int ensure_slot(struct mcp_ir_slot *s, size_t nfloat)
{
	const size_t sh[GGML_MAX_DIMS] = {1, 1, 1, 1};

	if (s->base)
		return 0;
	if (mcp_ir_slot_alloc(s, nfloat * sizeof(float), GGML_TYPE_F32, sh))
		return -1;
	s->ne[0] = nfloat;
	return 0;
}

static int _append(char *out, size_t cap, size_t *used, const char *s)
{
	size_t n = strlen(s);
	if (*used + n + 1 > cap)
		return -1;
	memcpy(out + *used, s, n + 1);
	*used += n;
	return 0;
}

int mcp_ir_say(const char *speech, char *out, size_t cap)
{
	ll_ensure();
	struct mcp_ir_slot slot[10];
	char line[512];
	const char *p;
	size_t used = 0;
	int rc = 0;
	int i;

	out[0] = '\0';
	memset(slot, 0, sizeof slot);
	p = speech;
	while (*p) {
		const char *nl = strchr(p, '\n');
		size_t len = nl ? (size_t)(nl - p) : strlen(p);
		char *tok[16];
		int nt = 0;
		char *save = NULL;
		int op_at = -1;

		if (len >= sizeof line)
			len = sizeof line - 1;
		memcpy(line, p, len);
		line[len] = '\0';
		p = nl ? nl + 1 : p + len;
		{
			char *q = line;
			while (*q == ' ' || *q == '\t')
				q++;
			if (*q == '#' || *q == '\0')
				continue;
		}
		for (save = line; nt < 16; ) {
			char *t = strtok(save, " \t\r");
			save = NULL;
			if (!t)
				break;
			tok[nt++] = t;
		}
		if (nt == 0)
			continue;
		if (strcmp(tok[0], "SET") == 0) {
			int id, has = 0;
			size_t sh[GGML_MAX_DIMS] = {1, 1, 1, 1};
			float *f;
			int k, nfloat;
			char buf[128];
			if (nt < 3) { rc = -2; break; }
			id = slot_ref(tok[1], sh, &has);
			nfloat = nt - 2;
			if (id < 0 || id > 9) { rc = -2; break; }
			if (has && shape_n(sh) != (size_t)nfloat) {
				snprintf(buf, sizeof buf,
					 "SET $%d status=shape want=%zu got=%d\n",
					 id, shape_n(sh), nfloat);
				if (_append(out, cap, &used, buf)) { rc = -5; break; }
				rc = -2;
				break;
			}
			if (ensure_slot(&slot[id], (size_t)nfloat)) { rc = -4; break; }
			if (has)
				for (k = 0; k < GGML_MAX_DIMS; k++)
					slot[id].ne[k] = sh[k];
			f = slot[id].base;
			for (k = 0; k < nfloat; k++)
				f[k] = strtof(tok[k + 2], NULL);
			continue;
		}
		if (strcmp(tok[0], "GET") == 0 || strcmp(tok[0], "GET.vec") == 0) {
			int has = 0;
			size_t sh[GGML_MAX_DIMS] = {1, 1, 1, 1};
			int id = nt >= 2 ? slot_ref(tok[1], sh, &has) : -1;
			int vec = strcmp(tok[0], "GET.vec") == 0;
			char buf[1024];
			if (id < 0 || id > 9 || !slot[id].base) {
				snprintf(buf, sizeof buf, "%s $%d status=empty\n", tok[0], id < 0 ? -1 : id);
				if (_append(out, cap, &used, buf)) { rc = -5; break; }
				rc = -6;
				continue;
			}
			if (has) {
				int dim;
				for (dim = 0; dim < GGML_MAX_DIMS; dim++) {
					if (sh[dim] != slot[id].ne[dim]) {
						snprintf(buf, sizeof buf,
							 "%s $%d status=shape want=%zu have=%zu\n",
							 tok[0], id, shape_n(sh), slot[id].ne[0]);
						if (_append(out, cap, &used, buf)) { rc = -5; break; }
						rc = -2;
						break;
					}
				}
				if (rc)
					break;
			}
			if (!vec) {
				{
					char shs[64];
					shape_txt(slot[id].ne, shs, sizeof shs);
					snprintf(buf, sizeof buf,
						 "GET $%d(%s) status=ok dtype=f32\n",
						 id, shs);
				}
			} else {
				size_t n = slot_n(&slot[id]);
				float *f = slot[id].base;
				size_t u = 0;
				int wr;
				{
					char shs[64];
					shape_txt(slot[id].ne, shs, sizeof shs);
					wr = snprintf(buf, sizeof buf, "GET.vec $%d(%s)", id, shs);
				}
				if (wr < 0) { rc = -5; break; }
				u = (size_t)wr;
				for (i = 0; i < (int)n && u + 24 < sizeof buf; i++) {
					wr = snprintf(buf + u, sizeof buf - u, " %.6g", (double)f[i]);
					if (wr < 0) break;
					u += (size_t)wr;
				}
				if (u + 2 < sizeof buf) {
					buf[u++] = '\n';
					buf[u] = '\0';
				}
			}
			if (_append(out, cap, &used, buf)) { rc = -5; break; }
			continue;
		}
		for (i = 0; i < nt; i++) {
			size_t L = strlen(tok[i]);
			if (mcp_ir_op_parse(tok[i], L) != GGML_OP_COUNT) {
				op_at = i;
				break;
			}
		}
		if (op_at < 0 || op_at + 1 >= nt) { rc = -3; break; }
		{
			uint8_t op = mcp_ir_op_parse(tok[op_at], strlen(tok[op_at]));
			size_t dsh[GGML_MAX_DIMS] = {1, 1, 1, 1};
			int dhas = 0;
			int dst = slot_ref(tok[op_at + 1], dsh, &dhas);
			int s0 = op_at >= 1 ? slot_id(tok[0]) : -1;
			int s1 = op_at >= 2 ? slot_id(tok[1]) : -1;
			size_t nfloat;
			if (dst < 0 || dst > 9 || s0 < 0) { rc = -3; break; }
			if (!slot[s0].base || (op_at >= 2 && !slot[s1].base)) { rc = -6; break; }
			nfloat = slot_n(&slot[s0]);
			if (op_at >= 2) {
				size_t n1 = slot_n(&slot[s1]);
				if (n1 < nfloat) nfloat = n1;
			}
			if (op != GGML_OP_MUL_MAT) {
				if (dhas && shape_n(dsh) != nfloat) { rc = -2; break; }
				if (ensure_slot(&slot[dst], nfloat)) { rc = -4; break; }
				if (dhas)
					for (i = 0; i < GGML_MAX_DIMS; i++)
						slot[dst].ne[i] = dsh[i];
				else
					mcp_ir_ne_set(slot[dst].ne, nfloat, 1, 1, 1);
			}
			if (op == GGML_OP_ADD || op == GGML_OP_MUL || op == GGML_OP_SUB || op == GGML_OP_DIV) {
				if (s1 < 0) { rc = -3; break; }
				rc = ll_binop(op, &slot[dst], &slot[s0], &slot[s1]);
			} else if (op == GGML_OP_MUL_MAT) {
				size_t M, K, N, need;
				if (s1 < 0) { rc = -3; break; }
				M = slot[s0].ne[0];
				K = slot[s0].ne[1];
				if (slot[s1].ne[0] != K) { rc = -2; break; }
				N = slot[s1].ne[1];
				need = M * N;
				if (dhas && shape_n(dsh) != need) { rc = -2; break; }
				if (!slot[dst].base) {
					size_t sh[GGML_MAX_DIMS] = {1,1,1,1};
					if (mcp_ir_slot_alloc(&slot[dst], need * sizeof(float), GGML_TYPE_F32, sh))
						{ rc = -4; break; }
				}
				mcp_ir_ne_set(slot[dst].ne, M, N, 1, 1);
				if (dhas)
					for (i = 0; i < GGML_MAX_DIMS; i++)
						slot[dst].ne[i] = dsh[i];
				ll_ensure();
				rc = ll_mul_mat(&slot[dst], &slot[s0], &slot[s1]);
			} else if (op == GGML_OP_UNARY || op == GGML_OP_RMS_NORM || op == GGML_OP_IR_NEG) {
				rc = ll_unary(op, &slot[dst], &slot[s0]);
			} else
				rc = -3;
			if (rc) break;
			rc = 0;
		}
	}
	for (i = 0; i < 10; i++)
		mcp_ir_slot_free(&slot[i]);
	return rc;
}

