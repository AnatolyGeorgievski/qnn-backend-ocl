/**
 * mcp_ir.h — MCP+IR · тензорный ассемблер + слоты подкэша (Zero-Copy)
 * Не ICD. Граф → LL → HW. Тела ops — отдельно.
 * Типы и ops — qnn.h (ggml 10.0.1 + группа QNN).
 */
#ifndef GODMINER_MCP_IR_H
#define GODMINER_MCP_IR_H

#define QNN_IDS_ONLY
#include "qnn.h"

#include <stddef.h>
#include <stdint.h>

#define MCP_IR_SLOT_HOST_PINNED    (1u << 0)
#define MCP_IR_SLOT_DEVICE_VISIBLE (1u << 1)
#define MCP_IR_SLOT_MMAP_SHARED    (1u << 2)

/**
 * Заполнить ne. Лишняя ось равна 1, не 0.
 * Вектор длины n: mcp_ir_ne_set(ne, n, 1, 1, 1).
 * Ноль в оси — пустой тензор, его не подменяют единицей.
 */
static inline void mcp_ir_ne_set(size_t ne[GGML_MAX_DIMS],
		size_t ne0, size_t ne1, size_t ne2, size_t ne3)
{
	const size_t src[GGML_MAX_DIMS] = { ne0, ne1, ne2, ne3 };
	int i;
	for (i = 0; i < GGML_MAX_DIMS; i++)
		ne[i] = src[i];
}

/** Слот подкэша: дескриптор без владения копией байт в MCP. */
struct mcp_ir_slot {
	void    *base;       /* host VA; device map отдельно в LL */
	size_t   bytes;
	uint8_t  dtype;      /* ggml_type */
	uint8_t  flags;
	uint16_t _pad;
	size_t   ne[GGML_MAX_DIMS]; /* как ggml_tensor.ne: ne[i]==1 при i>=n_dim */
	uint8_t  h128[16];   /* content tag; пусто = ещё не считали */
};

#define MCP_IR_RING 9  /* $1..$9 как set -t */

struct mcp_ir_node {
	uint8_t  op;         /* ggml_op / QNN */
	uint8_t  dtype;      /* result dtype */
	uint8_t  n_src;
	uint8_t  _pad;
	int16_t  src[4];     /* индекс слота или −k = back-ref $k */
	uint32_t out_shape[3];
};

struct mcp_ir_graph {
	struct mcp_ir_slot slot[MCP_IR_RING + 8]; /* кольцо + m* запас */
	uint16_t n_slot;
	uint16_t n_node;
	struct mcp_ir_node *node; /* владелец снаружи / arena */
};

/** Имя op → код; неизвестно → GGML_OP_COUNT. SILU → GGML_OP_UNARY. */
uint8_t mcp_ir_op_parse(const char *s, size_t n);

/** f32|bf16|f16|q4_K|q6_K → dtype; иначе GGML_TYPE_COUNT. */
uint8_t mcp_ir_dtype_parse(const char *s, size_t n);

/** −k back-ref → индекс слота кольца (head = последний записанный). */
int mcp_ir_resolve_src(int16_t src, uint16_t ring_head);

/** Прямая речь / ассемблерная строка → один node (без malloc). */
int mcp_ir_node_from_line(const char *line, struct mcp_ir_node *out);

/** Подкэш: anonymous mmap слот (HOST_PINNED|MMAP_SHARED). 0 ок. */
int mcp_ir_slot_alloc(struct mcp_ir_slot *s, size_t bytes, uint8_t dtype,
		      const size_t ne[GGML_MAX_DIMS]);
void mcp_ir_slot_free(struct mcp_ir_slot *s);

/**
 * Host LL для учёбы: ADD/MUL f32 поэлементно по min(bytes)/4.
 * src[0], src[1] → dst. Не ICD. 0 ок.
 */
int mcp_ir_host_mul_mat(struct mcp_ir_slot *dst,
			const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);

int mcp_ir_host_binop(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);

/** Host unary f32: SILU (GGML_OP_UNARY) / RMS_NORM (eps 1e-5). */
int mcp_ir_host_unary(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a);

/** Фраза на кольце. head — сколько слотов уже живы.
 *  Строка пишет следующий слот и сдвигает head.
 *  ADD, MUL, SILU, RMS_NORM. Пустая строка и BARRIER проходят. Иной op — −3.
 */
int mcp_ir_host_phrase(struct mcp_ir_slot *ring, uint16_t cap, uint16_t *head,
		       const char *const *lines, int nlines);

/**
 * Прямая речь: SET, строки графа, GET / GET.vec.
 * GET — статус слота. GET.vec — вектор f32. Ответ дописывается в out.
 * speech и out заданы, cap не меньше 2, в out помещается cap байт.
 * 0 — все GET попали в живой слот.
 */
int mcp_ir_say(const char *speech, char *out, size_t cap);

/** LL: host по умолчанию. mcp_ir_ll_ocl("B60") — ops на устройстве. */
void mcp_ir_ll_host(void);
int mcp_ir_ll_ocl(const char *device_substr);
int mcp_ir_ocl_binop(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);
int mcp_ir_ocl_unary(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a);
int mcp_ir_ocl_mul_mat(struct mcp_ir_slot *dst,
			const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);

typedef int (*mcp_ir_binop_fn)(uint8_t, struct mcp_ir_slot *,
			      const struct mcp_ir_slot *, const struct mcp_ir_slot *);
typedef int (*mcp_ir_unary_fn)(uint8_t, struct mcp_ir_slot *,
			      const struct mcp_ir_slot *);
typedef int (*mcp_ir_mul_mat_fn)(struct mcp_ir_slot *,
			 const struct mcp_ir_slot *, const struct mcp_ir_slot *);
void mcp_ir_ll_set(mcp_ir_binop_fn b, mcp_ir_unary_fn u, mcp_ir_mul_mat_fn m);

#endif
