/**
 * mcp_ir.h — MCP+IR · тензорный ассемблер + слоты подкэша (Zero-Copy)
 * Не ICD. Граф → LL → HW. Тела ops — отдельно.
 */
#ifndef GODMINER_MCP_IR_H
#define GODMINER_MCP_IR_H

#include <stddef.h>
#include <stdint.h>

enum mcp_ir_op {
	MCP_IR_NOP = 0,
	MCP_IR_VIEW,
	MCP_IR_CONT,
	MCP_IR_PERMUTE,
	MCP_IR_MUL_MAT,
	MCP_IR_ADD,
	MCP_IR_MUL,
	MCP_IR_RMS_NORM,
	MCP_IR_SILU,
	MCP_IR_BARRIER,
	MCP_IR_OP_COUNT
};

enum mcp_ir_dtype {
	MCP_IR_F32 = 0,
	MCP_IR_BF16,
	MCP_IR_F16,
	MCP_IR_Q4_K,
	MCP_IR_Q6_K,
	MCP_IR_DTYPE_COUNT
};

#define MCP_IR_SLOT_HOST_PINNED    (1u << 0)
#define MCP_IR_SLOT_DEVICE_VISIBLE (1u << 1)
#define MCP_IR_SLOT_MMAP_SHARED    (1u << 2)

/** Слот подкэша: дескриптор без владения копией байт в MCP. */
struct mcp_ir_slot {
	void    *base;       /* host VA; device map отдельно в LL */
	size_t   bytes;
	uint8_t  dtype;      /* mcp_ir_dtype */
	uint8_t  flags;
	uint16_t _pad;
	uint32_t shape[3];   /* ne0,ne1,ne2 — как в ассемблере */
	uint8_t  h128[16];   /* content tag; пусто = ещё не считали */
};

#define MCP_IR_RING 9  /* $1..$9 как set -t */

struct mcp_ir_node {
	uint8_t  op;         /* mcp_ir_op */
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

/** Имя op → код; неизвестно → MCP_IR_OP_COUNT. */
uint8_t mcp_ir_op_parse(const char *s, size_t n);

/** f32|bf16|f16|q4_K|q6_K → dtype; иначе MCP_IR_DTYPE_COUNT. */
uint8_t mcp_ir_dtype_parse(const char *s, size_t n);

/** −k back-ref → индекс слота кольца (head = последний записанный). */
int mcp_ir_resolve_src(int16_t src, uint16_t ring_head);

/** Прямая речь / ассемблерная строка → один node (без malloc). */
int mcp_ir_node_from_line(const char *line, struct mcp_ir_node *out);

/** Подкэш: anonymous mmap слот (HOST_PINNED|MMAP_SHARED). 0 ок. */
int mcp_ir_slot_alloc(struct mcp_ir_slot *s, size_t bytes, uint8_t dtype,
		      const uint32_t shape[3]);
void mcp_ir_slot_free(struct mcp_ir_slot *s);

/**
 * Host LL для учёбы: ADD/MUL f32 поэлементно по min(bytes)/4.
 * src[0], src[1] → dst. Не ICD. 0 ок.
 */
int mcp_ir_host_binop(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);

/** Host unary f32: SILU / RMS_NORM (eps 1e-5). */
int mcp_ir_host_unary(uint8_t op, struct mcp_ir_slot *dst,
		      const struct mcp_ir_slot *a);

#endif
