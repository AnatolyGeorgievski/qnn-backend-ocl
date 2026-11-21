/**
 * mcp_ir_ocl.c — LL на OpenCL: ADD/MUL/SUB/DIV/SILU/RMS/NEG/MUL_MAT f32.
 * Устройство ищется подстрокой имени (B60). Не ICD QNN — тонкий транспорт.
 */
#include "mcp_ir.h"
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static cl_context ctx;
static cl_command_queue q;
static cl_program prog;
static cl_kernel k_add, k_mul, k_sub, k_div, k_silu, k_rms, k_neg, k_mul_mat;
static int ready;
int mcp_ir_ocl_binop(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);
int mcp_ir_ocl_unary(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a);
int mcp_ir_ocl_mul_mat(struct mcp_ir_slot *dst,
		       const struct mcp_ir_slot *a, const struct mcp_ir_slot *b);

static int dbg_mul_mat(void)
{
	const char *d = getenv("QNN_DBG");
	return d && strstr(d, "mul_mat");
}

static const char src[] =
	"__kernel void ir_add(__global const float *a, __global const float *b,\n"
	"                     __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) d[i] = a[i] + b[i];\n"
	"}\n"
	"__kernel void ir_mul(__global const float *a, __global const float *b,\n"
	"                     __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) d[i] = a[i] * b[i];\n"
	"}\n"
	"__kernel void ir_silu(__global const float *a, __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) { float x = a[i]; d[i] = x / (1.f + exp(-x)); }\n"
	"}\n"
	"__kernel void ir_rms(__global const float *a, __global float *d, const uint n) {\n"
	"  if (get_global_id(0) != 0) return;\n"
	"  float sum = 0.f;\n"
	"  for (uint i = 0; i < n; i++) sum += a[i] * a[i];\n"
	"  float scale = 1.f / sqrt(sum / (float)n + 1e-5f);\n"
	"  for (uint i = 0; i < n; i++) d[i] = a[i] * scale;\n"
	"}\n"
	"__kernel void ir_sub(__global const float *a, __global const float *b,\n"
	"                     __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) d[i] = a[i] - b[i];\n"
	"}\n"
	"__kernel void ir_div(__global const float *a, __global const float *b,\n"
	"                     __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) d[i] = a[i] / b[i];\n"
	"}\n"
	"__kernel void ir_neg(__global const float *a, __global float *d, const uint n) {\n"
	"  uint i = get_global_id(0);\n"
	"  if (i < n) d[i] = -a[i];\n"
	"}\n"
	"/* C = A·B, column-major: A[i+k·M], B[k+j·K], C[i+j·M] */\n"
	"__kernel void ir_mul_mat(__global const float *A, __global const float *B,\n"
	"                        __global float *C,\n"
	"                        const uint M, const uint K, const uint N) {\n"
	"  uint gid = get_global_id(0);\n"
	"  if (gid >= M * N) return;\n"
	"  uint i = gid % M;\n"
	"  uint j = gid / M;\n"
	"  float s = 0.f;\n"
	"  for (uint k = 0; k < K; k++)\n"
	"    s += A[i + k * M] * B[k + j * K];\n"
	"  C[i + j * M] = s;\n"
	"}\n";

static size_t slot_n(const struct mcp_ir_slot *s)
{
	size_t n = 1;
	int i;
	for (i = 0; i < GGML_MAX_DIMS; i++)
		n *= s->ne[i];
	return n;
}

static int build(const char *want)
{
	cl_uint np = 0, nd;
	cl_platform_id plats[8];
	cl_int err;
	cl_device_id dev = NULL;
	char name[256];
	const char *psrc = src;
	size_t sl = sizeof src - 1;
	cl_uint pi, di;

	if (clGetPlatformIDs(8, plats, &np) || !np)
		return -1;
	for (pi = 0; pi < np && !dev; pi++) {
		cl_device_id ds[8];
		nd = 0;
		if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, 8, ds, &nd))
			continue;
		for (di = 0; di < nd; di++) {
			name[0] = 0;
			clGetDeviceInfo(ds[di], CL_DEVICE_NAME, sizeof name, name, NULL);
			if (!want || !*want || strstr(name, want)) {
				dev = ds[di];
				break;
			}
		}
	}
	if (!dev)
		return -2;
	ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	if (err) return -3;
	q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
	if (err) {
		q = clCreateCommandQueue(ctx, dev, 0, &err);
		if (err) return -4;
	}
	prog = clCreateProgramWithSource(ctx, 1, &psrc, &sl, &err);
	if (err) return -5;
	err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
	if (err) {
		char log[2048];
		clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
		fprintf(stderr, "ocl build: %s\n", log);
		return -6;
	}
	k_add = clCreateKernel(prog, "ir_add", &err);
	k_mul = clCreateKernel(prog, "ir_mul", &err);
	k_sub = clCreateKernel(prog, "ir_sub", &err);
	k_div = clCreateKernel(prog, "ir_div", &err);
	k_silu = clCreateKernel(prog, "ir_silu", &err);
	k_rms = clCreateKernel(prog, "ir_rms", &err);
	k_neg = clCreateKernel(prog, "ir_neg", &err);
	k_mul_mat = clCreateKernel(prog, "ir_mul_mat", &err);
	if (!k_add || !k_mul || !k_sub || !k_div || !k_silu || !k_rms || !k_neg || !k_mul_mat)
		return -7;
	clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL);
	fprintf(stderr, "mcp_ir_ocl: %s\n", name);
	ready = 1;
	mcp_ir_ll_set(mcp_ir_ocl_binop, mcp_ir_ocl_unary, mcp_ir_ocl_mul_mat);
	return 0;
}

int mcp_ir_ll_ocl(const char *device_substr)
{
	int rc;
	if (ready) {
		mcp_ir_ll_set(mcp_ir_ocl_binop, mcp_ir_ocl_unary, mcp_ir_ocl_mul_mat);
		return 0;
	}
	rc = build(device_substr ? device_substr : "B60");
	return rc;
}

static int run_binop(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	size_t n, bytes;
	cl_mem A, B, D;
	cl_int err;
	cl_kernel k;
	cl_uint nu;
	size_t g;

	if (!ready)
		return -10;
	if (dst->dtype != GGML_TYPE_F32 || a->dtype != GGML_TYPE_F32 || b->dtype != GGML_TYPE_F32)
		return -2;
	n = slot_n(dst);
	{
		size_t na = slot_n(a), nb = slot_n(b);
		if (na < n) n = na;
		if (nb < n) n = nb;
	}
	if (op == GGML_OP_ADD)
		k = k_add;
	else if (op == GGML_OP_MUL)
		k = k_mul;
	else if (op == GGML_OP_SUB)
		k = k_sub;
	else if (op == GGML_OP_DIV)
		k = k_div;
	else
		return -3;
	bytes = n * sizeof(float);
	nu = (cl_uint)n;
	A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a->base, &err);
	B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b->base, &err);
	D = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	if (!A || !B || !D)
		return -4;
	clSetKernelArg(k, 0, sizeof A, &A);
	clSetKernelArg(k, 1, sizeof B, &B);
	clSetKernelArg(k, 2, sizeof D, &D);
	clSetKernelArg(k, 3, sizeof nu, &nu);
	g = n;
	err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, NULL);
	if (!err)
		err = clEnqueueReadBuffer(q, D, CL_TRUE, 0, bytes, dst->base, 0, NULL, NULL);
	clReleaseMemObject(A);
	clReleaseMemObject(B);
	clReleaseMemObject(D);
	return err ? -5 : 0;
}

static int run_unary(uint8_t op, struct mcp_ir_slot *dst, const struct mcp_ir_slot *a)
{
	size_t n, bytes;
	cl_mem A, D;
	cl_int err;
	cl_kernel k;
	cl_uint nu;
	size_t g;

	if (!ready)
		return -10;
	if (dst->dtype != GGML_TYPE_F32 || a->dtype != GGML_TYPE_F32)
		return -2;
	n = slot_n(dst);
	{
		size_t na = slot_n(a);
		if (na < n) n = na;
	}
	if (op == GGML_OP_UNARY)
		k = k_silu;
	else if (op == GGML_OP_RMS_NORM)
		k = k_rms;
	else if (op == GGML_OP_IR_NEG)
		k = k_neg;
	else
		return -3;
	bytes = n * sizeof(float);
	nu = (cl_uint)n;
	A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a->base, &err);
	D = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	if (!A || !D)
		return -4;
	clSetKernelArg(k, 0, sizeof A, &A);
	clSetKernelArg(k, 1, sizeof D, &D);
	clSetKernelArg(k, 2, sizeof nu, &nu);
	g = (op == GGML_OP_RMS_NORM) ? 1 : n;
	err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, NULL);
	if (!err)
		err = clEnqueueReadBuffer(q, D, CL_TRUE, 0, bytes, dst->base, 0, NULL, NULL);
	clReleaseMemObject(A);
	clReleaseMemObject(D);
	return err ? -5 : 0;
}

int mcp_ir_ocl_mul_mat(struct mcp_ir_slot *dst,
		       const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	size_t M, K, N;
	size_t ba, bb, bc;
	cl_mem A, B, C;
	cl_int err;
	cl_uint Mu, Ku, Nu;
	size_t g;

	if (!ready)
		return -10;
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

	if (dbg_mul_mat())
		fprintf(stderr, "[qnn_dbg mul_mat] enter M=%zu K=%zu N=%zu\n", M, K, N);

	ba = M * K * sizeof(float);
	bb = K * N * sizeof(float);
	bc = M * N * sizeof(float);
	Mu = (cl_uint)M;
	Ku = (cl_uint)K;
	Nu = (cl_uint)N;

	A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ba, a->base, &err);
	B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bb, b->base, &err);
	C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bc, NULL, &err);
	if (!A || !B || !C) {
		if (A) clReleaseMemObject(A);
		if (B) clReleaseMemObject(B);
		if (C) clReleaseMemObject(C);
		return -5;
	}
	clSetKernelArg(k_mul_mat, 0, sizeof A, &A);
	clSetKernelArg(k_mul_mat, 1, sizeof B, &B);
	clSetKernelArg(k_mul_mat, 2, sizeof C, &C);
	clSetKernelArg(k_mul_mat, 3, sizeof Mu, &Mu);
	clSetKernelArg(k_mul_mat, 4, sizeof Ku, &Ku);
	clSetKernelArg(k_mul_mat, 5, sizeof Nu, &Nu);
	g = M * N;
	if (dbg_mul_mat())
		fprintf(stderr, "[qnn_dbg mul_mat] branch: ir_mul_mat g=%zu\n", g);
	err = clEnqueueNDRangeKernel(q, k_mul_mat, 1, NULL, &g, NULL, 0, NULL, NULL);
	if (!err)
		err = clEnqueueReadBuffer(q, C, CL_TRUE, 0, bc, dst->base, 0, NULL, NULL);
	clReleaseMemObject(A);
	clReleaseMemObject(B);
	clReleaseMemObject(C);
	if (dbg_mul_mat())
		fprintf(stderr, "[qnn_dbg mul_mat] leave err=%d\n", (int)err);
	return err ? -6 : 0;
}

int mcp_ir_ocl_binop(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a, const struct mcp_ir_slot *b)
{
	return run_binop(op, dst, a, b);
}

int mcp_ir_ocl_unary(uint8_t op, struct mcp_ir_slot *dst,
		     const struct mcp_ir_slot *a)
{
	return run_unary(op, dst, a);
}
