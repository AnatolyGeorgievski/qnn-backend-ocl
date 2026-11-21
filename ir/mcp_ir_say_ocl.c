/* речь на OpenCL LL — тот же текст, что mcp_ir_say. */
#include "mcp_ir.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static const char speech[] =
	"SET $0(2x1) 3 4\n"
	"$0 RMS_NORM $1(2)\n"
	"SET $2(2\u00b71) 2 2\n"
	"$1 $2 MUL $3(2)\n"
	"GET.vec $3(2)\n"
	"GET.vec $1(2)\n";

int main(void)
{
	char out[2048];
	const char *dev = getenv("MCP_IR_DEVICE");
	float scale, v0, v1;
	int rc;

	if (!dev || !*dev)
		dev = "B60";
	rc = mcp_ir_ll_ocl(dev);
	if (rc) {
		fprintf(stderr, "FAIL  ll_ocl rc=%d\n", rc);
		return 1;
	}
	rc = mcp_ir_say(speech, out, sizeof out);
	fputs(speech, stdout);
	fputs(out, stdout);
	if (rc) {
		printf("FAIL  say rc=%d\n", rc);
		return 1;
	}
	scale = 1.f / sqrtf((9.f + 16.f) / 2.f + 1e-5f);
	v0 = 3.f * scale;
	v1 = 4.f * scale;
	{
		char want[128];
		snprintf(want, sizeof want, "GET.vec $1(2) %.6g %.6g", (double)v0, (double)v1);
		if (!strstr(out, want)) {
			printf("FAIL  vec $1 want %s\n", want);
			return 1;
		}
		snprintf(want, sizeof want, "GET.vec $3(2) %.6g %.6g",
			 (double)(v0 * 2.f), (double)(v1 * 2.f));
		if (!strstr(out, want)) {
			printf("FAIL  vec $3 want %s\n", want);
			return 1;
		}
	}
	printf("OK    SET graph GET on ocl\n");

	/* один метод: MUL_MAT на ocl LL */
	{
		const char more[] =
			"SET $5(2x3) 1 0 0 1 0 0\n"
			"SET $6(3x2) 1 0 0 1 0 0\n"
			"$5 $6 MUL_MAT $7(2x2)\n"
			"GET.vec $7(2·2)\n";
		char bout[2048];
		int r2 = mcp_ir_say(more, bout, sizeof bout);
		fputs(more, stdout);
		fputs(bout, stdout);
		if (r2) {
			printf("FAIL  MUL_MAT say rc=%d\n", r2);
			return 1;
		}
		if (!strstr(bout, "GET.vec $7(2·2) 1 0 1 0") &&
		    !strstr(bout, "GET.vec $7(2·2) 1 0 1 0\n")) {
			printf("FAIL  MUL_MAT vec: %s\n", bout);
			return 1;
		}
		printf("OK    MUL_MAT on ocl\n");
	}
	return 0;
}
