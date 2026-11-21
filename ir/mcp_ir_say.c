/* речь SET / граф / GET. Ответ — статус и вектор. */
#include "mcp_ir.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static const char speech[] =
	"SET $0(2x1) 3 4\n"
	"$0 RMS_NORM $1(2)\n"
	"SET $2(2\u00b71) 2 2\n"
	"$1 $2 MUL $3(2)\n"
	"GET.vec $3(2)\n"
	"SET $4(2\u00b72) 1 2 3 4\n"
	"GET $1(2)\n"
	"GET.vec $1(2)\n"
	"GET $4(2\u00b72)\n"
	"GET.vec $4(2\u00b72)\n";

int main(void)
{
	char out[2048];
	float scale, v0, v1;
	int rc;

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
	if (!strstr(out, "GET $1(2) status=ok dtype=f32")) {
		printf("FAIL  no status for $1\n");
		return 1;
	}
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
	if (!strstr(out, "GET $4(2\u00b72) status=ok dtype=f32") ||
		    !strstr(out, "GET.vec $4(2\u00b72) 1 2 3 4")) {
			printf("FAIL  cdot shape\n");
			return 1;
		}
		printf("OK    SET graph GET\n");

	/* расширение: SUB NEG DIV MUL_MAT */
	{
		const char more[] =
			"SET $0(2) 5 7\n"
			"SET $1(2) 2 3\n"
			"$0 $1 SUB $2(2)\n"
			"$2 NEG $3(2)\n"
			"$3 $1 DIV $4(2)\n"
			"GET.vec $4(2)\n"
			"SET $5(2x3) 1 0 0 1 0 0\n"
			"SET $6(3x2) 1 0 0 1 0 0\n"
			"$5 $6 MUL_MAT $7(2x2)\n"
			"GET.vec $7(2·2)\n";
		char bout[2048];
		int r2 = mcp_ir_say(more, bout, sizeof bout);
		fputs(more, stdout);
		fputs(bout, stdout);
		if (r2) { printf("FAIL  more say rc=%d\n", r2); return 1; }
		if (!strstr(bout, "GET.vec $4(2) -1.5 -1.33333") &&
		    !strstr(bout, "GET.vec $4(2) -1.5 -1.333333")) {
			/*  (5-2)=3 neg=-3 /2 = -1.5; (7-3)=4 neg=-4 /3 = -1.333... */
			printf("FAIL  SUB/NEG/DIV vec: %s\n", bout);
			return 1;
		}
		if (!strstr(bout, "GET.vec $7(2·2)")) {
			printf("FAIL  no MUL_MAT vec\n"); return 1;
		}
		printf("OK    SUB NEG DIV MUL_MAT\n");
	}
	return 0;
}
