/* речь из stdin → ответ stdout */
#include "mcp_ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
	char *speech = NULL;
	size_t cap = 0, n = 0;
	char out[8192];
	int c, rc;

	speech = malloc(65536);
	if (!speech)
		return 2;
	cap = 65536;
	while ((c = getchar()) != EOF) {
		if (n + 1 >= cap) {
			cap *= 2;
			speech = realloc(speech, cap);
			if (!speech)
				return 2;
		}
		speech[n++] = (char)c;
	}
	speech[n] = 0;
	rc = mcp_ir_say(speech, out, sizeof out);
	fputs(out, stdout);
	free(speech);
	return rc ? 1 : 0;
}
