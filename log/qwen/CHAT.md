# Qwen · чат ≠ старт ≠ ir_say

| слой | что | как |
|--|--|--|
| запуск модели | `llama-server` грузит GGUF | `qwen-up` / `qwen-switch.sh q8` на arc3 |
| чат | webui того же процесса | http://192.168.1.187:18081/ |
| инструмент | модель зовёт из чата | файлы, fetch, **mcp-ir_ir_say** |

Один инструмент «и чат и запуск» не делается: чат — UI сервера, запуск — процесс с весами, ir_say — MCP tool.

Очередь и контракт помощника: `ROAD.md` · `HELPER.md`.

Сейчас проверено: health ok, `mcp-ir_ir_say` → `GET.vec $1(2) -3 -4`.


## 2026-11-21 · индекс Blackwell self

Постановка: свой метод + каркас вызовов только 120a / self-run. Результат: `index/BLACKWELL-SELF.md` (на arc3 `~/work/qnn-backend-ocl/index/`). Метод: скрипт→env→server→load_model→common_init→4 окна cuda.cu. OpenCL — вне контура.


## 2026-11-21 · BACKEND-LADDER

Метод: один `qnn-cuda.cu` + лесенка; mmv/mmq/mma/mul_mat/fattn и common отдельно. План: `index/BACKEND-LADDER.md`. Пазл: common→каркас→stub лесенка→mul_mat→mmv/mmq→fattn.

## 2026-11-21 · DEBUG-ONE mul_mat

В METHOD — «Отладка · один метод». Практика: `cuda120/qnn_dbg.h`, stub `mul_mat.cu`, `index/DEBUG-ONE.md`. Без жёсткого «только write» — жевал чтение; с запретом explore — сразу написал.
