# Вехи QNN Backend-OCL

| # | цель | корень |
|--|--|--|
| 1 | тензорный ассемблер · ICD QNN на базе Intel ICD (Arc / xe) | `/home/george/qnn-backend-ocl` |
| 2 | поддержка Мити: тот же / совместимый backend под Adreno | профиль ADRENO; прогон при `mitya_move` |
| — | quant-утилита BF16/F16→f8 | `…/tools/quantize` |
| — | RPC → docID/ZC | `/home/george/qnn-backend-rpc` |

Знания — QuantumNN. Сводка: `QuantumNN/PLAN-quant-rpc.md`. Инвариант: не `And(arc, adreno)`.


## Модели прогона

Сейчас только Qwen и Gemma. У обеих метка чата — один токен `<|im_start|>` (у Qwen это 151644).

TODO, не этот этап: ряд моделей другой архитектуры. TinyLlama сюда не входит: шаблон пишет `<|system|>`, в словаре такого токена нет.
