# BACKEND-LADDER — qnn-cuda (cuda120/)

## 1. Метод
Один backend-файл несёт каркас (init/buffer/supports_op/graph_compute/reg) и лесенку
`compute_forward` (switch по op). Ядра вынесены в отдельные файлы (mul_mat, fattn),
общее — в common. Всё своими словами, без копирования llama.cpp.

## 2. Дерево (под `cuda120/`)
- `qnn-cuda.cu` — init, buffer, supports_op, graph_compute, reg
- `graph.cu` (или секция в backend) — evaluate + `compute_forward` лесенка
- `common.cu` / `common.cuh` — утилиты, типы, хелперы
- `mul_mat.cu` — диспетчер MUL_MAT
- `mmvf.cu` `mmvq.cu` `mmq.cu` (+ cuh), `mma.cuh` — ядра матричных умножений
- `fattn.cu` (+ cuh) — flash attention

## 3. Лесенка (op → функция → файл)
| op | функция | файл |
|---|---|---|
| MUL_MAT | mul_mat (диспетчер) → mmvf/mmvq/mmq/cublas | mul_mat.cu (~2446) |
| FLASH_ATTN_EXT | flash_attn_ext | fattn.cu (~819) |
| остальные | stub (no-op / assert) | graph.cu, switch в compute_forward (~2708) |

Якоря:
- mul_mat ~2446 → mmvf/mmvq/mmq/cublas
- compute_forward ~2708 — switch по op
- graph_evaluate ~5176 → вызывает compute_forward
- graph_compute ~5422
- supports_op ~6199
- flash_attn_ext в fattn.cu ~819

## 4. Порядок сборки (пазл)
1. common
2. backend каркас (init/buffer/supports_op/graph_compute/reg)
3. лесенка stub (switch в compute_forward)
4. mul_mat диспетчер
5. mmv/mmq ядра
6. fattn

## 5. DONE
Коротко. llama.cpp не трогать.
