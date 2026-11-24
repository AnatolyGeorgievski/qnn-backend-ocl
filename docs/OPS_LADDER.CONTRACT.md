# Контракт · лесенка операций · по группам

<tags>контракт, лесенка, операции, opencl, B60, прямая-речь</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
  - name: kernels/ggml-opencl.cl
  - name: src/ggml-opencl.cpp
  - name: ChangeLog
path:
  - docs/PLAN.md
  - ir/STUMBLE.md
  - эта запись
</topo>

<toc>
- [Окно](#окно)
- [Матрица · приоритет](#матрица--приоритет)
- [Группа A · elementwise ADD/MUL](#группа-a--elementwise-addmul)
- [Группа B · нормы и fused](#группа-b--нормы-и-fused)
- [Группа C · hybrid / SSM → QNN](#группа-c--hybrid--ssm--qnn)
- [Группа D · приёмка](#группа-d--приёмка)
- [IR · лесенка разложений](#ir--лесенка-разложений)
- [Вне окна](#вне-окна)
- [Переделка при unsat](#переделка-при-unsat)
</toc>

<refs>
- ChangeLog · 2026-11-24
- README · What works
- log канон (godminer): `2026-10-29 прямая речь контракт.md`
</refs>

## Окно

Точка внимания: лесенка ops OpenCL на Intel Arc Pro **B60**.  
Глубина: **группы A–D** + закладка **IR**. Я (`[[cursor]]`) держу ядра + host + сверку `test-backend-ops`. Ты (`[[george]]`) ставишь клетку матрицы в работу и принимаешь по model-тесту и unit.  
Устройство: phantom, `GGML_OPENCL_DEVICE=B60`, overlay `llama-our-ocl-1001` / `build-rpc-ocl`, RPC **10.0.1** когда речь через RPC.

Общее для всех групп:

1. Я не объявляю `supports_op = true`, пока нет ядра и пути host для этого типа.
2. Я не крашу процесс `GGML_ASSERT` на виде / длине строки: либо считаю, либо `supports_op` = false.
3. Готово группы: `test-backend-ops test -o <OP>… -b 'GPU OpenCL'` → **fail = 0**; skip только там, где контракт группы явно разрешает.
4. Приоритет клеток — **блокеры прохождения тестов по моделям**, не косметика сигнатур ядер.

---

## Матрица · приоритет

1. **RMS_NORM и сочетания** (группа B) — нормы/fused на пути model-теста.
2. **SSM-семейство** (группа C) — переход к **QNN** и **гибридным** архитектурам (`SSM_CONV`, `GATED_DELTA_NET`, `SOFTPLUS`, `SSM_SCAN`).
3. Прочие блокеры из факта model-прогона.
4. Позже: offset cleanup; mem (куча/слайсы) под on-demand MoE; полная IR-лесенка.

Unit зелёный ≠ клетка закрыта по моделям (группа D).

---

## Группа A · elementwise ADD/MUL

Операции: `ADD`, `MUL` (flat / row / general).

### Узел A0 — ядро

1. Я (`kernels/ggml-opencl.cl`) для **f32** считаю `+` / `*` в float.
2. Я для **f16** не делаю native half `*`/`+` на B60. Я считаю `fma` в f32 и пишу `convert_half_rte`.
3. Я для **bf16** (same-type) считаю `fma` в f32 через `convert_bf16_to_f32` / `convert_f32_to_bf16`.
4. Я для **mixed** `dst=bf16`, `src0=bf16`, `src1=f32` иду в `kernel_add_bf16_f32` / `kernel_mul_bf16_f32` тем же правилом fma→round.
5. Готово: нет NaN на конечных входах; выход совпадает с CPU в допуске harness.

### Узел A1 — host

1. Я (`ggml_cl_add` / `ggml_cl_mul` в `src/ggml-opencl.cpp`) выбираю ядро по `dst->type` и типам `src0`/`src1`.
2. Я в flat/row путь вхожу только если `src0->type == src1->type == dst->type` и размеры кратны пути `*8`. Иначе — general.
3. Я для f16/bf16 flat ставлю размер региона = `nelements * ggml_type_size(type)`, не `sizeof(float)`.
4. Готово на B60 (факт 2026-11-24): ADD f32 58/58 · ADD f16 47/47 · ADD bf16 51/51 · MUL f32 48/48 · MUL f16 46/46 · MUL bf16 49/49.

### Запрет A

- Я не пускаю float8-ядро на буфер half/bf16.
- Я не чиню f16 «выключить finite-math» вместо fma→round.

---

## Группа B · нормы и fused

Операции: `RMS_NORM`, `L2_NORM`, `RMS_NORM_MUL_ADD`, `ADD_RMS_NORM`, `RMS_NORM_SCALE` (и f32 `NORM` / `NORM_SCALE` как сосед).

### Узел B0 — RMS_NORM

1. Я считаю сумму квадратов и `rsqrt(mean+eps)` в **f32** для типов f32 / f16 / bf16.
2. Я пишу выход: f32 как есть; f16 через `convert_half_rte`; bf16 через `convert_f32_to_bf16`.
3. Я принимаю строку с `nb[0] == ggml_type_size(type)` (сплошная внутренняя ось). Иначе `supports_op` = false.
4. Готово: **RMS_NORM 121/121** (0 skip).

### Узел B1 — L2_NORM

1. Я ставлю `scale = 1/max(sqrt(sum), eps)` как CPU, не `rsqrt(sum+eps)`.
2. Я читаю/пишу по `nb00`/`nb0` (виды и permute).
3. Готово: **L2_NORM 31/31**.

### Узел B2 — fused

1. Я для `RMS_NORM` → `MUL` (и дальше `ADD`) не ломаю цепочку: eps беру с узла RMS_NORM; веса f32 при активациях bf16 идут через mixed ядра группы A.
2. Готово: **RMS_NORM_MUL_ADD 113/113** · **ADD_RMS_NORM 25/25** · **RMS_NORM_SCALE 10/10**.

### Запрет B

- Я не оставляю bf16 RMS в skip, если harness его гоняет и устройство даёт `cl_intel_bfloat16_conversions`.
- Я не крашу сервер на `ne00 % 8` / `contiguous_1` assert.

---

## Группа C · hybrid / SSM → QNN

Операции: `SSM_CONV`, `GATED_DELTA_NET`, `SOFTPLUS` (f32/f16/bf16), плюс опоры CPY views / `ARGSORT` / `MUL_MAT_ID` где модель уже принята.  
Смысл в матрице: мост к **гибридным** графам и к **QNN** (разложение → шаблон команд / ICD), не «ещё один unit».

### Узел C0

1. Я держу unit-зелень hybrid без «skip в ok»: SSM_CONV **60/60**, GDN **43/43**, SSM_SCAN **17/17**.
2. Я для SOFTPLUS считаю f32/f16 и **bf16** (contiguous): softplus в f32, store `convert_f32_to_bf16`.
3. `SSM_SCAN` — OpenCL path есть; pure Mamba-130m F16 smoke (greedy seed = CPU).
4. Готово hybrid-unit: нет missing-op crash на Qwen3.5 graph; речь модели и QNN — группа D / IR, не один unit C.

### Запрет C

- Я не объявляю hybrid «закрыт по речи» / «готов к QNN», пока закрыт только unit.
- Я не маскирую `SSM_SCAN` под ok через `SSM_CONV` (SCAN закрыт отдельно unit + Mamba smoke).

---

## Группа D · приёмка

### Узел D0 — unit

1. Ты гоняешь на phantom:  
   `LD_LIBRARY_PATH=/tmp/gdn-test GGML_OPENCL_DEVICE=B60 ./test-backend-ops test -o <OP> -b 'GPU OpenCL'`.
2. Я после правки ядра/host копирую overlay в `llama.cpp-10.0.1/ggml/src/ggml-opencl`, собираю `ninja ggml-opencl`, кладу `.so` в `llama-our-ocl-1001/bin` и `/tmp/gdn-test`, перезапускаю RPC `:18769` если речь через RPC.
3. Готово: строка `N/N tests passed` и `Backend OpenCL: OK`; числа — в `ChangeLog` той же даты.

### Узел D1 — speech

1. Ты гоняешь модель: `llama-cli --rpc … -dev RPC0 -ngl 99`, temp 0, шаблон как в ChangeLog.
2. Готово speech: связный ответ не зациклен на special-токенах.  
   Сейчас (2026-11-24): tinyllama / Qwen2.5 / Gemma E4B / Gemma MoE — **ok**; Qwen3.5-27B — **fail**; матрица: RMS → SSM → речь.

### Запрет D

- Я не пишу в README «ok», пока нет строки в ChangeLog с ok/fail/skip.
- Я не двигаю STUMBLE в ok по unit, если речь ещё fail.

---

## IR · лесенка разложений

Идея лесенки продолжается на IR ([docs/MCP_IR.md](MCP_IR.md), [ir/STUMBLE.md](../ir/STUMBLE.md)):

1. Я для каждого разложения высокоуровневого op выбираю **шаблон команд GPU** под max perf (subgroup/tile/fma-path), не «любой корректный» CL.
2. Ступени совпадают с матрицей ops: elementwise → **RMS/fused** → **SSM/hybrid** → accept; SSM-rungs раньше косметики ADD-offset.
3. Форма: `OP → decompose(pattern) → cmd_template(device)`. OpenCL C — тонкий носитель, пока LL-HW / QNN ICD не заменит середину.
4. Готово закладки: этот раздел в контракте; реализация IR — отдельная постановка.

---

## Вне окна

- **TODO · кернелы:** убрать `offset*`/`off*`; смещение на хосте (sub-buffer / указатель на первый элемент). После зелени RMS+SSM на model-пути.
- **TODO · память GPU (позже):** менеджер с выравниванием на страницу кеша; методы **куча** и **слайсы**. Нужен для **отладки** и **замещения тензоров при on-demand MoE** (модель грузится по необходимости). Не блокер RMS/SSM.
- `SSM_SCAN` — закрыт unit **17/17** + Mamba-130m F16 smoke; дальше QNN ICD / IR templates.
- `RMS_NORM_BACK` — не поддерживаю.
- `RMS_NORM_MUL_ROPE` yarn / SET_ROWS fails — ROPE/SET_ROWS, не RMS.
- SCALE/CPY bf16 дыры — соседние; чиню отдельной постановкой.
- Adreno profile, QNN ICD / SPIR-V ladder (`QNN_OCL_KIND`) — `docs/PLAN.md` + раздел IR.

---

## Переделка при unsat

| симптом | действие |
|--|--|
| NaN на f16/bf16 ADD/MUL | Я возвращаю fma→f32→round; не ослабляю допуск harness |
| RMS bf16 skip при живом BF16 на карте | Я дописываю typed kernel + `supports_op` |
| flat путь ест half как float | Я режу flat same-type only + `ggml_type_size` |
| unit ok, речь Qwen3.5 fail | Я не закрываю группу D; новый ход — logits/smoke, не откат A–B |
| assert вместо skip | Я чиню `supports_op` / ядро; assert не оставляю |

**подпись:** [[cursor]]
