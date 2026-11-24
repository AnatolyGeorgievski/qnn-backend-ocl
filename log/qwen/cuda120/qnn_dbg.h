/*
 * qnn_dbg.h — минимальный отладочный логгер для QNN CUDA backend.
 *
 * Включение: переменная окружения QNN_DBG.
 *   QNN_DBG=mul_mat        — логировать только mul_mat
 *   QNN_DBG=mul_mat,mmvf   — логировать mul_mat и mmvf
 *   QNN_DBG=*              — логировать всё
 *
 * Правило METHOD: один метод за ход.
 * Для отладки mul_mat ставим ровно: QNN_DBG=mul_mat
 */
#ifndef QNN_DBG_H
#define QNN_DBG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * QNN_DBG_ON(name) — 1, если имя name включено в QNN_DBG.
 *
 * Логика:
 *   - QNN_DBG не задан или пустой  -> 0
 *   - QNN_DBG="*"                  -> 1 (всё)
 *   - иначе: name должен встретиться как отдельный токен,
 *     разделённый запятыми (пробелы вокруг токенов допустимы).
 */
static inline int qnn_dbg_on(const char * name) {
    const char * dbg = getenv("QNN_DBG");
    if (!dbg || !*dbg) return 0;
    if (strcmp(dbg, "*") == 0) return 1;

    /* проход по токенам, разделённым ',' */
    const char * p = dbg;
    while (p) {
        const char * comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);

        /* обрезаем пробелы слева */
        while (len > 0 && (*p == ' ' || *p == '\t')) { p++; len--; }
        /* обрезаем пробелы справа */
        while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t')) len--;

        if (len == strlen(name) && strncmp(p, name, len) == 0) {
            return 1;
        }

        p = comma ? comma + 1 : NULL;
    }
    return 0;
}

#define QNN_DBG_ON(name) qnn_dbg_on(name)

/*
 * QNN_DBG(name, fmt, ...) — печать в stderr, если name включён.
 *
 * Формат строки:
 *   [qnn_dbg <name>] <fmt>
 *
 * Пример вывода:
 *   [qnn_dbg mul_mat] enter: m=4096 n=11008 k=4096
 */
#define QNN_DBG(name, fmt, ...) do { \
    if (QNN_DBG_ON(name)) { \
        fprintf(stderr, "[qnn_dbg %s] " fmt "\n", (name), ##__VA_ARGS__); \
    } \
} while (0)

#endif /* QNN_DBG_H */
