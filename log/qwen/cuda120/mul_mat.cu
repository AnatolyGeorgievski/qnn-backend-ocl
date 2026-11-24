/*
 * mul_mat.cu — stub диспетчера qnn_cuda_mul_mat.
 *
 * Правило METHOD: один метод за ход.
 * Отладка: QNN_DBG=mul_mat
 *
 * Ветки (mmvf / mmvq / mmq / cublas) пока не реализованы —
 * каждая только печатает DBG-строку и возвращает 0.
 */
#include "qnn_dbg.h"

#include <cuda_runtime.h>

/*
 * qnn_cuda_mul_mat — stub.
 *
 * Параметры (подписи как в реальном backend, значения пока не используются):
 *   a, b  — указатели на матрицы
 *   c     — указатель на результат
 *   m, n, k — размеры (C[m,n] = A[m,k] * B[k,n])
 *   ne    — число элементов
 *   type  — тип данных (для выбора ветки)
 *
 * Возврат: 0 — успех (stub), -1 — ошибка.
 */
extern "C" int qnn_cuda_mul_mat(
    const void * a,
    const void * b,
    void * c,
    int m, int n, int k,
    int ne,
    int type) {

    /* ---- enter ---- */
    QNN_DBG("mul_mat", "enter: m=%d n=%d k=%d ne=%d type=%d", m, n, k, ne, type);

    /*
     * Ветки диспетчера.
     * Порядок проверки — от частного к общему:
     *   1. mmvf  — matrix-matrix vector float  (B — вектор, fp32)
     *   2. mmvq  — matrix-matrix vector quant  (B — вектор, квант)
     *   3. mmq   — matrix-matrix quant         (B — матрица, квант)
     *   4. cublas — fallback, обычная fp32/fp16 матрица
     *
     * Пока каждая ветка только логируется.
     */

    /* --- ветка mmvf --- */
    if (0) { /* TODO: условие: B — вектор, fp32 */
        QNN_DBG("mul_mat", "branch: mmvf (B=vector fp32)");
        /* TODO: запуск ядра mmvf */
        QNN_DBG("mul_mat", "leave: mmvf done");
        return 0;
    }

    /* --- ветка mmvq --- */
    if (0) { /* TODO: условие: B — вектор, квантованный */
        QNN_DBG("mul_mat", "branch: mmvq (B=vector quant)");
        /* TODO: запуск ядра mmvq */
        QNN_DBG("mul_mat", "leave: mmvq done");
        return 0;
    }

    /* --- ветка mmq --- */
    if (0) { /* TODO: условие: B — матрица, квантованная */
        QNN_DBG("mul_mat", "branch: mmq (B=matrix quant)");
        /* TODO: запуск ядра mmq */
        QNN_DBG("mul_mat", "leave: mmq done");
        return 0;
    }

    /* --- ветка cublas (fallback) --- */
    {
        QNN_DBG("mul_mat", "branch: cublas (fallback fp32/fp16)");
        /* TODO: cublasSgemm / cublasHgemm */
        QNN_DBG("mul_mat", "leave: cublas done");
        return 0;
    }
}
