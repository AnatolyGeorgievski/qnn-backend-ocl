// Конвертация f32/f16/bf16 ↔ Q8_0. Отдельная программа: только -cl-std=CL3.0,
// без -cl-mad-enable и fast-math.
// Проба, не релиз. Перенос остатка включается одной строкой ниже.
// Если её закомментировать, rem остаётся 0 и квантование как раньше: каждый отсчёт сам по себе.
struct __attribute__((packed)) block_q8_0 { half d; char qs[32]; };
inline void quant_block(float x[32], global struct block_q8_0 * y) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = fmax(amax, fabs(x[j]));
    float d = 0.f, id = 0.f;
    if (amax != 0.f) { d = amax / 127.f; id = 127.f / amax; }
    y->d = convert_half_rte(d);
    // d_h — то, что реально лежит в блоке. Остаток r не выходит за 32 отсчёта.
    float d_h = convert_float(y->d);
    float rem = 0.f;
    for (int j = 0; j < 32; ++j) {
        float acc = x[j] + rem;
        char q = convert_char_sat_rte(acc * id);
        y->qs[j] = q;
        rem = acc - (float) q * d_h;
    }
}
#define IDX(b, ne0, ne1, ne2, i0, i1, i2, i3) \
    do { int _i=(b)*32; int _n0=(ne0)>0?(ne0):1; int _n1=(ne1)>0?(ne1):1; int _n2=(ne2)>0?(ne2):1; int _t=_i/_n0; \
         (i0)=_i%_n0; (i1)=_t%_n1; _t/=_n1; (i2)=_t%_n2; (i3)=_t/_n2; } while (0)
kernel void kernel_cpy_f32_q8_0(global char * src, ulong off0, global char * dst, ulong off1,
        int ne0, int ne1, int ne2, int ne3, ulong nb00, ulong nb01, ulong nb02, ulong nb03, int nblocks) {
    int b = get_global_id(0); if (b >= nblocks) return;
    int i0,i1,i2,i3; IDX(b,ne0,ne1,ne2,i0,i1,i2,i3);
    ulong base = off0 + (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
    float x[32];
    for (int j = 0; j < 32; ++j) x[j] = *(global float *)(src + base + (ulong)j*nb00);
    quant_block(x, (global struct block_q8_0 *)(dst + off1) + b); (void)ne3;
}
kernel void kernel_cpy_f16_q8_0(global char * src, ulong off0, global char * dst, ulong off1,
        int ne0, int ne1, int ne2, int ne3, ulong nb00, ulong nb01, ulong nb02, ulong nb03, int nblocks) {
    int b = get_global_id(0); if (b >= nblocks) return;
    int i0,i1,i2,i3; IDX(b,ne0,ne1,ne2,i0,i1,i2,i3);
    ulong base = off0 + (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
    float x[32];
    for (int j = 0; j < 32; ++j) x[j] = *(global half *)(src + base + (ulong)j*nb00);
    quant_block(x, (global struct block_q8_0 *)(dst + off1) + b); (void)ne3;
}
kernel void kernel_cpy_bf16_q8_0(global char * src, ulong off0, global char * dst, ulong off1,
        int ne0, int ne1, int ne2, int ne3, ulong nb00, ulong nb01, ulong nb02, ulong nb03, int nblocks) {
    int b = get_global_id(0); if (b >= nblocks) return;
    int i0,i1,i2,i3; IDX(b,ne0,ne1,ne2,i0,i1,i2,i3);
    ulong base = off0 + (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
    float x[32];
    for (int j = 0; j < 32; ++j) {
        ushort u = *(global ushort *)(src + base + (ulong)j*nb00);
        x[j] = as_float((uint)u << 16);
    }
    quant_block(x, (global struct block_q8_0 *)(dst + off1) + b); (void)ne3;
}
kernel void kernel_cpy_q8_0_f32(global char * src, ulong off0, global char * dst, ulong off1,
        int ne0, int ne1, int ne2, int ne3, ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3, int nblocks) {
    int b = get_global_id(0); if (b >= nblocks) return;
    int i0,i1,i2,i3; IDX(b,ne0,ne1,ne2,i0,i1,i2,i3);
    ulong sbase = off0 + (ulong)(i0/32)*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
    global struct block_q8_0 * blk = (global struct block_q8_0 *)(src + sbase);
    float d = convert_float(blk->d);
    ulong dbase = off1 + (ulong)i0*nb0 + (ulong)i1*nb1 + (ulong)i2*nb2 + (ulong)i3*nb3;
    for (int j = 0; j < 32; ++j)
        *(global float *)(dst + dbase + (ulong)j*nb0) = (float)blk->qs[j] * d;
    (void)ne3;
}
