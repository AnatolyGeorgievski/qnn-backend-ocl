#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#error "Subgroup not supported on your device."
#endif

#ifdef cl_intel_required_subgroup_size
// Always use subgroup size of 32 on Intel.
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
// Always use subgroups size of 64 on Adreno.
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
// TODO: do not know how to choose subgroup size on other GPUs.
#error "Selecting subgroup size is not supported on your device."
#endif

kernel void kernel_im2col_f32(
        global float * src1,
        global float * dst,
        uint IW,
        uint IH,
        uint IC,
        uint OW,
        uint OH,
        uint KW,
        uint KH,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        uint d0,
        uint d1
) {
    uint pos = get_global_id(0);
    if (pos >= OW) {
        return;
    }
    uint row = get_group_id(1);
    uint im = get_group_id(2);
    uint batch = im / IC;
    uint ic = im % IC;

    // Сплошная картинка: шаг канала IW*IH, шаг плана IW*IH*IC. Хост это уже проверил.
    uint wh = mul24(KH, KW);
    uint src_base = mad24(batch, IC, ic);
    src_base = mul24(src_base, IH);
    src_base = mul24(src_base, IW);

    uint slot_base = mad24(batch, OH, row);
    slot_base = mad24(slot_base, OW, pos);
    slot_base = mul24(slot_base, mul24(IC, wh));
    slot_base = mad24(ic, wh, slot_base);

    int i0 = (int) pos * s0 - p0;
    int j0 = (int) row * s1 - p1;

    for (uint ky = 0; ky < KH; ky++) {
        uint dj = ky * d1;
        int j = j0 + (int) dj;
        uint slot_y = mad24(ky, KW, slot_base);
        bool row_ok = j >= 0 && j < (int) IH;
        uint src_y = 0;
        if (row_ok) {
            src_y = mad24((uint) j, IW, src_base);
        }
        for (uint kx = 0; kx < KW; kx++) {
            uint di = kx * d0;
            int i = i0 + (int) di;
            bool in_bound = row_ok && i >= 0 && i < (int) IW;
            dst[slot_y + kx] = in_bound ? src1[src_y + (uint) i] : 0.f;
        }
    }
}


kernel void kernel_im2col_f16(
        global float * src1,
        global half * dst,
        uint IW,
        uint IH,
        uint IC,
        uint OW,
        uint OH,
        uint KW,
        uint KH,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        uint d0,
        uint d1
) {
    uint pos = get_global_id(0);
    if (pos >= OW) {
        return;
    }
    uint row = get_group_id(1);
    uint im = get_group_id(2);
    uint batch = im / IC;
    uint ic = im % IC;

    // Сплошная картинка: шаг канала IW*IH, шаг плана IW*IH*IC. Хост это уже проверил.
    uint wh = mul24(KH, KW);
    uint src_base = mad24(batch, IC, ic);
    src_base = mul24(src_base, IH);
    src_base = mul24(src_base, IW);

    uint slot_base = mad24(batch, OH, row);
    slot_base = mad24(slot_base, OW, pos);
    slot_base = mul24(slot_base, mul24(IC, wh));
    slot_base = mad24(ic, wh, slot_base);

    int i0 = (int) pos * s0 - p0;
    int j0 = (int) row * s1 - p1;

    for (uint ky = 0; ky < KH; ky++) {
        uint dj = ky * d1;
        int j = j0 + (int) dj;
        uint slot_y = mad24(ky, KW, slot_base);
        bool row_ok = j >= 0 && j < (int) IH;
        uint src_y = 0;
        if (row_ok) {
            src_y = mad24((uint) j, IW, src_base);
        }
        for (uint kx = 0; kx < KW; kx++) {
            uint di = kx * d0;
            int i = i0 + (int) di;
            bool in_bound = row_ok && i >= 0 && i < (int) IW;
            dst[slot_y + kx] = in_bound ? src1[src_y + (uint) i] : 0.f;
        }
    }
}

#if 0
// выделение плитки в колонку изображения
kernel void im2col(
    global float *src,// изображение
    global float *dst,// уменьшенное 
    uint ne00, // число элементов в строке
    uint ne01, // число строк
    uint ne0, // число элементов в строке
    uint ne1, // число строк
    uint kw, // ширина плитки 
    uint kh // высота плитки 
){
    uint pos = get_group_id(0);// позиция записи
    uint row = get_group_id(1);// строка выходного изображения
    uint  im = get_group_id(2);// номер плана 

    dst += im*(ne0*ne1);
    bool in_bound = i>=0 && i<ne00 && j>=0 && j<ne01;
    dst[mad24(row, ne0, pos)] = in_bound?src[mad24(i,ne00,j)]:0.f;
}
#endif
// усреднение по плитке размером kh*hw
kernel void pool_2D_avg(// op = {OP_AVG , OP_MAX}
    global float *src,// изображение
    global float *dst,// уменьшенное 
    uint ne00, // число элементов в строке
    uint ne01, // число строк
    uint ne0, // число элементов в строке
    uint ne1, // число строк
    uint kw, // ширина плитки 
    uint kh // высота плитки 
){
    uint pos = get_group_id(0);// позиция записи
    uint row = get_group_id(1);// строка выходного изображения
    uint  im = get_group_id(2);// номер плана 

    dst += im*ne0 *ne1;
    src += im*ne00*ne01 + row*kh*ne00 + kw*pos;

    uint tid = get_sub_group_local_id();
    const uint stride = get_max_sub_group_size();
    float sum = 0;
    for (uint i=0; i<kh; i++, src+=ne00)
        for (uint j=tid; i<kw; i+=stride)
            sum += src[j];
    sum = sub_group_reduce_add(sum);
    if (get_sub_group_local_id()==0)
        dst[mad24(row, ne0, pos)] = sum/(kh*kw);
}
kernel void pool_2D_max(// op = {OP_AVG , OP_MAX}
    global float *src,// изображение
    global float *dst,// уменьшенное 
    uint ne00, // число элементов в строке
    uint ne01, // число строк
    uint ne0, // число элементов в строке
    uint ne1, // число строк
    uint kw, // ширина плитки 
    uint kh // высота плитки 
){
    uint pos = get_group_id(0);// позиция записи
    uint row = get_group_id(1);// строка выходного изображения
    uint  im = get_group_id(2);// номер плана 

    dst += im*ne0 *ne1;
    src += im*ne00*ne01 + row*kh*ne00 + kw*pos;

    uint tid = get_sub_group_local_id();
    const uint stride = get_max_sub_group_size();
    float sum = -FLT_MAX;
    for (uint i=0; i<kh; i++, src+=ne00)
        for (uint j=tid; i<kw; i+=stride)
            sum = fmax(src[j], sum);
    sum = sub_group_reduce_max(sum);
    if (get_sub_group_local_id()==0)
        dst[mad24(row, ne0, pos)] = sum;
}