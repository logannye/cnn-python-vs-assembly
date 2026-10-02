// Handwritten Apple AArch64 convolution kernels; see ABI.md.
// x18 is reserved on Darwin. All tensors and accumulators are float32.
.text
.p2align 4

// Allocate a zero-padded CHW image. x0=input, w1=C, w2=W -> x0=allocation.
// The allocation is C*(W+2)*(W+2)*4 bytes. Caller owns it.
Lpad_image:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    uxtw x20, w1
    uxtw x21, w2
    add x22, x21, #2
    mul x23, x22, x22
    lsl x23, x23, #2
    lsl x22, x22, #2
    mul x0, x20, x23
    mov x28, x0
    bl _malloc
    cbz x0, Lallocation_failed
    mov x24, x0
    mov x1, #0
    mov x2, x28
    bl _memset
    add x25, x24, x22
    add x25, x25, #4
    mov x26, x20
Lpad_channel:
    mov x27, x21
Lpad_row:
    mov x0, x25
    mov x1, x19
    lsl x2, x21, #2
    bl _memcpy
    add x19, x19, x21, lsl #2
    add x25, x25, x22
    subs x27, x27, #1
    b.ne Lpad_row
    add x25, x25, x22, lsl #1
    subs x26, x26, #1
    b.ne Lpad_channel
    mov x0, x24
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #96
    ret
Lallocation_failed:
    bl _abort
    brk #0

// Four output channels, sixteen adjacent pixels, one kernel tap.
// v0-v15 are accumulators; v16-v19=input, v20-v23=broadcast weights.
.macro FWD_TAP off
    ldr q16, [x0, #\off]
    ldr q17, [x0, #(\off + 16)]
    ldr q18, [x0, #(\off + 32)]
    ldr q19, [x0, #(\off + 48)]
    ld1r {v20.4s}, [x11], #4
    ld1r {v21.4s}, [x12], #4
    ld1r {v22.4s}, [x13], #4
    ld1r {v23.4s}, [x14], #4
    fmla v0.4s, v16.4s, v20.4s
    fmla v1.4s, v17.4s, v20.4s
    fmla v2.4s, v18.4s, v20.4s
    fmla v3.4s, v19.4s, v20.4s
    fmla v4.4s, v16.4s, v21.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v6.4s, v18.4s, v21.4s
    fmla v7.4s, v19.4s, v21.4s
    fmla v8.4s, v16.4s, v22.4s
    fmla v9.4s, v17.4s, v22.4s
    fmla v10.4s, v18.4s, v22.4s
    fmla v11.4s, v19.4s, v22.4s
    fmla v12.4s, v16.4s, v23.4s
    fmla v13.4s, v17.4s, v23.4s
    fmla v14.4s, v18.4s, v23.4s
    fmla v15.4s, v19.4s, v23.4s
.endm

.globl _cnn_conv_forward
.p2align 4
_cnn_conv_forward:
    stp x29, x30, [sp, #-224]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    ldr w24, [x4]
    ldr w25, [x4, #4]
    ldr w26, [x4, #8]
    mul w28, w25, w25
    lsl x28, x28, #2
    add w8, w25, #2
    mul w7, w8, w8
    lsl x7, x7, #2
    str x7, [sp, #160]  // padded channel bytes
    lsl x8, x8, #2
    str x8, [sp, #168]  // padded row bytes
    mov x7, #36
    mul x7, x24, x7
    str x7, [sp, #176]  // weight output-channel bytes
    mov w1, w24
    mov w2, w25
    bl Lpad_image
    mov x27, x0
    mov w23, #0
Lfwd_output_group:
    mov w9, #0
Lfwd_row:
    mov w10, #0
Lfwd_tile:
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    ldr x5, [sp, #176]
    madd x11, x23, x5, x20
    add x12, x11, x5
    add x13, x12, x5
    add x14, x13, x5
    ldr x7, [sp, #168]
    ldr x8, [sp, #160]
    madd x15, x9, x7, x27
    add x15, x15, x10, lsl #2
    mov w16, w24
Lfwd_input_channel:
    mov x0, x15
    mov w17, #3
Lfwd_kernel_row:
    FWD_TAP 0
    FWD_TAP 4
    FWD_TAP 8
    add x0, x0, x7
    subs w17, w17, #1
    b.ne Lfwd_kernel_row
    add x15, x15, x8
    subs w16, w16, #1
    b.ne Lfwd_input_channel
    cbz x21, Lfwd_store
    add x0, x21, x23, lsl #2
    ld1r {v20.4s}, [x0], #4
    ld1r {v21.4s}, [x0], #4
    ld1r {v22.4s}, [x0], #4
    ld1r {v23.4s}, [x0]
    fadd v0.4s, v0.4s, v20.4s
    fadd v1.4s, v1.4s, v20.4s
    fadd v2.4s, v2.4s, v20.4s
    fadd v3.4s, v3.4s, v20.4s
    fadd v4.4s, v4.4s, v21.4s
    fadd v5.4s, v5.4s, v21.4s
    fadd v6.4s, v6.4s, v21.4s
    fadd v7.4s, v7.4s, v21.4s
    fadd v8.4s, v8.4s, v22.4s
    fadd v9.4s, v9.4s, v22.4s
    fadd v10.4s, v10.4s, v22.4s
    fadd v11.4s, v11.4s, v22.4s
    fadd v12.4s, v12.4s, v23.4s
    fadd v13.4s, v13.4s, v23.4s
    fadd v14.4s, v14.4s, v23.4s
    fadd v15.4s, v15.4s, v23.4s
Lfwd_store:
    madd x0, x23, x28, x22
    mul w4, w9, w25
    add w4, w4, w10
    add x0, x0, x4, lsl #2
    add x1, x0, x28
    add x2, x1, x28
    add x3, x2, x28
    stp q0, q1, [x0]
    stp q2, q3, [x0, #32]
    stp q4, q5, [x1]
    stp q6, q7, [x1, #32]
    stp q8, q9, [x2]
    stp q10, q11, [x2, #32]
    stp q12, q13, [x3]
    stp q14, q15, [x3, #32]
    add w10, w10, #16
    cmp w10, w25
    b.lo Lfwd_tile
    add w9, w9, #1
    cmp w9, w25
    b.lo Lfwd_row
    add w23, w23, #4
    cmp w23, w26
    b.lo Lfwd_output_group
    mov x0, x27
    bl _free
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    ldp x29, x30, [sp], #224
    ret

// Reduce sixteen spatial lanes into one scalar, then add to existing gradient.
.macro REDUCE_ADD a,b,c,d,ptr
    fadd v\a\().4s, v\a\().4s, v\b\().4s
    fadd v\c\().4s, v\c\().4s, v\d\().4s
    fadd v\a\().4s, v\a\().4s, v\c\().4s
    faddp v\a\().4s, v\a\().4s, v\a\().4s
    faddp s\a, v\a\().2s
    ldr s24, [\ptr]
    fadd s\a, s\a, s24
    str s\a, [\ptr], #4
.endm

.globl _cnn_conv_backward
.p2align 4
_cnn_conv_backward:
    stp x29, x30, [sp, #-256]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mov x19, x0  // input, later padded input
    mov x20, x1  // weights
    mov x21, x2  // gradient output
    mov x22, x3  // gradient input or NULL
    mov x23, x4  // gradient weights
    mov x24, x5  // gradient bias
    ldr w25, [x6]      // C
    ldr w26, [x6, #4]  // W
    ldr w27, [x6, #8]  // O
    mul w28, w26, w26
    lsl x28, x28, #2  // output plane bytes
    mov x7, #36
    mul x7, x25, x7
    str x7, [sp, #160] // output weight plane bytes
    add w7, w26, #2
    mul w8, w7, w7
    lsl x8, x8, #2
    str x8, [sp, #168] // padded input plane bytes
    lsl x7, x7, #2
    str x7, [sp, #176] // padded row bytes
    mov w1, w25
    mov w2, w26
    bl Lpad_image
    mov x19, x0

    // Bias derivatives: sum all pixels of each output channel.
    mov x0, x21
    mov x1, x24
    mov w2, w27
Lbias_channel:
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    lsr x3, x28, #6
Lbias_pixels:
    ldp q16, q17, [x0], #32
    ldp q18, q19, [x0], #32
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v17.4s
    fadd v2.4s, v2.4s, v18.4s
    fadd v3.4s, v3.4s, v19.4s
    subs x3, x3, #1
    b.ne Lbias_pixels
    REDUCE_ADD 0,1,2,3,x1
    subs w2, w2, #1
    b.ne Lbias_channel

    // Weight derivatives: four output channels share each shifted input tile.
    mov w9, #0  // output group start
Ldw_group:
    ldr x7, [sp, #160]
    madd x11, x9, x7, x23
    add x12, x11, x7
    add x13, x12, x7
    add x14, x13, x7
    mov w10, #0 // input channel
Ldw_channel:
    mov w15, #0 // kernel row
Ldw_kernel_row:
    mov w16, #0 // kernel column
Ldw_kernel_column:
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    ldr x7, [sp, #168]
    madd x0, x10, x7, x19
    ldr x7, [sp, #176]
    madd x0, x15, x7, x0
    add x0, x0, x16, lsl #2
    madd x1, x9, x28, x21
    add x2, x1, x28
    add x3, x2, x28
    add x4, x3, x28
    mov w5, w26
Ldw_image_row:
    lsr w6, w26, #4
Ldw_image_tile:
    ldp q16, q17, [x0], #32
    ldp q18, q19, [x0], #32
    ldp q20, q21, [x1], #32
    ldp q22, q23, [x1], #32
    fmla v0.4s, v16.4s, v20.4s
    fmla v1.4s, v17.4s, v21.4s
    fmla v2.4s, v18.4s, v22.4s
    fmla v3.4s, v19.4s, v23.4s
    ldp q20, q21, [x2], #32
    ldp q22, q23, [x2], #32
    fmla v4.4s, v16.4s, v20.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v6.4s, v18.4s, v22.4s
    fmla v7.4s, v19.4s, v23.4s
    ldp q20, q21, [x3], #32
    ldp q22, q23, [x3], #32
    fmla v8.4s, v16.4s, v20.4s
    fmla v9.4s, v17.4s, v21.4s
    fmla v10.4s, v18.4s, v22.4s
    fmla v11.4s, v19.4s, v23.4s
    ldp q20, q21, [x4], #32
    ldp q22, q23, [x4], #32
    fmla v12.4s, v16.4s, v20.4s
    fmla v13.4s, v17.4s, v21.4s
    fmla v14.4s, v18.4s, v22.4s
    fmla v15.4s, v19.4s, v23.4s
    subs w6, w6, #1
    b.ne Ldw_image_tile
    add x0, x0, #8 // two padded columns between image rows
    subs w5, w5, #1
    b.ne Ldw_image_row
    REDUCE_ADD 0,1,2,3,x11
    REDUCE_ADD 4,5,6,7,x12
    REDUCE_ADD 8,9,10,11,x13
    REDUCE_ADD 12,13,14,15,x14
    add w16, w16, #1
    cmp w16, #3
    b.lo Ldw_kernel_column
    add w15, w15, #1
    cmp w15, #3
    b.lo Ldw_kernel_row
    add w10, w10, #1
    cmp w10, w25
    b.lo Ldw_channel
    add w9, w9, #4
    cmp w9, w27
    b.lo Ldw_group
    mov x0, x19
    bl _free
    cbz x22, Lbackward_done

    // Input derivative is a same-padded convolution of grad_output with the
    // spatially reversed, input/output-transposed weights. Round its output
    // count to four to support an optional RGB input gradient as well.
    add w7, w25, #3
    and w7, w7, #0xfffffffc
    str w7, [sp, #184] // padded transposed output count
    mov x8, #36
    mul x8, x27, x8
    str x8, [sp, #192] // transposed weight output plane bytes
    mul x0, x7, x8
    str x0, [sp, #200]
    bl _malloc
    cbz x0, Lallocation_failed
    mov x19, x0
    mov x1, #0
    ldr x2, [sp, #200]
    bl _memset
    mov w9, #0 // original input channel = transposed output
Ltranspose_input:
    ldr x8, [sp, #192]
    madd x0, x9, x8, x19
    mov w10, #0 // original output channel = transposed input
Ltranspose_output:
    ldr x7, [sp, #160]
    madd x1, x10, x7, x20
    mov x8, #36
    madd x1, x9, x8, x1
    add x1, x1, #32
    mov w11, #9
Ltranspose_tap:
    ldr s0, [x1], #-4
    str s0, [x0], #4
    subs w11, w11, #1
    b.ne Ltranspose_tap
    add w10, w10, #1
    cmp w10, w27
    b.lo Ltranspose_output
    add w9, w9, #1
    cmp w9, w25
    b.lo Ltranspose_input
    ldr w7, [sp, #184]
    mov x23, x22
    cmp w7, w25
    b.eq Ltranspose_forward
    mul x0, x7, x28
    bl _malloc
    cbz x0, Lallocation_failed
    mov x23, x0
Ltranspose_forward:
    str w27, [sp, #208]
    str w26, [sp, #212]
    ldr w7, [sp, #184]
    str w7, [sp, #216]
    mov x0, x21
    mov x1, x19
    mov x2, #0
    mov x3, x23
    add x4, sp, #208
    bl _cnn_conv_forward
    cmp x23, x22
    b.eq Ltranspose_free_weights
    mov x0, x22
    mov x1, x23
    mul x2, x25, x28
    bl _memcpy
    mov x0, x23
    bl _free
Ltranspose_free_weights:
    mov x0, x19
    bl _free
Lbackward_done:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    ldp x29, x30, [sp], #256
    ret
