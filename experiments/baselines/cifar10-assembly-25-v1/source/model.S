// Handwritten Apple AArch64 CNN layers, loss, elementary math, and Adam.
// See ABI.md and MODEL.md. Every scratch tensor is supplied by the caller.
.text
.p2align 2
.equ C1, 0
.equ P1, 65536
.equ I1, 81920
.equ C2, 86016
.equ P2, 118784
.equ I2, 126976
.equ AVG, 129024
.equ DAVG, 129152
.equ DP2, 129280
.equ DC2, 137472
.equ DP1, 170240
.equ DC1, 186624
.equ WS_BYTES, 252160
.macro PTR dst, base, offset
    add \dst, \base, #(\offset >> 12), lsl #12
    add \dst, \dst, #(\offset & 4095)
.endm
.macro CONST dst, label
    adrp \dst, \label@PAGE
    add \dst, \dst, \label@PAGEOFF
.endm

.globl _cnn_workspace_bytes
_cnn_workspace_bytes:
    mov x0, #0xd900
    movk x0, #3, lsl #16
    ret

.globl _cnn_forward
_cnn_forward:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x0, x20
    mov x1, x19
    add x2, x19, #1728
    mov x3, x21
    CONST x4, Lshape1
    bl _cnn_conv_forward
    mov x0, x21
    PTR x1, x21, P1
    PTR x2, x21, I1
    mov w3, #16
    mov w4, #32
    bl Lrelu_pool
    PTR x0, x21, P1
    add x1, x19, #1792
    PTR x2, x19, 20224
    PTR x3, x21, C2
    CONST x4, Lshape2
    bl _cnn_conv_forward
    PTR x0, x21, C2
    PTR x1, x21, P2
    PTR x2, x21, I2
    mov w3, #32
    mov w4, #16
    bl Lrelu_pool
    // Global average of each contiguous 8x8 channel; four parallel sums.
    PTR x9, x21, P2
    PTR x10, x21, AVG
    mov w11, #32
    mov w12, #0x3c800000
    fmov s3, w12
Lavg_channel:
    movi v0.4s, #0
    mov w12, #16
Lavg_sum:
    ldr q1, [x9], #16
    fadd v0.4s, v0.4s, v1.4s
    subs w12, w12, #1
    b.ne Lavg_sum
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    fmul s0, s0, s3
    str s0, [x10], #4
    subs w11, w11, #1
    b.ne Lavg_channel
    // Linear classifier. Each output has 32 weights and one bias.
    PTR x9, x19, 20352
    PTR x10, x19, 21632
    PTR x11, x21, AVG
    mov w12, #10
Llinear_class:
    mov x13, x11
    movi v0.4s, #0
    mov w14, #8
Llinear_dot:
    ldr q1, [x9], #16
    ldr q2, [x13], #16
    fmla v0.4s, v1.4s, v2.4s
    subs w14, w14, #1
    b.ne Llinear_dot
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    ldr s1, [x10], #4
    fadd s0, s0, s1
    str s0, [x22], #4
    subs w12, w12, #1
    b.ne Llinear_class
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// Internal: relu(input) in place then pool2x2; x0 input x1 output x2 mask,
// w3 channels w4 width. Mask uint8 0,1,2,3 in row-major first-argmax order.
Lrelu_pool:
    mul w5, w4, w4
    mul w5, w5, w3
    mov x6, x0
    movi v0.4s, #0
Lrelu_loop:
    ldr q1, [x6]
    fmax v1.4s, v1.4s, v0.4s
    str q1, [x6], #16
    subs w5, w5, #4
    b.ne Lrelu_loop
    lsl x5, x4, #2
    lsr w6, w4, #1
    mul w6, w6, w3 // Total output rows across all channels.
Lpool_row:
    lsr w7, w4, #1
Lpool_col:
    ldp s0, s1, [x0]
    add x8, x0, x5
    ldp s2, s3, [x8]
    mov w9, #0
    fcmp s1, s0
    b.le Lpool_two
    fmov s0, s1
    mov w9, #1
Lpool_two:
    fcmp s2, s0
    b.le Lpool_three
    fmov s0, s2
    mov w9, #2
Lpool_three:
    fcmp s3, s0
    b.le Lpool_store
    fmov s0, s3
    mov w9, #3
Lpool_store:
    str s0, [x1], #4
    strb w9, [x2], #1
    add x0, x0, #8
    subs w7, w7, #1
    b.ne Lpool_col
    add x0, x0, x5
    subs w6, w6, #1
    b.ne Lpool_row
    ret

.globl _cnn_backward
_cnn_backward:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    PTR x9, x21, DAVG
    movi v0.4s, #0
    stp q0, q0, [x9]
    stp q0, q0, [x9, #32]
    stp q0, q0, [x9, #64]
    stp q0, q0, [x9, #96]
    PTR x10, x19, 20352
    PTR x11, x23, 20352
    PTR x12, x23, 21632
    mov w13, #10
Llinear_grad_class:
    ldr s0, [x22], #4
    dup v0.4s, v0.s[0]
    ldr s1, [x12]
    fadd s1, s1, s0
    str s1, [x12], #4
    PTR x14, x21, AVG
    PTR x15, x21, DAVG
    mov w16, #8
Llinear_grad_dot:
    ldr q1, [x14], #16
    ldr q2, [x11]
    fmla v2.4s, v1.4s, v0.4s
    str q2, [x11], #16
    ldr q1, [x10], #16
    ldr q2, [x15]
    fmla v2.4s, v1.4s, v0.4s
    str q2, [x15], #16
    subs w16, w16, #1
    b.ne Llinear_grad_dot
    subs w13, w13, #1
    b.ne Llinear_grad_class
    PTR x9, x21, DAVG
    PTR x10, x21, DP2
    mov w11, #32
    mov w12, #0x3c800000
    fmov s3, w12
Lavg_grad_channel:
    ldr s0, [x9], #4
    fmul s0, s0, s3
    dup v0.4s, v0.s[0]
    mov w12, #16
Lavg_grad_fill:
    str q0, [x10], #16
    subs w12, w12, #1
    b.ne Lavg_grad_fill
    subs w11, w11, #1
    b.ne Lavg_grad_channel
    PTR x0, x21, C2
    PTR x1, x21, DP2
    PTR x2, x21, I2
    PTR x3, x21, DC2
    mov w4, #32
    mov w5, #16
    bl Lunpool_relu
    PTR x0, x21, P1
    add x1, x19, #1792
    PTR x2, x21, DC2
    PTR x3, x21, DP1
    add x4, x23, #1792
    PTR x5, x23, 20224
    CONST x6, Lshape2
    bl _cnn_conv_backward
    mov x0, x21
    PTR x1, x21, DP1
    PTR x2, x21, I1
    PTR x3, x21, DC1
    mov w4, #16
    mov w5, #32
    bl Lunpool_relu
    mov x0, x20
    mov x1, x19
    PTR x2, x21, DC1
    mov x3, #0
    mov x4, x23
    add x5, x23, #1728
    CONST x6, Lshape1
    bl _cnn_conv_backward
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

// x0 saved relu, x1 grad pool, x2 masks, x3 grad conv; w4 C w5 W.
Lunpool_relu:
    mul w6, w5, w5
    mul w6, w6, w4
    mov x7, x3
    movi v0.4s, #0
Lunpool_zero:
    str q0, [x7], #16
    subs w6, w6, #4
    b.ne Lunpool_zero
    lsl x6, x5, #2
    lsr w7, w5, #1
    mul w7, w7, w4
Lunpool_row:
    lsr w8, w5, #1
Lunpool_col:
    ldrb w9, [x2], #1
    and w10, w9, #1
    lsl x10, x10, #2
    lsr w9, w9, #1
    madd x10, x9, x6, x10
    ldr s0, [x0, x10]
    ldr s1, [x1], #4
    fcmp s0, #0.0
    b.le Lunpool_skip
    str s1, [x3, x10]
Lunpool_skip:
    add x0, x0, #8
    add x3, x3, #8
    subs w8, w8, #1
    b.ne Lunpool_col
    add x0, x0, x6
    add x3, x3, x6
    subs w7, w7, #1
    b.ne Lunpool_row
    ret

.globl _cnn_loss
_cnn_loss:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp d8, d9, [sp, #64]
    str d10, [sp, #80]
    mov x19, x0
    mov w20, w1
    mov x21, x2
    mov x22, x3
    ldr s8, [x19]
    mov w23, #1
Lloss_max:
    ldr s0, [x19, x23, lsl #2]
    fmax s8, s8, s0
    add w23, w23, #1
    cmp w23, #10
    b.lo Lloss_max
    fmov s9, wzr
    mov w23, #0
Lloss_exp:
    ldr s0, [x19, x23, lsl #2]
    fsub s0, s0, s8
    bl _cnn_exp
    str s0, [x21, x23, lsl #2]
    fadd s9, s9, s0
    add w23, w23, #1
    cmp w23, #10
    b.lo Lloss_exp
    fmov s0, s9
    bl _cnn_log
    ldr s1, [x19, x20, lsl #2]
    fsub s1, s1, s8
    fsub s10, s0, s1
    mov w23, #0
    fmov s2, #1.0
Lloss_probs:
    ldr s0, [x21, x23, lsl #2]
    fdiv s0, s0, s9
    str s0, [x21, x23, lsl #2]
    cmp w23, w20
    b.ne Lloss_grad_store
    fsub s0, s0, s2
Lloss_grad_store:
    str s0, [x22, x23, lsl #2]
    add w23, w23, #1
    cmp w23, #10
    b.lo Lloss_probs
    fmov s0, s10
    ldr d10, [sp, #80]
    ldp d8, d9, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

// expf via double range reduction, degree-12 Taylor, and exact binary scaling.
.globl _cnn_exp
_cnn_exp:
    fcmp s0, s0
    b.vs Lexp_return
    CONST x9, Lexp_constants
    fcvt d0, s0
    ldr d1, [x9]
    fcmp d0, d1
    b.lt Lexp_zero
    ldr d1, [x9, #8]
    fcmp d0, d1
    b.gt Lexp_inf
    ldr d1, [x9, #16]
    fmul d1, d0, d1
    fcvtas x10, d1
    scvtf d1, x10
    ldr d2, [x9, #24]
    fmul d1, d1, d2
    fsub d0, d0, d1
    add x11, x9, #32
    ldr d1, [x11], #8
    mov w12, #12
Lexp_horner:
    ldr d2, [x11], #8
    fmadd d1, d1, d0, d2
    subs w12, w12, #1
    b.ne Lexp_horner
    add x10, x10, #1023
    lsl x10, x10, #52
    fmov d2, x10
    fmul d0, d1, d2
    fcvt s0, d0
Lexp_return:
    ret
Lexp_zero:
    fmov s0, wzr
    ret
Lexp_inf:
    mov w9, #0x7f800000
    fmov s0, w9
    ret

// logf via x=m*2^e, m in [1,2), y=(m-1)/(m+1), atanh series.
// Float input is represented exactly in double. Includes subnormal normalization.
.globl _cnn_log
_cnn_log:
    fcmp s0, #0.0
    b.eq Llog_minus_inf
    b.lt Llog_nan
    fcvt d0, s0
    fmov x9, d0
    ubfx x10, x9, #52, #11
    cmp x10, #2047
    b.eq Llog_return
    sub x10, x10, #1023
    and x9, x9, #0x000fffffffffffff
    mov x11, #0x3ff0000000000000
    orr x9, x9, x11
    fmov d1, x9
    fmov d2, #1.0
    fsub d0, d1, d2
    fadd d1, d1, d2
    fdiv d0, d0, d1
    fmul d1, d0, d0
    CONST x9, Llog_coefficients
    ldr d2, [x9], #8
    mov w11, #11
Llog_horner:
    ldr d3, [x9], #8
    fmadd d2, d2, d1, d3
    subs w11, w11, #1
    b.ne Llog_horner
    fmul d0, d0, d2
    fadd d0, d0, d0
    scvtf d1, x10
    ldr d2, [x9]
    fmadd d0, d1, d2, d0
Llog_return:
    fcvt s0, d0
    ret
Llog_minus_inf:
    mov w9, #0xff800000
    fmov s0, w9
    ret
Llog_nan:
    mov w9, #0x7fc00000
    fmov s0, w9
    ret

.globl _cnn_adam
_cnn_adam:
    // Doubles for Python scalar beta**step and resulting bias corrections.
    CONST x9, Ladam_constants
    ldp d0, d1, [x9]
    fmov d2, #1.0
    fmov d3, #1.0
    mov w10, w4
Ladam_power:
    tbz w10, #0, Ladam_square
    fmul d2, d2, d0
    fmul d3, d3, d1
Ladam_square:
    lsr w10, w10, #1
    cbz w10, Ladam_bias
    fmul d0, d0, d0
    fmul d1, d1, d1
    b Ladam_power
Ladam_bias:
    fmov d0, #1.0
    fsub d2, d0, d2
    fsub d3, d0, d3
    fsqrt d3, d3
    ldr d0, [x9, #16]
    fdiv d2, d0, d2
    fneg d2, d2
    fcvt s2, d2
    fcvt s3, d3
    dup v2.4s, v2.s[0] // -step_size
    dup v3.4s, v3.s[0] // sqrt(bias2)
    ldr s4, [x9, #24]
    ldr s5, [x9, #28]
    ldr s6, [x9, #32]
    ldr s7, [x9, #36]
    dup v4.4s, v4.s[0] // .1
    dup v5.4s, v5.s[0] // .999
    dup v6.4s, v6.s[0] // .001
    dup v7.4s, v7.s[0] // epsilon
    mov w10, #1354 // 5416 parameters in SIMD, then two scalar.
Ladam_vector:
    ldr q16, [x0]
    ldr q17, [x1], #16
    ldr q18, [x2]
    ldr q19, [x3]
    fsub v20.4s, v17.4s, v18.4s
    fmla v18.4s, v20.4s, v4.4s
    fmul v19.4s, v19.4s, v5.4s
    fmul v20.4s, v17.4s, v17.4s
    fmla v19.4s, v20.4s, v6.4s
    fsqrt v20.4s, v19.4s
    fdiv v20.4s, v20.4s, v3.4s
    fadd v20.4s, v20.4s, v7.4s
    fmul v21.4s, v18.4s, v2.4s
    fdiv v21.4s, v21.4s, v20.4s
    fadd v16.4s, v16.4s, v21.4s
    str q18, [x2], #16
    str q19, [x3], #16
    str q16, [x0], #16
    subs w10, w10, #1
    b.ne Ladam_vector
    mov w10, #2
Ladam_scalar:
    ldr s16, [x0]
    ldr s17, [x1], #4
    ldr s18, [x2]
    ldr s19, [x3]
    fsub s20, s17, s18
    fmadd s18, s20, s4, s18
    fmul s19, s19, s5
    fmul s20, s17, s17
    fmadd s19, s20, s6, s19
    fsqrt s20, s19
    fdiv s20, s20, s3
    fadd s20, s20, s7
    fmul s21, s18, s2
    fdiv s21, s21, s20
    fadd s16, s16, s21
    str s18, [x2], #4
    str s19, [x3], #4
    str s16, [x0], #4
    subs w10, w10, #1
    b.ne Ladam_scalar
    ret

.section __TEXT,__const
.p2align 3
Lshape1: .long 3,32,16
Lshape2: .long 16,16,32
.p2align 3
Lexp_constants:
    .double -104.0, 88.72283905206835, 1.4426950408889634074, 0.69314718055994530942
    .double 0.0000000020876756987868098979
    .double 0.000000025052108385441718775
    .double 0.00000027557319223985890653
    .double 0.0000027557319223985890653
    .double 0.000024801587301587301587
    .double 0.00019841269841269841270
    .double 0.0013888888888888888889
    .double 0.0083333333333333333333
    .double 0.041666666666666666667
    .double 0.16666666666666666667
    .double 0.5, 1.0, 1.0
Llog_coefficients:
    .double 0.043478260869565217391 // 1/23 through 1/1
    .double 0.047619047619047619048
    .double 0.052631578947368421053
    .double 0.058823529411764705882
    .double 0.066666666666666666667
    .double 0.076923076923076923077
    .double 0.090909090909090909091
    .double 0.11111111111111111111
    .double 0.14285714285714285714
    .double 0.2, 0.33333333333333333333, 1.0
    .double 0.69314718055994530942
Ladam_constants:
    .double 0.9, 0.999, 0.001
    .float 0.1, 0.999, 0.001, 0.00000001
