// Handwritten Apple ARM64 standalone training runtime. See ABI.md/RUNTIME.md.
// C library calls provide only operating-system/process/I/O/memory services.
.text
.p2align 2
.macro ADDR reg, sym
 adrp \reg, \sym@PAGE
 add \reg, \reg, \sym@PAGEOFF
.endm
.macro LD reg, sym
 ADDR x9, \sym
 ldr \reg, [x9]
.endm
.macro ST reg, sym
 ADDR x9, \sym
 str \reg, [x9]
.endm
.macro ENTER
 stp x29,x30,[sp,#-160]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 stp x23,x24,[sp,#48]
 stp x25,x26,[sp,#64]
 stp x27,x28,[sp,#80]
 stp d8,d9,[sp,#96]
 stp d10,d11,[sp,#112]
 stp d12,d13,[sp,#128]
 stp d14,d15,[sp,#144]
.endm
.macro LEAVE
 ldp x19,x20,[sp,#16]
 ldp x21,x22,[sp,#32]
 ldp x23,x24,[sp,#48]
 ldp x25,x26,[sp,#64]
 ldp x27,x28,[sp,#80]
 ldp d8,d9,[sp,#96]
 ldp d10,d11,[sp,#112]
 ldp d12,d13,[sp,#128]
 ldp d14,d15,[sp,#144]
 ldp x29,x30,[sp],#160
 ret
.endm
.macro CHECK_ZERO
 cbnz w0,Lfatal
.endm

// x0 = malloc size; fatal allocation failure.
_rt_alloc:
 stp x29,x30,[sp,#-16]!
 bl _malloc
 cbz x0,Lfatal
 ldp x29,x30,[sp],#16
 ret

// Seconds from mach_absolute_time, double precision.
_rt_now:
 stp x29,x30,[sp,#-16]!
 bl _mach_absolute_time
 ucvtf d0,x0
 LD d1,_rt_tick_seconds
 fmul d0,d0,d1
 ldp x29,x30,[sp],#16
 ret

// x0 root, x1 basename, x2 destination, both root/name <= 1800/100.
_rt_path:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 bl _strlen
 cmp x0,#1800
 b.hi Lfatal
 mov x0,x21
 mov x1,x19
 bl _strcpy
 mov x0,x21
 ADDR x1,Lslash
 bl _strcat
 mov x0,x21
 mov x1,x20
 bl _strcat
 mov x0,x21
 LEAVE

// Exact size, checked fread and EOF. x0 basename, x1 expected byte size.
_rt_read:
 ENTER
 sub sp,sp,#2048
 mov x19,x0
 mov x20,x1
 LD x0,_rt_input_root
 mov x1,x19
 mov x2,sp
 bl _rt_path
 ADDR x1,Lrb
 bl _fopen
 cbz x0,Lfilefatal
 mov x21,x0
 mov x0,x20
 bl _rt_alloc
 mov x22,x0
 mov x1,#1
 mov x2,x20
 mov x3,x21
 bl _fread
 cmp x0,x20
 b.ne Lfilefatal
 mov x0,x21
 bl _fgetc
 cmn w0,#1
 b.ne Lfilefatal
 mov x0,x21
 bl _ferror
 cbnz w0,Lfilefatal
 mov x0,x21
 bl _fclose
 cbnz w0,Lfilefatal
 mov x0,x22
 add sp,sp,#2048
 LEAVE
Lfilefatal:
 mov x0,sp
 bl _puts
 b Lfatal

// x0 basename, x1 bytes pointer, x2 byte count.
_rt_write:
 ENTER
 sub sp,sp,#2048
 mov x19,x0
 mov x20,x1
 mov x21,x2
 LD x0,_rt_output_root
 mov x1,x19
 mov x2,sp
 bl _rt_path
 ADDR x1,Lwb
 bl _fopen
 cbz x0,Lfilefatal
 mov x22,x0
 mov x0,x20
 mov x1,#1
 mov x2,x21
 mov x3,x22
 bl _fwrite
 cmp x0,x21
 b.ne Lfilefatal
 mov x0,x22
 bl _fclose
 cbnz w0,Lfilefatal
 add sp,sp,#2048
 LEAVE

// Accept a newly created or empty directory; never overwrite existing results.
_rt_directory:
 ENTER
 LD x0,_rt_output_root
 mov w1,#493
 bl _mkdir
 cbz w0,Ldir_open
 bl ___error
 ldr w0,[x0]
 cmp w0,#17
 b.ne Lfatal
Ldir_open:
 LD x0,_rt_output_root
 bl _opendir
 cbz x0,Lfatal
 mov x19,x0
Ldir_read:
 bl ___error
 str wzr,[x0]
 mov x0,x19
 bl _readdir
 cbz x0,Ldir_check_end
 add x20,x0,#21
 ldrb w0,[x20]
 cmp w0,#46
 b.ne Lnonempty
 ldrb w0,[x20,#1]
 cbz w0,Ldir_read
 cmp w0,#46
 b.ne Lnonempty
 ldrb w0,[x20,#2]
 cbz w0,Ldir_read
Lnonempty:
 ADDR x0,Lexisting
 bl _puts
 b Lfatal
Ldir_check_end:
 bl ___error
 ldr w0,[x0]
 cbnz w0,Lfatal
Ldir_end:
 mov x0,x19
 bl _closedir
 CHECK_ZERO
 LEAVE

// Float32 torchvision ToTensor then Normalize: (pixel/255 - .5)/.5.
// x0 points to 3072 NHWC uint8 pixels, w1 flip, x2 3072 CHW float output.
.globl _cnn_transform
_cnn_transform:
 fmov s2,#0.5
 mov w3,#0x437f0000
 fmov s3,w3
 mov w4,#0
Ltransform_c:
 mov w5,#0
Ltransform_y:
 mov w6,#0
Ltransform_x:
 mov w7,#31
 sub w7,w7,w6
 cmp w1,#0
 csel w7,w6,w7,eq
 lsl w8,w5,#5
 add w8,w8,w7
 add w8,w8,w8,lsl #1
 add w8,w8,w4
 ldrb w10,[x0,w8,uxtw]
 ucvtf s0,w10
 fdiv s0,s0,s3
 fsub s0,s0,s2
 fdiv s0,s0,s2
 str s0,[x2],#4
 add w6,w6,#1
 cmp w6,#32
 b.lo Ltransform_x
 add w5,w5,#1
 cmp w5,#32
 b.lo Ltransform_y
 add w4,w4,#1
 cmp w4,#3
 b.lo Ltransform_c
 ret

// Worker record offsets: id0 workspace8 image16 logits24 probs32 dlogits40
// gradients48 loss56 correct64 lastgeneration72. 96 bytes total.
// Job modes: 1 training; 2 evaluation; 3 forward-only timing; 4 shutdown.
_rt_worker:
 ENTER
 mov x19,x0
Lworker_wait:
 ADDR x0,_rt_mutex
 bl _pthread_mutex_lock
 CHECK_ZERO
Lworker_predicate:
 LD x20,_rt_generation
 ldr x21,[x19,#72]
 cmp x20,x21
 b.ne Lworker_ready
 ADDR x0,_rt_job_cond
 ADDR x1,_rt_mutex
 bl _pthread_cond_wait
 CHECK_ZERO
 b Lworker_predicate
Lworker_ready:
 str x20,[x19,#72]
 LD w22,_rt_mode
 LD x23,_rt_job_count
 ldr x20,[x19]
 mul x21,x23,x20
 lsr x21,x21,#1
 add x20,x20,#1
 mul x23,x23,x20
 lsr x23,x23,#1
 ADDR x0,_rt_mutex
 bl _pthread_mutex_unlock
 CHECK_ZERO
 cmp w22,#4
 b.eq Lworker_exit
 str xzr,[x19,#56]
 str xzr,[x19,#64]
 cmp w22,#1
 b.ne Lworker_loop
 ldr x0,[x19,#48]
 mov w1,#0
 mov x2,#21672
 bl _memset
Lworker_loop:
 cmp x21,x23
 b.hs Lworker_done
 LD x24,_rt_job_records
 cmp w22,#3
 b.eq Lworker_timing
 cmp w22,#1
 b.ne Lworker_eval_index
 add x24,x24,x21,lsl #3
 ldp w25,w1,[x24]
 cmp w1,#1
 b.hi Lfatal
 b Lworker_image
Lworker_eval_index:
 ldr w25,[x24,x21,lsl #2]
 mov w1,#0
Lworker_image:
 mov w0,#60000
 mov w2,#50000
 cmp w22,#1
 csel w0,w2,w0,eq
 cmp w25,w0
 b.hs Lfatal
 LD x0,_rt_images
 mov x26,#3072
 madd x0,x25,x26,x0
 ldr x2,[x19,#16]
 bl _cnn_transform
 LD x0,_rt_params
 ldr x1,[x19,#16]
 ldr x2,[x19,#8]
 ldr x3,[x19,#24]
 bl _cnn_forward
 ldr x0,[x19,#24]
 LD x1,_rt_labels
 ldr w1,[x1,x25,lsl #2]
 cmp w1,#10
 b.hs Lfatal
 mov w26,w1
 ldr x2,[x19,#32]
 ldr x3,[x19,#40]
 bl _cnn_loss
 fmov w0,s0
 and w0,w0,#0x7f800000
 mov w1,#0x7f800000
 cmp w0,w1
 b.eq Lnumericfatal
 fcvt d0,s0
 ldr d1,[x19,#56]
 fadd d0,d0,d1
 str d0,[x19,#56]
 // Probabilities are checked independently; baseline accuracy uses logits.
 ldr x24,[x19,#32]
 mov w27,#0
Lworker_probability_finite:
 ldr w0,[x24,x27,lsl #2]
 and w0,w0,#0x7f800000
 mov w1,#0x7f800000
 cmp w0,w1
 b.eq Lnumericfatal
 add w27,w27,#1
 cmp w27,#10
 b.lo Lworker_probability_finite
 // Strict greater-than keeps the first index when logits tie.
 ldr x24,[x19,#24]
 ldr s1,[x24]
 mov w27,#0
 mov w28,#0
Lworker_argmax:
 ldr s0,[x24,x27,lsl #2]
 fmov w0,s0
 and w0,w0,#0x7f800000
 mov w1,#0x7f800000
 cmp w0,w1
 b.eq Lnumericfatal
 fcmp s0,s1
 b.le Lworker_notmax
 fmov s1,s0
 mov w28,w27
Lworker_notmax:
 add w27,w27,#1
 cmp w27,#10
 b.lo Lworker_argmax
 cmp w28,w26
 b.ne Lworker_prediction
 ldr x0,[x19,#64]
 add x0,x0,#1
 str x0,[x19,#64]
Lworker_prediction:
 cmp w22,#1
 b.eq Lworker_backward
 LD x0,_rt_job_predictions
 cbz x0,Lworker_next
 mov x1,#40
 madd x0,x21,x1,x0
 ldr x1,[x19,#32]
 mov x2,#40
 bl _memcpy
 b Lworker_next
Lworker_backward:
 LD x0,_rt_params
 ldr x1,[x19,#16]
 ldr x2,[x19,#8]
 ldr x3,[x19,#40]
 ldr x4,[x19,#48]
 bl _cnn_backward
 b Lworker_next
Lworker_timing:
 mov x0,#12288
 madd x1,x21,x0,x24
 LD x0,_rt_params
 ldr x2,[x19,#8]
 ldr x3,[x19,#24]
 bl _cnn_forward
Lworker_next:
 add x21,x21,#1
 b Lworker_loop
Lworker_done:
 ADDR x0,_rt_mutex
 bl _pthread_mutex_lock
 CHECK_ZERO
 LD x0,_rt_done
 add x0,x0,#1
 ST x0,_rt_done
 cmp x0,#2
 b.ne Lworker_unlock
 ADDR x0,_rt_done_cond
 bl _pthread_cond_signal
 CHECK_ZERO
Lworker_unlock:
 ADDR x0,_rt_mutex
 bl _pthread_mutex_unlock
 CHECK_ZERO
 b Lworker_wait
Lworker_exit:
 mov x0,#0
 LEAVE

// Publish one immutable job under mutex; block until both workers complete.
// w0 mode, x1 records, x2 count, x3 probabilities output or NULL.
_rt_dispatch:
 ENTER
 mov w19,w0
 mov x20,x1
 mov x21,x2
 mov x22,x3
 ADDR x0,_rt_mutex
 bl _pthread_mutex_lock
 CHECK_ZERO
 ST w19,_rt_mode
 ST x20,_rt_job_records
 ST x21,_rt_job_count
 ST x22,_rt_job_predictions
 ST xzr,_rt_done
 LD x0,_rt_generation
 add x0,x0,#1
 ST x0,_rt_generation
 ADDR x0,_rt_job_cond
 bl _pthread_cond_broadcast
 CHECK_ZERO
Ldispatch_wait:
 LD x0,_rt_done
 cmp x0,#2
 b.eq Ldispatch_done
 ADDR x0,_rt_done_cond
 ADDR x1,_rt_mutex
 bl _pthread_cond_wait
 CHECK_ZERO
 b Ldispatch_wait
Ldispatch_done:
 ADDR x0,_rt_mutex
 bl _pthread_mutex_unlock
 CHECK_ZERO
 ADDR x0,_rt_workers
 ldr d0,[x0,#56]
 ldr d1,[x0,#152]
 fadd d0,d0,d1
 ldr x1,[x0,#64]
 ldr x2,[x0,#160]
 add x0,x1,x2
 LEAVE

_rt_start_workers:
 ENTER
 ADDR x0,_rt_mutex
 mov x1,#0
 bl _pthread_mutex_init
 CHECK_ZERO
 ADDR x0,_rt_job_cond
 mov x1,#0
 bl _pthread_cond_init
 CHECK_ZERO
 ADDR x0,_rt_done_cond
 mov x1,#0
 bl _pthread_cond_init
 CHECK_ZERO
 mov x19,#0
 ADDR x20,_rt_workers
Lstart_worker:
 str x19,[x20]
 bl _cnn_workspace_bytes
 bl _rt_alloc
 str x0,[x20,#8]
 mov x0,#12288
 bl _rt_alloc
 str x0,[x20,#16]
 mov x0,#40
 bl _rt_alloc
 str x0,[x20,#24]
 mov x0,#40
 bl _rt_alloc
 str x0,[x20,#32]
 mov x0,#40
 bl _rt_alloc
 str x0,[x20,#40]
 mov x0,#21672
 bl _rt_alloc
 str x0,[x20,#48]
 ADDR x0,_rt_threads
 add x0,x0,x19,lsl #3
 mov x1,#0
 ADDR x2,_rt_worker
 mov x3,x20
 bl _pthread_create
 CHECK_ZERO
 add x20,x20,#96
 add x19,x19,#1
 cmp x19,#2
 b.lo Lstart_worker
 LEAVE

_rt_stop_workers:
 ENTER
 ADDR x0,_rt_mutex
 bl _pthread_mutex_lock
 CHECK_ZERO
 mov w0,#4
 ST w0,_rt_mode
 LD x0,_rt_generation
 add x0,x0,#1
 ST x0,_rt_generation
 ADDR x0,_rt_job_cond
 bl _pthread_cond_broadcast
 CHECK_ZERO
 ADDR x0,_rt_mutex
 bl _pthread_mutex_unlock
 CHECK_ZERO
 ADDR x19,_rt_threads
 mov x20,#0
Ljoin:
 ldr x0,[x19,x20,lsl #3]
 mov x1,#0
 bl _pthread_join
 CHECK_ZERO
 add x20,x20,#1
 cmp x20,#2
 b.lo Ljoin
 LEAVE

// Sum workers in fixed order, mean by actual batch size, then Adam.
_rt_update:
 ENTER
 mov x19,x0
 ADDR x1,_rt_workers
 ldr x2,[x1,#48]
 ldr x3,[x1,#144]
 LD x4,_rt_mean_gradient
 ucvtf s2,x19
 mov x5,#5418
Lgradient_mean:
 ldr s0,[x2],#4
 ldr s1,[x3],#4
 fadd s0,s0,s1
 fdiv s0,s0,s2
 fmov w6,s0
 and w6,w6,#0x7f800000
 mov w7,#0x7f800000
 cmp w6,w7
 b.eq Lnumericfatal
 str s0,[x4],#4
 subs x5,x5,#1
 b.ne Lgradient_mean
 LD w4,_rt_step
 add w4,w4,#1
 ST w4,_rt_step
 LD x0,_rt_params
 LD x1,_rt_mean_gradient
 LD x2,_rt_first_moment
 LD x3,_rt_second_moment
 bl _cnn_adam
 LD x0,_rt_params
 mov x1,#5418
 bl _rt_finite_array
 LD x0,_rt_first_moment
 mov x1,#5418
 bl _rt_finite_array
 LD x0,_rt_second_moment
 mov x1,#5418
 bl _rt_finite_array
 LEAVE

// x0 float32 array, x1 number of entries; reject all nonfinite state.
_rt_finite_array:
 ldr w2,[x0],#4
 and w2,w2,#0x7f800000
 mov w3,#0x7f800000
 cmp w2,w3
 b.eq Lnumericfatal
 subs x1,x1,#1
 b.ne _rt_finite_array
 ret

// Evaluate a split in batches (without augmentation); return loss sum d0,
// correct count x0. x0 indices,x1 count,x2 probabilities or NULL.
_rt_evaluate:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 mov x22,#0
 mov x23,#0
 fmov d8,xzr
Leval_batch:
 sub x24,x20,x22
 mov x0,#128
 cmp x24,x0
 csel x24,x24,x0,lo
 mov w0,#2
 add x1,x19,x22,lsl #2
 mov x2,x24
 mov x3,#40
 madd x3,x22,x3,x21
 cmp x21,#0
 csel x3,xzr,x3,eq
 bl _rt_dispatch
 fadd d8,d8,d0
 add x23,x23,x0
 add x22,x22,x24
 cmp x22,x20
 b.lo Leval_batch
 fmov d0,d8
 mov x0,x23
 LEAVE

// One final split; x0 indices,x1 count,x2 filename. Save time in d0.
_rt_final_split:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 bl _rt_now
 fmov d8,d0
 mov x0,x19
 mov x1,x20
 LD x2,_rt_predictions
 bl _rt_evaluate
 bl _rt_now
 fsub d8,d0,d8
 mov x0,x21
 LD x1,_rt_predictions
 mov x2,#40
 mul x2,x2,x20
 bl _rt_write
 fmov d0,d8
 LEAVE

// Forward only batch timing. Fixed normalized images, no loss or transforms.
// w0 batch size, x1 output filename. 20 warmups and 100 timed calls.
_rt_timing:
 ENTER
 mov w19,w0
 mov x20,x1
 mov x21,#0
 ADDR x22,_rt_timing_values
Ltiming_pass:
 cmp x21,#20
 b.lo Ltiming_run
 bl _rt_now
 fmov d8,d0
Ltiming_run:
 cmp w19,#1
 b.ne Ltiming_batch
 // Like PyTorch batch1 timing, avoid thread-dispatch overhead for one image.
 LD x0,_rt_params
 LD x1,_rt_timing_images
 ADDR x2,_rt_workers
 ldr x3,[x2,#24]
 ldr x2,[x2,#8]
 bl _cnn_forward
 b Ltiming_finish
Ltiming_batch:
 mov w0,#3
 LD x1,_rt_timing_images
 mov x2,#128
 mov x3,#0
 bl _rt_dispatch
Ltiming_finish:
 cmp x21,#20
 b.lo Ltiming_next
 bl _rt_now
 fsub d0,d0,d8
 mov x0,#1000
 ucvtf d1,x0
 fmul d0,d0,d1
 str d0,[x22],#8
Ltiming_next:
 add x21,x21,#1
 cmp x21,#120
 b.lo Ltiming_pass
 mov x0,x20
 ADDR x1,_rt_timing_values
 mov x2,#800
 bl _rt_write
 LEAVE

.globl _main
_main:
 ENTER
 cmp w0,#3
 b.lo Lusage
 cmp w0,#5
 b.hi Lusage
 mov w19,w0
 mov x20,x1
 ldr x0,[x20,#8]
 ST x0,_rt_input_root
 ldr x0,[x20,#16]
 ST x0,_rt_output_root
 mov x21,#25
 mov x22,#45000
 cmp w19,#4
 b.lo Lmain_args
 ldr x0,[x20,#24]
 bl _atoi
 cmp w0,#1
 b.lt Lusage
 cmp w0,#25
 b.gt Lusage
 mov w21,w0
 cmp w19,#5
 b.lo Lmain_args
 ldr x0,[x20,#32]
 bl _atoi
 cmp w0,#1
 b.lt Lusage
 mov w1,#45000
 cmp w0,w1
 b.gt Lusage
 mov w22,w0
Lmain_args:
 ST x21,_rt_epochs
 ST x22,_rt_train_count
 mov x0,#45000
 ST x0,_rt_eval_train_count
 mov x0,#5000
 ST x0,_rt_eval_val_count
 mov x0,#10000
 ST x0,_rt_eval_test_count
 ADDR x0,Leval_limit_name
 bl _getenv
 cbz x0,Lmain_limits_done
 bl _atoi
 cmp w0,#1
 b.lt Lusage
 mov w1,#45000
 cmp w0,w1
 b.gt Lusage
 uxtw x0,w0
 ST x0,_rt_eval_train_count
 mov x1,#5000
 cmp x0,x1
 csel x1,x0,x1,lo
 ST x1,_rt_eval_val_count
 mov x1,#10000
 cmp x0,x1
 csel x1,x0,x1,lo
 ST x1,_rt_eval_test_count
Lmain_limits_done:
 bl _rt_directory
 ADDR x0,_rt_timebase
 bl _mach_timebase_info
 CHECK_ZERO
 ADDR x0,_rt_timebase
 ldp w1,w2,[x0]
 ucvtf d0,w1
 ucvtf d1,w2
 fdiv d0,d0,d1
 ADDR x0,Lbillion
 ldr d1,[x0]
 fdiv d0,d0,d1
 ST d0,_rt_tick_seconds
 ADDR x0,Limages
 mov x1,#32768
 movk x1,#2812,lsl #16
 // 60,000 * 3,072 = 184,320,000 = 0x0afc8000.
 bl _rt_read
 ST x0,_rt_images
 ADDR x0,Llabels
 mov x1,#43392
 movk x1,#3,lsl #16
 bl _rt_read
 ST x0,_rt_labels
 ADDR x0,Ltrain_indices
 mov x1,#48928
 movk x1,#2,lsl #16
 bl _rt_read
 ST x0,_rt_train_indices
 ADDR x0,Lval_indices
 mov x1,#20000
 bl _rt_read
 ST x0,_rt_val_indices
 ADDR x0,Linit
 mov x1,#21672
 bl _rt_read
 ST x0,_rt_params
 ADDR x0,Lschedule
 mov x1,#45000
 mul x1,x1,x21
 lsl x1,x1,#3
 bl _rt_read
 ST x0,_rt_schedule
 mov x0,#21672
 bl _rt_alloc
 ST x0,_rt_best
 mov x0,#21672
 bl _rt_alloc
 ST x0,_rt_mean_gradient
 mov x0,#1
 mov x1,#21672
 bl _calloc
 cbz x0,Lfatal
 ST x0,_rt_first_moment
 mov x0,#1
 mov x1,#21672
 bl _calloc
 cbz x0,Lfatal
 ST x0,_rt_second_moment
 mov x0,#40000
 bl _rt_alloc
 ST x0,_rt_test_indices
 mov w1,#50000
 mov w2,#10000
Ltest_indices:
 str w1,[x0],#4
 add w1,w1,#1
 subs w2,w2,#1
 b.ne Ltest_indices
 mov x0,#30528
 movk x0,#27,lsl #16
 // 1,800,000 bytes = 45,000 * 10 * 4.
 bl _rt_alloc
 ST x0,_rt_predictions
 // Validate all input labels and all split indices before computation.
 LD x0,_rt_labels
 mov x1,#60000
Lvalidate_labels:
 ldr w2,[x0],#4
 cmp w2,#10
 b.hs Lfatal
 subs x1,x1,#1
 b.ne Lvalidate_labels
 LD x0,_rt_train_indices
 mov x1,#45000
 bl _rt_validate_indices
 LD x0,_rt_val_indices
 mov x1,#5000
 bl _rt_validate_indices
 LD x0,_rt_params
 mov x1,#5418
Lvalidate_init:
 ldr w2,[x0],#4
 and w2,w2,#0x7f800000
 mov w3,#0x7f800000
 cmp w2,w3
 b.eq Lnumericfatal
 subs x1,x1,#1
 b.ne Lvalidate_init
 bl _rt_validate_protocol
 bl _rt_start_workers
 // Open CSV once; flush each epoch so interrupted runs remain inspectable.
 sub sp,sp,#2048
 LD x0,_rt_output_root
 ADDR x1,Lhistory
 mov x2,sp
 bl _rt_path
 ADDR x1,Lwb
 bl _fopen
 cbz x0,Lfilefatal
 ST x0,_rt_history
 mov x1,x0
 ADDR x0,Lhistory_header
 bl _fputs
 tbnz w0,#31,Lfatal
 add sp,sp,#2048
 fmov d0,#-1.0
 ST d0,_rt_best_accuracy
 mov x19,#1
 bl _rt_now
 ST d0,_rt_training_start
Lepoch:
 bl _rt_now
 fmov d9,d0
 fmov d8,d0
 fmov d12,xzr
 mov x23,#0
 mov x20,#0
Ltrain_batch:
 sub x24,x22,x20
 mov x0,#128
 cmp x24,x0
 csel x24,x24,x0,lo
 sub x1,x19,#1
 mov x0,#45000
 mul x1,x1,x0
 add x1,x1,x20
 LD x0,_rt_schedule
 add x1,x0,x1,lsl #3
 mov w0,#1
 mov x2,x24
 mov x3,#0
 bl _rt_dispatch
 fadd d12,d12,d0
 add x23,x23,x0
 mov x0,x24
 bl _rt_update
 add x20,x20,x24
 cmp x20,x22
 b.lo Ltrain_batch
 bl _rt_now
 fsub d10,d0,d8
 ST d10,_rt_epoch_train_seconds
 ucvtf d0,x22
 fdiv d12,d12,d0
 ucvtf d1,x23
 fdiv d13,d1,d0
 ST d12,_rt_epoch_train_loss
 ST d13,_rt_epoch_train_accuracy
 bl _rt_now
 fmov d8,d0
 LD x0,_rt_val_indices
 LD x1,_rt_eval_val_count
 mov x2,#0
 bl _rt_evaluate
 LD x1,_rt_eval_val_count
 ucvtf d1,x1
 fdiv d14,d0,d1
 ucvtf d0,x0
 fdiv d15,d0,d1
 ST d14,_rt_epoch_val_loss
 ST d15,_rt_epoch_val_accuracy
 bl _rt_now
 fsub d11,d0,d8
 ST d11,_rt_epoch_val_seconds
 LD d0,_rt_best_accuracy
 fcmp d15,d0
 b.gt Lselect
 b.lt Lselected
 LD d0,_rt_best_loss
 fcmp d14,d0
 b.ge Lselected
Lselect:
 ST d15,_rt_best_accuracy
 ST d14,_rt_best_loss
 ST x19,_rt_best_epoch
 LD x0,_rt_best
 LD x1,_rt_params
 mov x2,#21672
 bl _memcpy
 ADDR x0,Lbest
 LD x1,_rt_best
 mov x2,#21672
 bl _rt_write
Lselected:
 bl _rt_now
 fsub d0,d0,d9
 ST d0,_rt_epoch_seconds
 // Darwin variadic arguments are passed in consecutive 8-byte stack slots.
 sub sp,sp,#80
 str x19,[sp]
 str d12,[sp,#8]
 str d13,[sp,#16]
 str d14,[sp,#24]
 str d15,[sp,#32]
 str d10,[sp,#40]
 str d11,[sp,#48]
 str d0,[sp,#56]
 LD x0,_rt_best_epoch
 str x0,[sp,#64]
 LD x0,_rt_history
 ADDR x1,Lhistory_format
 bl _fprintf
 tbnz w0,#31,Lfatal
 add sp,sp,#80
 LD x0,_rt_history
 bl _fflush
 CHECK_ZERO
 sub sp,sp,#64
 str x19,[sp]
 str x21,[sp,#8]
 str d12,[sp,#16]
 str d13,[sp,#24]
 str d14,[sp,#32]
 str d15,[sp,#40]
 LD d0,_rt_epoch_seconds
 str d0,[sp,#48]
 ADDR x0,Lprogress
 bl _printf
 add sp,sp,#64
 mov x0,#0
 bl _fflush
 CHECK_ZERO
 add x19,x19,#1
 cmp x19,x21
 b.ls Lepoch
 bl _rt_now
 LD d1,_rt_training_start
 fsub d0,d0,d1
 ST d0,_rt_training_seconds
 LD x0,_rt_history
 bl _fclose
 CHECK_ZERO
 ADDR x0,Llast
 LD x1,_rt_params
 mov x2,#21672
 bl _rt_write
 LD x0,_rt_params
 LD x1,_rt_best
 mov x2,#21672
 bl _memcpy
 bl _rt_now
 fmov d8,d0
 LD x0,_rt_train_indices
 LD x1,_rt_eval_train_count
 ADDR x2,Lpred_train
 bl _rt_final_split
 ST d0,_rt_train_prediction_seconds
 LD x0,_rt_val_indices
 LD x1,_rt_eval_val_count
 ADDR x2,Lpred_val
 bl _rt_final_split
 ST d0,_rt_val_prediction_seconds
 LD x0,_rt_test_indices
 LD x1,_rt_eval_test_count
 ADDR x2,Lpred_test
 bl _rt_final_split
 ST d0,_rt_test_prediction_seconds
 bl _rt_now
 fsub d0,d0,d8
 ST d0,_rt_final_evaluation_seconds
 // Normalize the same first 128 official test images outside timed scope.
 mov x0,#24
 lsl x0,x0,#16
 bl _rt_alloc
 ST x0,_rt_timing_images
 mov x19,#0
Ltiming_prepare:
 mov x0,#50000
 add x0,x0,x19
 mov x1,#3072
 LD x2,_rt_images
 madd x0,x0,x1,x2
 mov x1,#12288
 LD x2,_rt_timing_images
 madd x2,x19,x1,x2
 mov w1,#0
 bl _cnn_transform
 add x19,x19,#1
 cmp x19,#128
 b.lo Ltiming_prepare
 mov w0,#1
 ADDR x1,Ltiming1
 bl _rt_timing
 mov w0,#128
 ADDR x1,Ltiming128
 bl _rt_timing
 bl _rt_stop_workers
 bl _rt_performance
 ADDR x0,Lcomplete
 bl _puts
 mov w0,#0
 LEAVE

// All split indices must refer to original training images and be sorted.
_rt_validate_indices:
 mov w3,#0
 mov w4,#0
Lvalidate_indices:
 ldr w2,[x0],#4
 mov w5,#50000
 cmp w2,w5
 b.hs Lfatal
 cbz w4,Lvalidate_index_first
 cmp w2,w3
 b.ls Lfatal
Lvalidate_index_first:
 mov w3,w2
 mov w4,#1
 subs x1,x1,#1
 b.ne Lvalidate_indices
 ret

// Verify split disjointness and exact schedule membership before worker launch.
// Sorted unique split indices have already been range-checked. Each epoch must
// be a permutation of all 45,000 training examples with valid binary flips.
_rt_validate_protocol:
 ENTER
 ADDR x19,_rt_membership
 LD x0,_rt_train_indices
 mov x1,#45000
 mov w3,#1
Lmembership_train:
 ldr w2,[x0],#4
 strb w3,[x19,x2]
 subs x1,x1,#1
 b.ne Lmembership_train
 LD x0,_rt_val_indices
 mov x1,#5000
 mov w3,#2
Lmembership_validation:
 ldr w2,[x0],#4
 ldrb w4,[x19,x2]
 cbnz w4,Lfatal
 strb w3,[x19,x2]
 subs x1,x1,#1
 b.ne Lmembership_validation
 LD x20,_rt_schedule
 LD x21,_rt_epochs
 ADDR x22,_rt_seen
Lschedule_epoch:
 mov x0,x22
 mov w1,#0
 mov x2,#50000
 bl _memset
 mov x23,#45000
 mov w5,#1
Lschedule_record:
 ldp w0,w1,[x20],#8
 mov w2,#50000
 cmp w0,w2
 b.hs Lfatal
 cmp w1,#1
 b.hi Lfatal
 ldrb w2,[x19,x0]
 cmp w2,#1
 b.ne Lfatal
 ldrb w2,[x22,x0]
 cbnz w2,Lfatal
 strb w5,[x22,x0]
 subs x23,x23,#1
 b.ne Lschedule_record
 subs x21,x21,#1
 b.ne Lschedule_epoch
 LEAVE

_rt_performance:
 ENTER
 sub sp,sp,#2048
 mov w0,#0
 ADDR x1,_rt_rusage
 bl _getrusage
 CHECK_ZERO
 ADDR x19,_rt_rusage
 ldr x0,[x19]
 scvtf d8,x0
 ldr w0,[x19,#8]
 scvtf d0,w0
 ADDR x0,Lmillion
 ldr d1,[x0]
 fdiv d0,d0,d1
 fadd d8,d8,d0
 ldr x0,[x19,#16]
 scvtf d9,x0
 ldr w0,[x19,#24]
 scvtf d0,w0
 fdiv d0,d0,d1
 fadd d9,d9,d0
 LD x0,_rt_output_root
 ADDR x1,Lperformance
 mov x2,sp
 bl _rt_path
 ADDR x1,Lwb
 bl _fopen
 cbz x0,Lfilefatal
 mov x20,x0
 sub sp,sp,#128
 LD d0,_rt_training_seconds
 str d0,[sp]
 LD x0,_rt_best_epoch
 str x0,[sp,#8]
 LD d0,_rt_final_evaluation_seconds
 str d0,[sp,#16]
 LD d0,_rt_train_prediction_seconds
 str d0,[sp,#24]
 LD d0,_rt_val_prediction_seconds
 str d0,[sp,#32]
 LD d0,_rt_test_prediction_seconds
 str d0,[sp,#40]
 ldr x0,[x19,#32]
 str x0,[sp,#48]
 str d8,[sp,#56]
 str d9,[sp,#64]
 LD x0,_rt_epochs
 str x0,[sp,#72]
 LD x0,_rt_train_count
 str x0,[sp,#80]
 LD x0,_rt_eval_train_count
 str x0,[sp,#88]
 LD x0,_rt_eval_val_count
 str x0,[sp,#96]
 LD x0,_rt_eval_test_count
 str x0,[sp,#104]
 mov x0,x20
 ADDR x1,Lperformance_format
 bl _fprintf
 tbnz w0,#31,Lfatal
 add sp,sp,#128
 mov x0,x20
 bl _fclose
 CHECK_ZERO
 add sp,sp,#2048
 LEAVE

Lusage:
 ADDR x0,Lusage_message
 bl _puts
 mov w0,#2
 bl _exit
Lnumericfatal:
 ADDR x0,Lnumeric_message
 bl _puts
 b Lfatal
Lfatal:
 ADDR x0,Lfatal_message
 bl _puts
 mov w0,#1
 bl _exit

.section __TEXT,__cstring,cstring_literals
Lslash: .asciz "/"
Lrb: .asciz "rb"
Lwb: .asciz "wb"
Limages: .asciz "images.bin"
Llabels: .asciz "labels.bin"
Ltrain_indices: .asciz "train_indices.bin"
Lval_indices: .asciz "val_indices.bin"
Linit: .asciz "init.bin"
Lschedule: .asciz "schedule.bin"
Lbest: .asciz "best.bin"
Llast: .asciz "last.bin"
Lhistory: .asciz "history.csv"
Lperformance: .asciz "performance.json"
Lpred_train: .asciz "predictions_train.bin"
Lpred_val: .asciz "predictions_validation.bin"
Lpred_test: .asciz "predictions_test.bin"
Ltiming1: .asciz "timing_batch1.bin"
Ltiming128: .asciz "timing_batch128.bin"
Leval_limit_name: .asciz "CNN_ASM_EVAL_LIMIT"
Lusage_message: .asciz "Usage: cnn-assembly INPUT_DIRECTORY OUTPUT_DIRECTORY [epochs:1..25] [train_limit:1..45000]"
Lfatal_message: .asciz "cnn-assembly fatal: invalid input, failed I/O/system call, or allocation failure"
Lnumeric_message: .asciz "cnn-assembly fatal: nonfinite numerical result"
Lexisting: .asciz "cnn-assembly refuses a nonempty output directory"
Lcomplete: .asciz "Assembly training, selected-checkpoint evaluation, and timing complete."
Lhistory_header: .asciz "epoch,train_loss,train_accuracy,val_loss,val_accuracy,train_seconds,val_seconds,epoch_seconds,selected_epoch\n"
Lhistory_format: .asciz "%lu,%.12g,%.12g,%.12g,%.12g,%.9f,%.9f,%.9f,%lu\n"
Lprogress: .asciz "epoch %lu/%lu train_loss=%.6f train_accuracy=%.6f val_loss=%.6f val_accuracy=%.6f epoch_seconds=%.3f\n"
Lperformance_format: .asciz "{\n  \"training_seconds\": %.9f,\n  \"best_epoch\": %lu,\n  \"parameter_count\": 5418,\n  \"threads\": 2,\n  \"final_evaluation_seconds\": %.9f,\n  \"prediction_seconds\": {\"train\": %.9f, \"validation\": %.9f, \"test\": %.9f},\n  \"peak_rss_bytes\": %lu,\n  \"process_user_cpu_seconds\": %.9f,\n  \"process_system_cpu_seconds\": %.9f,\n  \"epochs\": %lu,\n  \"train_limit\": %lu,\n  \"evaluation_counts\": {\"train\": %lu, \"validation\": %lu, \"test\": %lu},\n  \"timing_scope\": \"forward only; 20 warmups and 100 samples; batch1 caller and batch128 two persistent workers\",\n  \"floating_point\": \"float32 model/gradients; float64 reporting sums\"\n}\n"
.section __TEXT,__const
.p2align 3
Lbillion: .double 1000000000.0
Lmillion: .double 1000000.0

.comm _rt_input_root,8,3
.comm _rt_output_root,8,3
.comm _rt_images,8,3
.comm _rt_labels,8,3
.comm _rt_train_indices,8,3
.comm _rt_val_indices,8,3
.comm _rt_test_indices,8,3
.comm _rt_schedule,8,3
.comm _rt_params,8,3
.comm _rt_best,8,3
.comm _rt_mean_gradient,8,3
.comm _rt_first_moment,8,3
.comm _rt_second_moment,8,3
.comm _rt_predictions,8,3
.comm _rt_timing_images,8,3
.comm _rt_history,8,3
.comm _rt_epochs,8,3
.comm _rt_train_count,8,3
.comm _rt_eval_train_count,8,3
.comm _rt_eval_val_count,8,3
.comm _rt_eval_test_count,8,3
.comm _rt_step,8,3
.comm _rt_best_epoch,8,3
.comm _rt_best_accuracy,8,3
.comm _rt_best_loss,8,3
.comm _rt_epoch_train_seconds,8,3
.comm _rt_epoch_train_loss,8,3
.comm _rt_epoch_train_accuracy,8,3
.comm _rt_epoch_val_seconds,8,3
.comm _rt_epoch_val_loss,8,3
.comm _rt_epoch_val_accuracy,8,3
.comm _rt_epoch_seconds,8,3
.comm _rt_training_start,8,3
.comm _rt_training_seconds,8,3
.comm _rt_final_evaluation_seconds,8,3
.comm _rt_train_prediction_seconds,8,3
.comm _rt_val_prediction_seconds,8,3
.comm _rt_test_prediction_seconds,8,3
.comm _rt_tick_seconds,8,3
.comm _rt_timebase,8,3
.comm _rt_rusage,256,4
.comm _rt_mutex,64,3
.comm _rt_job_cond,48,3
.comm _rt_done_cond,48,3
.comm _rt_generation,8,3
.comm _rt_done,8,3
.comm _rt_mode,8,3
.comm _rt_job_records,8,3
.comm _rt_job_count,8,3
.comm _rt_job_predictions,8,3
.comm _rt_workers,192,4
.comm _rt_threads,16,3
.comm _rt_timing_values,800,4

.comm _rt_membership,50000,4
.comm _rt_seen,50000,4
