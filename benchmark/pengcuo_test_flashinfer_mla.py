import torch
import flashinfer
import sys
num_local_heads = 128 // 8 # TP = 8
batch_size = 32
head_dim_ckv = 512
head_dim_kpe = 64
page_size = 1
mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    backend="fa3"
)

import triton


for seqlen in [(int)(s * 1024) for s in [0.5, 1, 2, 4, 8, 16, 32, 64]]:
    context_len = seqlen
    q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    kv_lens = torch.full((batch_size,), context_len, dtype=torch.int32).to(0)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * context_len
    kv_indices = torch.arange(0, batch_size * context_len).to(0).int()
    q_nope = torch.randn(
        batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    q_pe = torch.zeros(
        batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    ckv = torch.randn(
        batch_size * context_len, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    )
    kpe = torch.zeros(
        batch_size * context_len, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    causal = True
    mla_wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_local_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )

    # print(f"q_inptr : {q_indptr.shape} {q_indptr.dtype}")
    # print(f"kv_indptr : {kv_indptr.shape} {kv_indptr.dtype}")
    # print(f"kv_indices : {kv_indices.shape} {kv_indices.dtype}")
    # print(f"kv_lens : {kv_lens}")
    # print(f"num_local_heads : {num_local_heads}")
    # print(f"head_dim_ckv : {head_dim_ckv}")
    # print(f"head_dim_kpe : {head_dim_kpe}")
    # print(f"page_size : {page_size}")
    o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    # print(o.shape)

    warmup_times = 100
    for i in range(warmup_times):
        o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    torch.cuda.synchronize()



    fn = lambda: mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    t = triton.testing.do_bench(fn) 

    
    b = batch_size
    s_q = 1
    h_kv = 1
    d = head_dim_ckv + head_dim_kpe
    dv = head_dim_ckv
    h_q = num_local_heads
    FLOPS = b * context_len * h_q * (d + dv) * 2
    bytes = (b * context_len * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
        torch.finfo(q_nope.dtype).bits // 8
    )
    mean_sk = context_len
    
    
    print(
        f"{b=}, {s_q=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=},  {t:.6f} , {FLOPS / 10 ** 9 / t:.0f} , {FLOPS}, {bytes / 10 ** 6 / t:.0f},  {bytes}, {mean_sk=}, "
    )
