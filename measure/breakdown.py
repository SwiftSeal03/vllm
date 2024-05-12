import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.usage.usage_lib import UsageContext
from vllm.sequence import ExecuteModelRequest

engine = None

def default_sp() -> SamplingParams:
  return SamplingParams(
    temperature=0.8, 
    max_tokens=1024,
    ignore_eos=True
  )

def add_req_toks(
  req_id: str,
  tokens: list[int], 
  sp: SamplingParams = default_sp()
) -> None:
  engine.add_request(
    request_id=req_id,
    prompt=None,
    prompt_token_ids=tokens,
    sampling_params=sp
  )

def bench_normal():
  print("Starting normal benchmark")
  ntoks = [x * 128 for x in range(1, 32)]
  bszs = [1, 2, 4, 8]
  
  keys = []
  for bsz in bszs:
    for ntok in ntoks:
      keys.append((bsz, ntok))
  bszs, ntoks = zip(*keys)
  
  pref_times = []
  dec_times = []
  for bsz, ntok in tqdm(keys):
    tokens = list(np.random.randint(0, 1000, ntok))
    tokenss = [tokens[i: i + ntok // bsz] for i in range(bsz)]
    req_ids = [str(i) for i in range(bsz)]
    
    pref_time = []
    dec_time = []
    for k in range(0, 3):
      for j in range(bsz):
        add_req_toks(req_ids[j], tokenss[j])
        
      start = time.perf_counter()
      
      engine.step()
      
      middle = time.perf_counter()
      
      for _ in range(10):
        engine.step()
        
      end = time.perf_counter()
      
      engine.abort_request(req_ids)
      if k > 0:
        pref_time.append(middle - start)
        dec_time.append((end - middle)/10)
        
    pref_times.append(np.mean(np.array(pref_time)))
    dec_times.append(np.mean(np.array(dec_time)))
    
  pframe = pd.DataFrame({"bsz": bszs, "ntok": ntoks, "time": pref_times})
  pframe.to_csv("prefill.csv", index=False)
  
  dframe = pd.DataFrame({"bsz": bszs, "ntok": ntoks, "time": dec_times})
  dframe.to_csv("decode.csv", index=False)
  
def bench_chunk():
  print("Starting chunk benchmark")
  engine.scheduler.scheduler_config.chunked_prefill_enabled = True
  old_max_num_batched_tokens = engine.scheduler.scheduler_config.max_num_batched_tokens
  strides = [x * 128 for x in range(1, 9)]
  ntok = 4096
  tokens = list(np.random.randint(0, 1000, ntok))
  
  prefs = []
  chnks = []
  times = []
  for stride in tqdm(strides):
    engine.scheduler.scheduler_config.max_num_batched_tokens = stride
    niter = ntok // stride
    test_times = [[] for _ in range(niter)]
    for k in range(3):
      add_req_toks("0", tokens)
      for j in range(niter):
        start = time.perf_counter()
        
        engine.step()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        if k > 0:
          test_times[j].append(end - start)
      engine.abort_request("0")
    
    for j in range(niter):
      prefs.append(stride * j)
      chnks.append(stride)
      times.append(np.mean(test_times[j]))
      
  frame = pd.DataFrame({"prefix": prefs, "chunk": chnks, "time": times})
  frame.to_csv("chunk.csv", index=False)
  
  engine.scheduler.scheduler_config.chunked_prefill_enabled = False
  engine.scheduler.scheduler_config.max_num_batched_tokens = old_max_num_batched_tokens
  
def bench_swap():
  print("Starting swap benchmark")
  ntoks = [x * 128 for x in range(1, 32)]
  
  # Initialize a request and prefill
  tokens = list(np.random.randint(0, 1000, ntoks[-1]))
  add_req_toks("0", tokens)
  engine.step()
  
  # Get physical block ids of the request
  seq_id = engine.scheduler.running[0].get_seqs()[0].seq_id
  gpu_blk_ids = [blk.block_number for blk in engine.scheduler.block_manager.block_tables[seq_id]]
  cpu_blk_ids = list(range(len(gpu_blk_ids)))
  
  swap_in_mapping = list(zip(cpu_blk_ids, gpu_blk_ids))
  swap_out_mapping = list(zip(gpu_blk_ids, cpu_blk_ids))
  
  si_times = []
  so_times = []
  for ntok in tqdm(ntoks):
    # We should bypass scheduler/block_manager and directly call the executor
    nblk = ntok // engine.scheduler.cache_config.block_size

    swap_out_req = ExecuteModelRequest(
      seq_group_metadata_list=[],
      blocks_to_swap_out=swap_out_mapping[:nblk],
    )
    start = time.perf_counter()

    engine.model_executor.execute_model(swap_out_req)

    torch.cuda.synchronize()
    end = time.perf_counter()

    so_times.append(end - start)

    swap_in_req = ExecuteModelRequest(
      seq_group_metadata_list=[],
      blocks_to_swap_in=swap_in_mapping[:nblk],
    )
    start = time.perf_counter()

    engine.model_executor.execute_model(swap_in_req)

    torch.cuda.synchronize()
    end = time.perf_counter()

    si_times.append(end - start)
    
  engine.abort_request("0")
  
  frame = pd.DataFrame({"ntok": ntoks, "swap_in": si_times, "swap_out": so_times})
  frame.to_csv("swap.csv", index=False)

if __name__ == "__main__":
  engine_args = EngineArgs(
    model="/data/jxl/Llama-2-7b-hf",
    download_dir="/data/jxl/Llama-2-7b-hf",
    swap_space=4,
    num_gpu_blocks_override=512,

    enforce_eager=True,
    disable_log_stats=True
  )

  engine = LLMEngine.from_engine_args(
    engine_args, usage_context=UsageContext.LLM_CLASS)
  
  bench_normal()
  bench_chunk()
  bench_swap()

  