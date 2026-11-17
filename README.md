<h1 align="center">Miner Llama, MTP-pipe edition</h1>
<p align="center"><b>Miner Llama v4.0 plus eight changes that let an MTP model's own draft head drive the pipeline</b></p>

This is a fork of [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama), itself a fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) for NVIDIA Pascal cards. Miner Llama pipelines decoding across
cards: it submits the next token's evaluation on a guess before the current one finishes. This fork makes the guess
come from the model's multi-token-prediction (MTP) head, keeps the pipeline running through tool calls and JSON
answers, and fixes what got in the way on a Windows rig.

Measured on one rig: four GTX 1070 cards (8 GB each, Pascal sm_61) on PCIe x1 risers, a 2-core Celeron G3900 with
8 GB of RAM, Windows 10. The model is Qwen3.6-35B-A3B with its MTP layer, Unsloth's UD-Q5_K_S GGUF, at 131,072 tokens
of context with a q8_0 KV cache.

| Same rig, same day, two runs where shown | Stock llama.cpp v1.0.0 | Miner Llama v4.0, tuned | **This fork** |
|---|---:|---:|---:|
| Tokens written per second (1,500 tokens, plain text) | 14.8, 14.0 | 53.6, 54.8 | **78.9, 82.2** |
| A tool call at ~39k tokens of context, fresh: seconds | 163.2 | 153.0 | **79.3** |
| reading the prompt, tok/s | 300 | 313 | **594** |
| writing the call, tok/s | 21.7 | 25.3 | **53.8** |
| the same call again, prompt cached: seconds | 41.4 | 27.6 | **12.8** |
| A tool call at ~72k (33k new tokens on the cached 39k): seconds | 234.1 | 240.8 | **125.6** |
| reading the new tokens, tok/s | 158 | 162 | **305** |
| writing the call, tok/s | 27.0 | 18.9 | **40.0** |
| the same call again, prompt cached: seconds | 30.4 | 36.8 | **16.6** |
| A short JSON-schema answer (49 tokens), warm: seconds | 1.76, 1.59 | 1.58, 1.39 | **1.09, 0.85** |

Where the time goes on a fresh long call: of the 73.7 s this fork saves at ~39k against tuned v4.0, about 59 s is
reading the prompt (125 s against 66 s) and about 15 s is writing the call. The prompt speed is the card-to-card
mailbox carrying prompt-sized hops. v4.0 has a mailbox too, but its polling times out on long prompts and resets the
driver: on Miner Llama v3.0, which has the same mailbox, an uncapped run read a 41k prompt at 543 tok/s and died at
about 59k tokens of a 104k one. So v4.0 runs here with prompt hops capped out of its mailbox. Change 2 (the gate) is
what keeps the speed safe at any length: a fresh 104k prompt passes at 323 tok/s. Writing is faster because the pipeline keeps running under the tool-call grammar (change 8) and guesses with the MTP head
(change 6). On short answers and plain text, which have little prompt to read, most of the gain is in writing.

How it was measured:

- One script ran all three servers back to back on 2026-11-17 (12:22–12:51 Pacific), with the same model, context,
  KV cache types and `-ub 256`. Each server ran at High process priority, and no display-driver event occurred
  while any of them was measured.
- Everything ran at temperature 0.
  - The plain-text test asks for 1,500 tokens from a 9-token prompt.
  - Each tool-call test sends a synthetic ledger as the prompt with two tool schemas on offer, and the model
    answers by calling one.
- **Stock llama.cpp v1.0.0** (b12359, the latest stable release) is ggml-org's official Windows CUDA 12.4 build.
  - It ran with MTP drafting (`--spec-type draft-mtp`) and the flags this rig's production server used on stock.
  - In the plain-text test three of its four cards spent most of the time in their P5/P8 clock states (averages of
    390–1,015 MHz). The other two servers kept all four cards in P2 at about 1.9 GHz.
  - Its drafting matched the earlier stock build b12110 exactly (997 of 1,505 drafts accepted, the same text). That
    build wrote 34.1 tok/s in the same test the same morning; its clocks were not recorded. The low clocks are the
    likely cause of the gap, not the drafting.
- **Miner Llama v4.0** was built from the release tag for Windows and sm_61.
  - It was tuned on this rig before the comparison: the pipeline at depth 2 with the `cont` guesser, v4.0's own
    card-to-card mailbox (`GGML_CUDA_HOP_MAILBOX=1`, capped at 64 KiB hops because v4.0 has no gate for larger
    ones; uncapped, its polling mailbox resets the driver on long prompts, as above), `GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_KV_SEQ_RM_BOUND=1`, and
    `--ctx-checkpoints-vram 2 --ctx-checkpoints-incall`.
  - Of the settings tried, this was best. v4.0's mailbox uncapped was not tried, for the reason above. The others were MTP drafting with and without v4.0's verify-kernel
    switches, pipeline depths 2–4, `-b 4096`, and the one-token Q5_K kernel (`GGML_CUDA_MMVQ_PASCAL_Q5K_1R`).
  - v4.0 pipelines only requests without a grammar. Its tool calls and JSON answers therefore run one token a step,
    without an MTP draft either. Its tool-call writing is above stock's at ~39k (25.3 against 21.7 tok/s) and below
    it at ~72k (18.9 against 27.0).
- Plain-text output was byte-identical between this fork and v4.0. Tool-call arguments were identical between the
  two at both lengths. Stock's kernels are not Miner Llama's, and its text differed (equally valid). All three gave
  the same JSON answer.

## What changed

Eight commits on top of `miner-llama-v4.0`, each building on its own:

1. **Speculation replay at temperature 0.** A replay after a checkpoint restore is accepted as it is, as at
   temperature > 0. Verifying it again could livelock under a grammar.
2. **`GGML_CUDA_HOP_MAILBOX_GATE_KB`.** Card-to-card hops above N KiB are ordered by CUDA events instead of polling.
   With the polling mailbox, long prompts timed out and reset the Windows driver. With the gate, prompts keep the
   mailbox's speed at any length: the ~39k prompt above reads at 594 tok/s.
3. **`--spec-ckpt-vram`.** The per-step speculation checkpoint of a hybrid model's recurrent state stays in device
   memory. Without it, every step copies 63 MiB to the host over a x1 riser.
4. **The MTP head honours `LLAMA_DRAFT_SHORTLIST`** for `qwen35moe`. Each draft row scores the shortlist instead of
   all 248k tokens.
5. **`--mtp-device N`.** The MTP layer and a copy of the output head go on one card. Under WDDM, MTP rows on the
   last card queued behind the pipeline's work.
6. **`--pipe-guess mtp`.** The pipeline's guesses come from the MTP head, with a row launched ahead. It gets 1,092 of
   1,904 guesses right in the plain-text test, against 458 for `cont`.
7. **`--pipe-mtp-walk N`.** A right guess keeps the MTP chain going instead of re-anchoring every token.
8. **`LLAMA_PIPE_GRAMMAR=1`.** The pipeline keeps running under a grammar. The host checks each drawn token, and a
   rejection leaves the pipe the way a wrong guess does.

Each commit message has its measurements, taken on the Miner Llama v3.0 base where the change was written. On v4.0
the full set matched v3.0 within 1 %, with the same outputs.

## Running it

The configuration measured above:

```powershell
$env:LLAMA_DRAFT_SHORTLIST = '65536'
$env:GGML_CUDA_HOP_MAILBOX = '1'; $env:GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS = '1000'
$env:GGML_CUDA_HOP_MAILBOX_MAX_KB = '0'; $env:GGML_CUDA_HOP_MAILBOX_GATE_KB = '64'
$env:LLAMA_PIPE_GRAMMAR = '1'; $env:LLAMA_ARG_CTX_CHECKPOINTS_VRAM = '2'
$env:GGML_CUDA_GRAPHS_FORCE = '1'; $env:LLAMA_SERVER_YIELD = '2'
$env:LLAMA_SAMPLING_TOP_K_ROWS = '1'; $env:LLAMA_KV_SEQ_RM_BOUND = '1'
llama-server -m Qwen3.6-35B-A3B-MTP-UD-Q5_K_S.gguf -ngl 99 -c 131072 -np 1 --load-mode none -fa on `
    -ctk q8_0 -ctv q8_0 -ts 0.85/1/1.05/1.1 -ub 256 `
    --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --cache-ram 3072 --jinja --reasoning off `
    -bs --pipe-guess mtp --spec-draft-backend-sampling --pipe-depth 2 `
    --mtp-device 0 --pipe-mtp-walk 1000 --spec-ckpt-vram --ctx-checkpoints-incall
```

Notes on these settings:

- **`-ts 0.85/1/1.05/1.1`** offsets the MTP head's copy on card 0.
- **Depth 2 and walk 1000** suit an evaluation of about 22 ms, which is what the mailbox hop gives on this rig. At
  about 32 ms (plain cross-card copies), depth 1 with re-anchoring (`--pipe-mtp-walk 0`) was better.
- **Memory:** a fresh 104k-token prompt passes with this setup, leaving about 580 MiB free on card 0.
- **Driver resets:** the polling mailbox needs the gate (`GATE_KB`) for long prompts on Windows. Without it they reset
  the display driver.

## Upstream

Everything not listed above is Miner Llama v4.0's own (see
[its repository](https://github.com/Cinderella-Man/miner-llama) and the
[docs repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs)), and llama.cpp's README is kept as
[README-llama.cpp.md](README-llama.cpp.md). MIT-licensed like both ([LICENSE](LICENSE)). Not affiliated with
ggml-org or with Miner Llama's author.
