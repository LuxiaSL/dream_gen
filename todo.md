async generation; need to parallelize the process of buffer building (generate keyframes *and* interp frames at the same time)
send job to comfyui, collect id, hold/future it, poll queue until frame is completed, retrieve it, use it
interpolate on pairs the entire time while this happens so that things are kept at pace and the jobs are done simultaneously
also: buffer is blocking the main loop, hacky workaround implemented but needs to be fixed, can likely be done alongside the parallelization of generation. essentially, large async update. things to watch for include CUDA/torch errors with parallelization, hacky/nonstandard solutions acceptable if initial development proves rough. 

---

# Async/Parallel Generation Architecture - TODO

## 🎯 **CRITICAL**: Current Implementation is Synchronous!

The current hybrid generator **blocks during interpolation** to generate the next keyframe. This defeats the entire purpose of interpolation smoothing!

## 📊 Current Flow (WRONG)

```
Frame 0:  Generate keyframe A (2.1s) ──────────┐
                                              WAIT
Frame 1:  Need to interpolate A→B             ↓
          But B doesn't exist yet!             │
          → Generate B NOW (2.1s) ─────────────┘
          → Then interpolate frame 1 (0.05s)
          
Frame 2:  Interpolate A→B (0.05s) ← fast
Frame 3:  Interpolate A→B (0.05s) ← fast
...
Frame 10: Interpolate A→B (0.05s) ← fast

Frame 11: Need keyframe C
          → Generate C NOW (2.1s) ─────────────┐
                                              WAIT
```

**Problem**: We **pause interpolation** to generate the next keyframe!

**Result**: Stuttering visuals, defeats the purpose of smooth interpolation

## ✅ Intended Flow (CORRECT)

```
Frame 0:  Generate keyframe A (2.1s) ────┐
                                        DONE
                                         ↓
          Queue generation of B in background ─────────┐
                                                      ASYNC
Frame 1:  Interpolate A→? (0.05s) ← smooth            ↓
Frame 2:  Interpolate A→? (0.05s) ← smooth            ↓
Frame 3:  Interpolate A→? (0.05s) ← smooth            ↓
...                                                    ↓
Frame 10: Interpolate A→? (0.05s) ← smooth            ↓
                                                       ↓
Frame 11: Check: Is B ready? ────────────────────────┘
          YES! → Interpolate A→B (0.05s)
          Queue generation of C in background ────────┐
                                                     ASYNC
Frame 12: Interpolate A→B (0.05s) ← smooth            ↓
...
```

**Key insight**: Generation happens **in parallel with interpolation**

**Result**: Smooth continuous output, no stuttering

## 🔧 What Needs to Change

### 1. **Async Generation Queue** (ComfyUI API already supports this!)

ComfyUI has a queue system. We need to:
- Queue prompt without waiting
- Store the `prompt_id` as a "future"
- Continue interpolating
- Poll for completion asynchronously

### 2. **Predictive Keyframe Scheduling**

Need to queue the NEXT keyframe as soon as CURRENT keyframe is ready (img2img)

### 3. **Frame Buffer Management**

Need to handle cases where generation is slower than interpolation, can simply be done by skipping and lowering interval for the interp loop to check explicitly for when the next keyframe is generated. or maybe make it so that after each keyframe is generated we attempt interpolation or add it to the queue.

**The math**: If generation time < interpolation sequence time, we get **perfect pipelining**.

## 📝 Notes

- ComfyUI's queue system already supports this - we just need to use it properly
- The `torch.cuda.synchronize()` fixes won't affect async - they just ensure clean state
- This is a **major** architectural change but will dramatically improve visual smoothness
- Current stuttering is because we block during interpolation to generate next keyframe

rn its all sequential. i generate keyframe which takes 2s, then do interpolation, another 2s. but the keyframe is just waiting for comfyui to finish, its not actually doing work.
i can do the interp at the same time as comfyui generating the diffusion img2img, which is the intent.
it lets it be at proper pace, that way the amount of time it takes to generate these things is the actual amount of frames per second that need to be shown,
plus a bit extra for buffer zone in case it lags for some reason.
like rn you'd have 2s wait for keyframe -> 2s of interp frames at the rate they generate, variable, anywhere from 4fps to 15fps -> 2s wait for next keyframe -> keyframe display -> interp gen and display -> wait for keyframe, when i can get rid of the wait for keyframe and generate/show the interps in between since they're faster.
its just awaiting an http call to the comfyui server.
generate interp frames ASAP between the prior keyframe and the newly generated one whenever it comes back each time.
this also can dovetail with making the caching system async so that everything that wants to run autonomously and smoothly can; unlocks a bit more power in evaluating the whole cache for health. 