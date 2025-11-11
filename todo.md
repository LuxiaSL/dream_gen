async generation; need to parallelize the process of buffer building (generate keyframes *and* interp frames at the same time)
send job to comfyui, collect id, hold/future it, poll queue until frame is completed, retrieve it, use it
interpolate on pairs the entire time while this happens so that things are kept at pace and the jobs are done simultaneously
also: buffer is blocking the main loop:

```
025-11-11 04:25:01 - core.generation_coordinator - INFO - Generating INTERPOLATIONS 1 -> 2 (10 frames)
2025-11-11 04:25:04 - core.generation_coordinator - INFO - [OK] Generated 10/10 interpolations in 2.51s
2025-11-11 04:25:04 - core.generation_coordinator - INFO -      Average: 0.251s per frame
2025-11-11 04:25:04 - core.generation_coordinator - INFO - ======================================================================
2025-11-11 04:25:04 - core.generation_coordinator - INFO - Generating KEYFRAME 3
2025-11-11 04:25:04 - core.generation_coordinator - INFO - Prompt: black, white, ethereal digital angel, dissolving into partic...
2025-11-11 04:25:04 - core.generator - INFO - Generating from image: keyframe_002.png (denoise=0.3)
2025-11-11 04:25:04 - core.comfyui_api - INFO - Image uploaded: keyframe_002 (2).png (type: input)
2025-11-11 04:25:04 - core.comfyui_api - INFO - Queued prompt: e2fdf0ff-f0ff-447a-ac5e-422bf8dc1558
2025-11-11 04:25:04 - core.generator - INFO - Workflow queued with prompt_id: e2fdf0ff-f0ff-447a-ac5e-422bf8dc1558
2025-11-11 04:25:09 - core.generator - INFO - Still waiting... (5.0s) - Queue: 1 running, 0 pending
2025-11-11 04:25:10 - core.generator - INFO - Prompt no longer in queue after 5.5s (12 polls)
2025-11-11 04:25:10 - core.generator - INFO - Generation time: 5.64s
2025-11-11 04:25:10 - core.generator - INFO - [OK] img2img generation complete: frame_00002.png
2025-11-11 04:25:10 - cache.manager - INFO - Added to cache: cache_00023_1762863910 (total: 24)
2025-11-11 04:25:10 - core.generation_coordinator - INFO - [OK] Keyframe 3 generated in 5.67s
2025-11-11 04:25:10 - core.generation_coordinator - INFO -      Saved to: keyframe_003.png
2025-11-11 04:25:10 - core.generation_coordinator - INFO - ======================================================================
2025-11-11 04:25:10 - core.generation_coordinator - INFO - Generating INTERPOLATIONS 2 -> 3 (10 frames)
2025-11-11 04:25:12 - core.generation_coordinator - INFO - [OK] Generated 10/10 interpolations in 2.59s
2025-11-11 04:25:12 - core.generation_coordinator - INFO -      Average: 0.259s per frame
2025-11-11 04:25:12 - core.display_selector - INFO - Display selector starting...
2025-11-11 04:25:12 - core.display_selector - INFO - Waiting for initial buffer to fill...
2025-11-11 04:25:12 - core.display_selector - INFO - Target: 10.0s
2025-11-11 04:25:15 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:20 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:25:20 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:25 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:30 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:30 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:25:35 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:40 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:25:40 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:45 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:50 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:25:50 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:25:55 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:26:00 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:26:00 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:26:05 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:26:10 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:26:10 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:26:15 - core.display_selector - INFO - Buffering... 6.0s / 10.0s (20.0%)
2025-11-11 04:26:20 - core.dream_controller - INFO - Buffer: 6.0s / 30.0s (20.0%) | KF: 2 | INT: 20 | Displayed: 0
2025-11-11 04:26:21 - core.dream_controller - INFO -
```

needs to be fixed, can likely be done alongside the parallelization of generation. essentially, large async update. things to watch for include CUDA/torch errors with parallelization, hacky/nonstandard solutions acceptable if initial development proves rough. 