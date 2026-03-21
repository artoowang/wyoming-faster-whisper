# Batch Test Results

## Summary

In this test, we want to test with the GLM-ASR model. Use the following command
to run the server:

```shell
script/run --model-type glm-asr --model "zai-org/GLM-ASR-Nano-2512" --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt <prompt>
```

Use the following prompt:

`The following is a transcription of a user command issued to a home assistant. Common command includes 'What time is it', 'Timer, XX minutes', 'Turn on Morning Scene', 'Turn off Mos Eisley'. Common device names include 'Mos Eisley', 'Morning Scene', 'Evening Scene', 'Night Scene'`

And use `batch_test_ground_truths.py` to run the test.

## Results

| # | File Name | Ground Truth | Transcription | Processing Time | Evaluation | Notes |
|---|-----------|--------------|---------------|-------|------------|-------|
| 1 | speech_20251019_211239.wav | Cancel timer | Cancel timer | 0.64s | PASS | |
| 2 | speech_20251020_212236.wav | What time is it | What time is it? | 0.40s | PASS | |
| 3 | speech_20251020_091442.wav | What time is it | What time is it? | 0.41s | PASS | |
| 4 | speech_20251019_233635.wav | Cancel timer | Cancel timer | 0.36s | PASS | |
| 5 | speech_20251019_233623.wav | What day is tomorrow | What day is tomorrow? | 0.41s | PASS | |
| 6 | speech_20251019_233612.wav | What time is it | What time is it? | 0.39s | PASS | |
| 7 | speech_20251019_223339.wav | What is the current time in Taiwan | What is the current time in Taiwan? | 0.47s | PASS | |
| 8 | speech_20251019_223319.wav | What is the current time in Taiwan | What is the current time in Taiwan? | 0.44s | PASS | |
| 9 | speech_20251019_212157.wav | What time is it | What time is it? | 0.40s | PASS | |
| 10 | speech_20251019_211301.wav | Cancel the previous timer set | Cancel the previous timer set. | 0.42s | PASS | |
| 11 | speech_20251019_211219.wav | Please cancel timer | Please cancel timer. | 0.39s | PASS | |
| 12 | speech_20251019_211016.wav | Timer, one minute | Timer, one minute. | 0.40s | PASS | |
| 13 | speech_20251019_192524.wav | Cancel all timers | Cancel all timers. | 0.39s | PASS | |
| 14 | speech_20251019_192509.wav | Cancel all timers | Cancel all timers. | 0.36s | PASS | |
| 15 | speech_20251019_191140.wav | Timer, one minute | Timer, one minute. | 0.55s | PASS | |
| 16 | speech_20251019_183218.wav | Set timer, one minute | Set timer one minute. | 0.40s | PASS | |
| 17 | speech_20251020_091445.wav | What time is it? It's 9:14am | What time is it? It's nine forty there. | 0.52s | FAIL: numeric conversion | |
| 18 | speech_20251019_192430.wav | Timer, one minute | Mos Eisley | 0.38s | FAIL: completely wrong | With the initial chime |
| 19 | debug_20251024_194442.wav | What time is it? | What time is it? | 0.37s | PASS | |
| 20 | debug_20251025_115718.wav | What time is it? | What time is it? | 0.37s | PASS | |
| 21 | debug_20251025_201119.wav | What time is it? | What time is it? | 0.39s | PASS | |
| 22 | debug_20251109_185336.wav | Turn off Mos Eisley | Turn off Morning Scene. | 0.38s | FAIL: wrong device name | With the initial chime |
| 23 | debug_20251028_225310.wav | Turn on night scene | Turn on Night Scene. | 0.43s | PASS | noise |
| 24 | debug_20251027_073026.wav | Turn on morning scene | Turn on Morning Scene | 0.40s | PASS | low volume |
| 25 | debug_20251112_225552.wav | Turn on night scene | Turn on Night Scene. | 0.41s | PASS | low volume |
| 26 | debug_20251024_195256.wav | Timer, 15 minutes | Time for fifteen minutes. | 0.39s | FAIL: numeric conversion, semantic | |
| 27 | debug_20251024_181834.wav | How about now? | How about now? | 0.38s | PASS | |
| 28 | debug_20251026_143348.wav | (empty) | Morning Scene... (hallucination) | 2.52s | FAIL: hallucination on empty audio | empty with initial chime |
| 29 | debug_20251105_074155.wav | Turn on morning scene | Turn on Morning Scene. | 0.42s | PASS | |
| 30 | debug_20251112_225605.wav | Turn on night scene | Turn on Night Scene | 0.36s | PASS | |
| 31 | debug_20251115_213955.wav | Turn off Mos Eisley | Turn off Morning Scene. | 0.40s | FAIL: wrong device name | |
| 32 | debug_denoised_20251115_213955.wav | Turn off Mos Eisley | The following is a transcription... | 1.34s | FAIL: hallucination | Heavily denoised |
| 33 | debug_20251123_203425.wav | Cancel timer | Cancel timer. | 0.37s | PASS | |
| 34 | debug_20251123_205211.wav | Cancel timer | Cancel timer | 0.32s | PASS | |
| 35 | debug_20251026_143348_trimmed.wav | (empty) | Morning Scene... (hallucination) | 2.30s | FAIL: hallucination on empty audio | empty |
| 36 | debug_20251108_075833.wav | Turn it on | Turn it off. | 0.37s | FAIL: opposite meaning | |
| 37 | debug_20251128_134301.wav | Cancel timer | Cancel task. | 0.37s | FAIL: semantic error | With water noise |

## Summary

- **Total**: 37 tests
- **Pass**: 28 (75.7%)
- **Fail**: 9 (24.3%)

### Failure Categories

| Category | Count | Files |
|----------|-------|-------|
| Numeric conversion | 2 | speech_20251020_091445.wav, debug_20251024_195256.wav |
| Hallucination on empty audio | 2 | debug_20251026_143348.wav, debug_20251026_143348_trimmed.wav |
| Wrong device name | 2 | debug_20251109_185336.wav, debug_20251115_213955.wav |
| Opposite meaning | 1 | debug_20251108_075833.wav |
| Semantic error | 1 | debug_20251128_134301.wav |
| Hallucination/wrong | 1 | debug_denoised_20251115_213955.wav |
| Completely wrong | 1 | speech_20251019_192430.wav |

debug_20251115_213955.wav is especially concerning, since the audio isn't that bad, but the result leads to a totally wrong device.