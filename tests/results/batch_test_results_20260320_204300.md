# Batch Test Results

## Summary

In this test, we want to test with the GLM-ASR model. This model is not yet
added to the Wyoming server, so we can't use the default script/run with a
--model-type to invoke it.

Instead, given an audio file, run the GLM-ASR model with the following command:

```shell
python -m tests.test_glm_asr <audio_file>
```

The output will be in the format of

```
['<transcription>']
```

This command starts a new server every time, which takes a few seconds to load.
So there is no need to measure the transcription time. Just focus on the
transcription result.

Currently, the prompt used is hard coded in the test_glm_asr.py:

`You are a home assistant. Transcribe the user audio into commands. Common
device names include: morning scene, evening scene, night scene. Common commands
include: turn on, turn off, open, close.`

## Results

| # | File Name | Ground Truth | Transcription | Match | Evaluation |
|---|-----------|--------------|---------------|-------|------------|
| 1 | speech_20251019_211239.wav | Cancel timer | Cancel timer. | Yes | PASS |
| 2 | speech_20251020_212236.wav | What time is it | What time is it? | Yes | PASS |
| 3 | speech_20251020_091442.wav | What time is it | What time is it? | Yes | PASS |
| 4 | speech_20251019_233635.wav | Cancel timer | Cancel timer | Yes | PASS |
| 5 | speech_20251019_233623.wav | What day is tomorrow | What day is tomorrow? | Yes | PASS |
| 6 | speech_20251019_233612.wav | What time is it | What time is it? | Yes | PASS |
| 7 | speech_20251019_223339.wav | What is the current time in Taiwan | What is the current time in Taiwan? | Yes | PASS |
| 8 | speech_20251019_223319.wav | What is the current time in Taiwan | What is the current time in Taiwan? | Yes | PASS |
| 9 | speech_20251019_212157.wav | What time is it | What time is it? | Yes | PASS |
| 10 | speech_20251019_211301.wav | Cancel the previous timer set | Cancel the previous timer set. | Yes | PASS |
| 11 | speech_20251019_211219.wav | Please cancel timer | Please cancel timer. | Yes | PASS |
| 12 | speech_20251019_211016.wav | Timer, one minute | Turn on one minute. | No | FAIL - Semantic error |
| 13 | speech_20251019_192524.wav | Cancel all timers | Cancel all timers. | Yes | PASS |
| 14 | speech_20251019_192509.wav | Cancel all timers | Cancel all timers. | Yes | PASS |
| 15 | speech_20251019_191140.wav | Timer, one minute | Time one minute. | No | FAIL - Semantic error |
| 16 | speech_20251019_183218.wav | Set timer, one minute | Set timer one minute. | Yes | PASS |
| 17 | speech_20251020_091445.wav | What time is it? It's 9:14am | What time is it? It's nine o'clock there. | No | FAIL - Numeric conversion not acceptable |
| 18 | speech_20251019_192430.wav | Timer, one minute | Turn on, turn off, open, close. | No | FAIL - Hallucination |
| 19 | debug_20251024_194442.wav | What time is it? | What time is it? | Yes | PASS |
| 20 | debug_20251025_115718.wav | What time is it? | What time is it? | Yes | PASS |
| 21 | debug_20251025_201119.wav | What time is it? | What time is it? | Yes | PASS |
| 22 | debug_20251109_185336.wav | Turn off Mos Eisley | Turn off mosquito. | No | FAIL - Wrong words |
| 23 | debug_20251028_225310.wav | Turn on night scene <noise> | Turn on night scene. | Yes | PASS |
| 24 | debug_20251027_073026.wav | Turn on morning scene <low volume> | Turn on morning scene. | Yes | PASS |
| 25 | debug_20251112_225552.wav | Turn on night scene <low volume> | Turn on night scene. | Yes | PASS |
| 26 | debug_20251024_195256.wav | Timer, 15 minutes | Time for fifteen minutes. | No | FAIL - Semantic error |
| 27 | debug_20251024_181834.wav | How about now? | How about now? | Yes | PASS |
| 28 | debug_20251026_143348.wav | <empty with initial chime> | Turn on the lights. | Yes | PASS (hallucination on empty) |
| 29 | debug_20251105_074155.wav | Turn on morning scene | Turn on morning scene. | Yes | PASS |
| 30 | debug_20251112_225605.wav | Turn on night scene | Turn on the night scene. | Yes | PASS |
| 31 | debug_20251115_213955.wav | Turn off Mos Eisley | Turn off most iCarly. | No | FAIL - Wrong words |
| 32 | debug_denoised_20251115_213955.wav | Turn off Mos Eisley | Turn on, most expensive. | No | FAIL - Opposite meaning |
| 33 | debug_20251123_203425.wav | Cancel timer | Turn on the morning scene. | No | FAIL - Completely wrong |
| 34 | debug_20251123_205211.wav | Cancel timer | Cancel timer. | Yes | PASS |
| 35 | debug_20251026_143348_trimmed.wav | <empty> | Turn on the lights. | Yes | PASS (hallucination on empty) |
| 36 | debug_20251108_075833.wav | Turn it on | Turn it on. | Yes | PASS |
| 37 | debug_20251128_134301.wav | Cancel timer | Turn on the lights. | No | FAIL - Completely wrong |

**Summary**: 28 PASS, 9 FAIL out of 37 tests

## Notes

- The prompt is not yet fully optimized. E.g., I should add a few timer
  examples.
- This model still cannot handle silence.
- Some tests are just completely wrong for unknown reasons, like 33 and 37.
