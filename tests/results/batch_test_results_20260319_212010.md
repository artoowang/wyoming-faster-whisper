# Batch Test Results - 20260319_212010

## Summary
- **Test command**:
    ```shell
    script/run --model large --model-type whisper-mps --language en --uri "tcp://0.0.0.0:10301" --log-format "%(asctime)s [%(levelname)s] %(name)s: %(message)s" --debug --initial-prompt "The following is a transcription of a user command issued to a home assistant. Common command includes 'What time is it', 'Timer, XX minutes', 'Turn on Morning Scene', 'Turn off Mos Eisley'. Common device names include 'Mos Eisley', 'Morning Scene', 'Evening Scene', 'Night Scene'"
    ```
- **Note**: This is identical to the actual wyoming-stt server I am running on 2026-03-19.
- **Total tests**: 37
- **Server**: localhost:10301
- **Backend**: whisper-mps
- **Pass**: 23 | **Fail**: 14

## Results

| # | File Name | Ground Truth | Transcription | Time (s) | Match | Evaluation |
|---|-----------|--------------|---------------|----------|-------|------------|
| 1 | speech_20251019_211239.wav | Cancel timer | Cancel Timer. | 0.36 | PASS | Minor punctuation difference only |
| 2 | speech_20251020_212236.wav | What time is it | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 3 | speech_20251020_091442.wav | What time is it | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 4 | speech_20251019_233635.wav | Cancel timer | Cancel timer. | 0.33 | PASS | Exact match (ignoring case/punctuation) |
| 5 | speech_20251019_233623.wav | What day is tomorrow | What day is tomorrow? | 0.34 | PASS | Minor punctuation difference only |
| 6 | speech_20251019_233612.wav | What time is it | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 7 | speech_20251019_223339.wav | What is the current time in Taiwan | What is the current time in Taiwan? | 0.37 | PASS | Minor punctuation difference only |
| 8 | speech_20251019_223319.wav | What is the current time in Taiwan | What is the current time in Taiwan? | 0.37 | PASS | Minor punctuation difference only |
| 9 | speech_20251019_212157.wav | What time is it | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 10 | speech_20251019_211301.wav | Cancel the previous timer set | Cancel the previous timer set. | 0.36 | PASS | Minor punctuation difference only |
| 11 | speech_20251019_211219.wav | Please cancel timer | Please cancel timer. | 0.33 | PASS | Minor punctuation difference only |
| 12 | speech_20251019_211016.wav | Timer, one minute | Timer, one minute. | 0.35 | PASS | Minor punctuation difference only |
| 13 | speech_20251019_192524.wav | Cancel all timers | Cancel all timers. | 0.35 | PASS | Minor punctuation difference only |
| 14 | speech_20251019_192509.wav | Cancel all timers | Cancel all time use. | 0.35 | FAIL | "timers" → "time use" - semantic error |
| 15 | speech_20251019_191140.wav | Timer, one minute | Timer, one minute. | 0.35 | PASS | Minor punctuation difference only |
| 16 | speech_20251019_183218.wav | Set timer, one minute | Set timer 1 minute. | 0.34 | PASS | "one" → "1" - acceptable numeric conversion |
| 17 | speech_20251020_091445.wav | What time is it? It's 9:14am | What time is it? It's 9.40am. | 0.44 | FAIL | Time misheard: "9:14" → "9.40" |
| 18 | speech_20251019_192430.wav | Timer, one minute | Timer, one minute. | 0.35 | PASS | Minor punctuation difference only |
| 19 | debug_20251024_194442.wav | What time is it? | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 20 | debug_20251025_115718.wav | What time is it? | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 21 | debug_20251025_201119.wav | What time is it? | What time is it? | 0.34 | PASS | Minor punctuation difference only |
| 22 | debug_20251109_185336.wav | Turn off Mos Eisley | Turn off Mos Eisley. | 0.35 | PASS | Minor punctuation difference only |
| 23 | debug_20251028_225310.wav | Turn on night scene <noise> | Turn on Night Scene. | 0.34 | PASS | Handles noisy audio well |
| 24 | debug_20251027_073026.wav | Turn on morning scene <low volume> | Turn on Morning Scene. | 0.34 | PASS | Handles low volume audio well |
| 25 | debug_20251112_225552.wav | Turn on night scene <low volume> | and so on. | 0.33 | FAIL | Hallucination: "and so on" is wrong |
| 26 | debug_20251024_195256.wav | Timer, 15 minutes | | 2.23 | FAIL | Empty transcription - model timeout |
| 27 | debug_20251024_181834.wav | How about now? | Oh no. | 0.32 | FAIL | Completely wrong - misheard/unclear audio |
| 28 | debug_20251026_143348.wav | <empty with initial chime> | Common device names include Common device names include | 0.40 | FAIL | Hallucination on empty audio |
| 29 | debug_20251105_074155.wav | Turn on morning scene | Tomorrow morning scene. | 0.33 | FAIL | Added "Tomorrow" - semantic error |
| 30 | debug_20251112_225605.wav | Turn on night scene | Turn off Night Scene | 2.28 | FAIL | "on" → "off" - opposite meaning |
| 31 | debug_20251115_213955.wav | Turn off Mos Eisley | and so on. | 0.33 | FAIL | Hallucination: "and so on" is wrong |
| 32 | debug_denoised_20251115_213955.wav | Turn off Mos Eisley | and so on. | 0.33 | FAIL | Hallucination: "and so on" is wrong |
| 33 | debug_20251123_203425.wav | Cancel timer | and so on. | 0.33 | FAIL | Hallucination: "and so on" is wrong |
| 34 | debug_20251123_205211.wav | Cancel timer | and also, Cancel Timer. | 0.37 | FAIL | Hallucination prefix |
| 35 | debug_20251026_143348_trimmed.wav | <empty> | Common device names include the following. | 0.36 | FAIL | Hallucination on empty audio |
| 36 | debug_20251108_075833.wav | Turn it on | and so on. | 0.34 | FAIL | Hallucination: "and so on" is wrong |
| 37 | debug_20251128_134301.wav | Cancel timer | Cancel that one. | 0.35 | FAIL | "timer" → "that one" - semantic error |
