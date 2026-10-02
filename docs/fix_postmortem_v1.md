# Fix 1 post-mortem

Declaration: `docs/fix_declaration.md` (committed before the fix, tag `fix-loop-before`).
After run: `benchmarks/arkitscenes_421383/lidar_benchmark_v1_after_fix1.json`.

## Result: prediction wrong, gate got worse

| | Before | Predicted | After fix 1 |
|---|---|---|---|
| Wall repeatability, median difference | 44 mm | 15 mm or less | **72 mm** |
| Wall pairs passing | 0 of 4 | 2 of 4 | **0 of 4** |
| Wall length median error vs laser | 45 mm | ~25 mm | 42.5 mm |
| Ceiling error (42444966 / 42444968) | -9.9 / -27.7 mm | not predicted | -0.5 / -43.6 mm |
| Ceiling spread across captures | 17.8 mm | not predicted | 43.2 mm |

The prediction was badly wrong. The fix changed the loop-closure graph (42444968: 15 -> 18 closures from 50 candidates) but did not merge the doubled walls, and it moved the second capture's ceiling the wrong way.

## Falsification test from the declaration: fired

The declaration said: "If the walls still show two peaks after the fix, the loops were still not closed." Point profiles after the fix still show both copies about 6 cm apart on 42444968 R1-W4 and 42444966 R1-W7.

## What was actually wrong: the mechanism

Splitting each doubled wall's points by when, from how far and at what angle they were seen:

| Wall | Inner copy (0 cm) | Outer copy (-6 cm) |
|---|---|---|
| 42444968 R1-W4 | seen 8 to 43 s; range 1.63 m; view cos 0.85 | seen 2 to 6 s only; range 2.19 m; view cos 0.96 |
| 42444966 R1-W7 | seen 6 to 93 s; range 1.33 m; view cos 0.89 | seen 9 to 10 s only; range 1.31 m; view cos 0.92 |

The copies separate by TIME. On 42444966 both copies were seen from the same range and angle, which rules out a sensor or angle effect there. The outer copy comes from a few seconds of frames placed about 6 cm off: once at the very start of the capture (tracking still settling) and once in a one-second blip.

So the symptom in the declaration was right (doubled walls drive the repeatability failure) and the mechanism was wrong. It is not slow drift between passes that loop closure between 3-second fragments can fix:
- the glitches are shorter than a fragment, and a fragment is corrected as one rigid piece, so a glitch inside a fragment cannot be removed;
- on 42444968 the glitch is in the first seconds, which belong to fragment 0, the fixed reference of the pose graph.

## What I would do differently

Test the time-split before declaring. The evidence available at declaration time (two peaks; laser shows one) did not distinguish slow drift from short glitches, and I chose the explanation without checking it.
