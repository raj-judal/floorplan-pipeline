# Device matrix
| Tier | Runs on | Capture app | Accuracy we can honestly state | Evidence |
|---|---|---|---|---|
| LiDAR | iPhone/iPad with LiDAR (Pro models) | Stray Scanner (free) | Ceiling height -9.9 / -27.7 mm vs laser; wall lengths median 45 mm short (worst 96 mm); 90% intervals calibrated to 9 of 10 laser truths (in-sample); repeatability fails (median 44 mm between captures) | ARKitScenes visit 421383, 2 captures, Faro laser; `benchmarks/arkitscenes_421383/` |
| Photo | Any iPhone 15 or newer (0.5x camera) | Camera (native) | Depth-model scale after camera-height correction: 0.998 and 0.988 of LiDAR truth on two real captures; per room with 6-8 photos about 6-8% (90%), plus ~5% from the camera-height assumption. On rendered protocol views, 90% intervals contained 9 of 13 room dimensions and larger rooms came out 30-45% short: treat as rough. Rooms are not stitched | Real video frames vs LiDAR (scale); rendered views of the real apartment scan (geometry) |
| Video | Any iPhone 15 or newer (0.5x camera) | Camera (native) | Same method as photos, using sampled frames: same scale accuracy; rooms not stitched | As for photos |
Notes
- Not measured: end-to-end accuracy of the photo and video tiers on real protocol captures (no device available to us). The LiDAR numbers come from an iPad Pro dataset; the walk-in uses an iPhone.
- On a non-Pro iPhone the LiDAR tier is unavailable; use photo or video.
- Processing ran on a laptop with an NVIDIA T500 (4 GB). The depth model takes 0.36 s per image on it; the LiDAR tier needs no GPU.
