"""Open-vocabulary damage detection: all prompts in ONE forward pass per image.
The transformers zero-shot pipeline runs the whole model once per prompt
(8 prompts -> 8 passes, ~30 s per frame on a 4 GB T500 for both OWLv2 and
Grounding DINO Tiny). Calling the model directly with every prompt together,
in half precision on the GPU, avoids that.
"""
from __future__ import annotations
import numpy as np
MODELS = {"owlv2": "google/owlv2-base-patch16-ensemble", "owlvit": "google/owlvit-base-patch32"}
PROMPTS = {
    "water stain on a wall or ceiling": "water_stain",
    "mould on a wall": "mold",
    "crack in a wall": "crack",
    "hole in a wall": "hole",
    "peeling paint": "peeling_paint",
    "white salt deposits on a wall": "efflorescence",
    "burn or scorch mark": "scorch",
    "warped floorboards": "warping",
}
class Detector:
    def __init__(self, model="owlvit", device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.kind = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODELS[model])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODELS[model], torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        self.texts = list(PROMPTS)
    def __call__(self, rgb: np.ndarray, threshold=0.05):
        """rgb: HxWx3 uint8. Returns [{'label', 'score', 'box': (x0, y0, x1, y1)}] in pixel coords."""
        torch = self.torch
        h, w = rgb.shape[:2]
        inputs = self.processor(text=[self.texts], images=rgb, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        # OWLv2 pads the image to a square (bottom/right), so boxes are relative to the square
        size = (max(h, w), max(h, w)) if self.kind == "owlv2" else (h, w)
        post = getattr(self.processor, "post_process_grounded_object_detection", None) \
            or self.processor.post_process_object_detection
        res = post(outputs=out, threshold=threshold, target_sizes=torch.tensor([size], device=self.device))[0]
        dets = []
        for s, l, b in zip(res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy(), res["boxes"].float().cpu().numpy()):
            x0, y0, x1, y1 = np.clip(b, 0, [w, h, w, h])
            dets.append({"label": PROMPTS[self.texts[int(l)]], "score": float(s), "box": (float(x0), float(y0), float(x1), float(y1))})
        return dets
