"""Hayai OCR v2.7 (Supernova) —— 漫画/字幕用中日英 OCR，ONNX 版。

跟旧版（v2.1，SigLIP2 编码器 + 两个 decoder）架构完全不同：
没有独立图像编码器，图按 16×16 patch 切开直接进 prefill（NaFlex），
一张图能一次送最多 32 个框，框越多越划算。

只依赖 onnxruntime + numpy + pillow + tokenizers，不需要 transformers。

模型来源：https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx
"""

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from PyQt5.QtCore import QPointF
from tokenizers import Tokenizer

from .model import Model
from .types import AutoLabelingResult


# ---------------------------------------------------------------------------
# 预处理：按图大小挑一个 patch 预算，把整图缩放到 patch 网格里
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16384)
def _naflex_target_size(height, width, max_patches=256, patch_size=16,
                        min_patches=64, max_scale=4.0):
    s_min, s_max = 1e-5, max_scale
    while (s_max - s_min) >= 1e-5:
        scale = 0.5 * (s_min + s_max)
        th = max(patch_size, int(math.ceil(height * scale / patch_size)) * patch_size)
        tw = max(patch_size, int(math.ceil(width * scale / patch_size)) * patch_size)
        if (th // patch_size) * (tw // patch_size) <= max_patches:
            s_min = scale
        else:
            s_max = scale

    th = max(patch_size, int(math.ceil(height * s_min / patch_size)) * patch_size)
    tw = max(patch_size, int(math.ceil(width * s_min / patch_size)) * patch_size)
    nh, nw = th // patch_size, tw // patch_size

    if nh * nw < min_patches:
        a, b = s_min, 100.0
        target = min(min_patches, max_patches)
        while (b - a) >= 1e-5:
            s = 0.5 * (a + b)
            th = max(patch_size, int(math.ceil(height * s / patch_size)) * patch_size)
            tw = max(patch_size, int(math.ceil(width * s / patch_size)) * patch_size)
            if (th // patch_size) * (tw // patch_size) <= target:
                a = s
            else:
                b = s
        th = max(patch_size, int(math.ceil(height * a / patch_size)) * patch_size)
        tw = max(patch_size, int(math.ceil(width * a / patch_size)) * patch_size)
        nh, nw = th // patch_size, tw // patch_size

    return th, tw, nh, nw


@lru_cache(maxsize=16384)
def _best_max_patches(height, width, patch_size=16, min_patches=64, max_scale=4.0):
    info = {}
    for mp in (128, 192, 256, 384):
        th, tw, nh, nw = _naflex_target_size(
            height, width, mp, patch_size, min_patches, max_scale
        )
        m_vis = ((nh + 1) // 2) * ((nw + 1) // 2)
        info[mp] = (nh * nw, min(th / max(1, height), tw / max(1, width)), m_vis)

    n128, s128, m128 = info[128]
    n192, s192, m192 = info[192]
    n256, s256, m256 = info[256]
    n384, _, m384 = info[384]

    if n128 == n384:
        return 128, n128, m128

    hh, ww = max(1, height), max(1, width)
    aspect = max(hh, ww) / min(hh, ww)
    w_over_h = ww / hh
    is_square = 0.77 <= w_over_h <= 1.30
    is_horizontal = w_over_h > 1.30

    if not is_square:
        if not is_horizontal and s128 >= 1.65 and aspect < 3.8:
            return 128, n128, m128
        if is_horizontal and s128 >= 2.35 and aspect < 3.2:
            return 128, n128, m128

    if n384 > n256:
        if (
            s256 < 1.28
            or (is_square and s256 < 2.20)
            or (is_horizontal and s256 < 1.45)
            or (aspect >= 3.8 and s256 < 1.75)
            or (aspect >= 5.0 and s256 < 2.20)
        ):
            return 384, n384, m384

    if n192 == n128:
        return 128, n128, m128
    if n256 == n192:
        return 192, n192, m192
    if not is_square:
        if not is_horizontal and s192 >= 1.55 and aspect < 3.6:
            return 192, n192, m192
        if is_horizontal and s192 >= 1.85 and aspect < 3.0:
            return 192, n192, m192

    return 256, n256, m256


class _Pipeline:
    """三个 ONNX 图（prefill / decode）跑一遍，返回图片上的文字。"""

    def __init__(self, model_dir, precision="int4", use_gpu=True, threads=0):
        d = Path(model_dir)
        self.meta = json.loads((d / "hayai_onnx_config.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.pos_w = np.load(d / "pos_embedding.npy")
        self.pos_side = int(round(self.pos_w.shape[0] ** 0.5))
        self.pos_grid = self.pos_w.reshape(self.pos_side, self.pos_side, -1).astype(
            np.float32
        )

        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1

        suffix = "" if precision == "fp32" else "_" + precision
        self.prefill = ort.InferenceSession(
            str(d / ("hayai_prefill" + suffix + ".onnx")), opts, providers=providers
        )
        self.decode = ort.InferenceSession(
            str(d / ("hayai_decode" + suffix + ".onnx")), opts, providers=providers
        )
        self.provider = self.prefill.get_providers()[0]

    @lru_cache(maxsize=512)
    def _pos_embed(self, nh, nw):
        def aa(out_sz, in_sz):
            scale = float(in_sz) / float(out_sz)
            inv = (1.0 / scale) if scale > 1.0 else 1.0
            centers = scale * (np.arange(out_sz, dtype=np.float32) + 0.5)
            j = np.arange(in_sz, dtype=np.float32) + 0.5
            w = np.maximum(0.0, 1.0 - np.abs(j[None, :] - centers[:, None]) * inv)
            return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

        wh = aa(nh, self.pos_side)
        ww = aa(nw, self.pos_side)
        return np.einsum("ih,jw,hwd->ijd", wh, ww, self.pos_grid).reshape(nh * nw, -1)

    def _prepack(self, images, max_patches="auto"):
        ps = self.meta["patch_size"]
        min_p = self.meta["min_patches"]
        ms = self.meta["max_scale"]
        mult = self.meta["pad_to_multiple_of"]
        pdim = ps * ps * 3

        decoded = []
        for im in images:
            rgb = im.convert("RGB") if im.mode != "RGB" else im
            w, h = rgb.size
            mp = (
                _best_max_patches(h, w, ps, min_p, ms)[0]
                if max_patches == "auto"
                else int(max_patches)
            )
            th, tw, nh, nw = _naflex_target_size(h, w, mp, ps, min_p, ms)
            if (tw, th) != (w, h):
                rgb = rgb.resize((tw, th), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.uint8)
            patches = (
                arr.reshape(nh, ps, nw, ps, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(nh * nw, pdim)
            )
            decoded.append((patches, nh, nw))

        b = len(decoded)
        max_valid = max(nh * nw for _, nh, nw in decoded)
        n_keep = max(mult, ((max_valid + mult - 1) // mult) * mult)

        pv = np.zeros((b, n_keep, pdim), dtype=np.float32)
        pos_embeds = np.zeros((b, n_keep, self.pos_w.shape[1]), dtype=np.float32)
        shapes = np.empty((b, 2), dtype=np.int64)
        for j, (patches, nh, nw) in enumerate(decoded):
            n = nh * nw
            pv[j, :n] = patches.astype(np.float32) * (2.0 / 255.0) - 1.0
            pos_embeds[j, :n] = self._pos_embed(nh, nw)
            shapes[j] = (nh, nw)
        ho = (shapes[:, 0] + 1) // 2
        wo = (shapes[:, 1] + 1) // 2
        l_vis = int(np.max(ho * wo))
        return pv, pos_embeds, shapes, np.arange(l_vis, dtype=np.int64)

    def run(self, images, max_new_tokens=96, batch_size=32):
        return [
            t for t, _ in self.run_with_score(images, max_new_tokens, batch_size)
        ]

    def run_with_score(self, images, max_new_tokens=96, batch_size=32):
        """跟 run 一样，但每个结果多带一个置信度（各字概率的平均）"""
        out = [("", None)] * len(images)
        # 按 patch 预算分桶，同桶的才拼一个 batch
        buckets = {}
        for idx, im in enumerate(images):
            w, h = im.size
            mp = _best_max_patches(
                h, w, self.meta["patch_size"], self.meta["min_patches"],
                self.meta["max_scale"],
            )[0]
            buckets.setdefault(mp, []).append(idx)

        for mp, idxs in buckets.items():
            for s in range(0, len(idxs), batch_size):
                chunk = idxs[s:s + batch_size]
                preds = self._run_batch(
                    [images[i] for i in chunk], mp, max_new_tokens
                )
                for i, jieguo in zip(chunk, preds):
                    out[i] = jieguo
        return out

    def _run_batch(self, images, max_patches, max_new_tokens):
        b = len(images)
        bos = self.meta["bos_token_id"]
        eos = self.meta["eos_token_id"]
        pad = self.meta["pad_token_id"]

        pv, pos_embeds, shapes, l_vis_arange = self._prepack(images, max_patches)
        logits, k, v, vis_valid = self.prefill.run(
            None,
            {
                "pixel_values": pv,
                "pos_embeds": pos_embeds,
                "spatial_shapes": shapes,
                "bos_ids": np.full((b, 1), bos, dtype=np.int64),
                "l_vis_arange": l_vis_arange,
            },
        )
        nxt = np.argmax(logits, axis=-1).astype(np.int64)
        unfinished = (nxt != eos) & (nxt != pad)
        seqs = [[int(t)] for t in nxt]
        probs = [[] for _ in range(b)]
        _p = _ruan_max(logits.reshape(b, -1))
        for j in range(b):
            if unfinished[j]:
                probs[j].append(float(_p[j, int(nxt[j])]))
        mask = np.concatenate([vis_valid, np.ones((b, 1), dtype=np.int64)], axis=1)

        for step in range(1, max_new_tokens):
            if not np.any(unfinished):
                break
            logits, k, v = self.decode.run(
                None,
                {
                    "input_ids": nxt[:, None],
                    "past_attention_mask": mask,
                    "position_id": np.array([step], dtype=np.int64),
                    "k_cache_in": k,
                    "v_cache_in": v,
                },
            )
            mask = np.concatenate([mask, np.ones((b, 1), dtype=np.int64)], axis=1)
            nxt = np.where(
                unfinished, np.argmax(logits, axis=-1).astype(np.int64), pad
            )
            _p = _ruan_max(logits.reshape(b, -1))
            for j in range(b):
                if unfinished[j]:
                    seqs[j].append(int(nxt[j]))
                    probs[j].append(float(_p[j, int(nxt[j])]))
            unfinished &= (nxt != eos) & (nxt != pad)

        jieguo = []
        for j, s in enumerate(seqs):
            wenben = self.tokenizer.decode(
                [t for t in s if t not in (bos, eos, pad)], skip_special_tokens=True
            )
            xin = float(np.mean(probs[j])) if probs[j] else None
            jieguo.append((wenben, xin))
        return jieguo


def _ruan_max(x):
    """按最后一维做 softmax，拿每个候选词的概率"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class HayaiOCRv27(Model):
    """Hayai OCR v2.7：只认字，框要外部给。"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_dir",
        ]
        widgets = [
            "button_run",
            "button_recog_selected",
            "button_filter_classes",
            "input_conf",
            "edit_conf",
        ]
        output_modes = {"rectangle": "Rectangle"}
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message=None):
        super().__init__(model_config, on_message)
        self.root = Path(__file__).resolve().parents[3]

        def path(p):
            p = Path(str(p))
            if p.is_absolute():
                return p
            gen = self.root / p
            return gen if gen.exists() else p

        d = path(self.config["model_dir"])
        precision = str(self.config.get("precision", "int4"))
        self.on_message(f"载入 Hayai OCR v2.7（{precision}）")
        self.pipe = _Pipeline(
            d,
            precision=precision,
            use_gpu=bool(self.config.get("use_gpu", True)),
            threads=int(self.config.get("threads", 0) or 0),
        )
        self.on_message(f"Hayai v2.7 后端：{self.pipe.provider}")
        self.batch_size = max(1, int(self.config.get("batch_size", 32)))
        self.max_new_tokens = max(8, int(self.config.get("max_new_tokens", 96)))
        self.ocr_threshold = float(self.config.get("ocr_threshold", 0.0))
        self.replace = False
        self.filter_classes = None
        self.batch_mode = False
        if self.config.get("warmup", True):
            try:
                self.pipe.run([Image.new("RGB", (128, 64), "white")], 8)
            except Exception:  # noqa
                pass
        self.on_message("Hayai OCR v2.7 载入完成")

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _points(box):
        pts = np.asarray(box, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        if pts.shape[0] < 4:
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        return pts

    def _crop(self, bgr, box):
        pts = self._points(box)
        x1 = max(0, int(np.floor(pts[:, 0].min())))
        y1 = max(0, int(np.floor(pts[:, 1].min())))
        x2 = min(bgr.shape[1], int(np.ceil(pts[:, 0].max())) + 1)
        y2 = min(bgr.shape[0], int(np.ceil(pts[:, 1].max())) + 1)
        if x2 <= x1 or y2 <= y1:
            return Image.new("RGB", (8, 8), "white")
        return Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

    def _load_bgr(self, image, image_path=None):
        if image_path and os.path.isfile(image_path):
            bgr = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is not None:
                return bgr
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image
        arr = np.asarray(image)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------ 对外
    def predict_shapes(self, image, filename=None):
        """纯识别模型：点「运行」不画框，文字只回填到已有的框上。"""
        return AutoLabelingResult([], replace=False)

    def predict_shapes_from_boxes(self, image, boxes, image_path=None):
        t0 = time.time()
        if image is None or not boxes:
            return AutoLabelingResult([], replace=False), {}
        bgr = self._load_bgr(image, image_path)
        if bgr is None or bgr.size == 0:
            return AutoLabelingResult([], replace=False), {}
        t_loaded = time.time()

        crops = [self._crop(bgr, b) for b in boxes]
        # 一次性整批送进去，这是 v2.7 比旧版快的地方
        yuedu = self.pipe.run_with_score(
            crops, max_new_tokens=self.max_new_tokens, batch_size=self.batch_size
        )

        from anylabeling.views.labeling.shape import Shape

        shapes = []
        for box, (text, xin) in zip(boxes, yuedu):
            pts = self._points(box)
            shape = Shape(
                label="OCR", score=None, shape_type="rectangle", description=text or ""
            )
            shape.points = [QPointF(float(x), float(y)) for x, y in pts]
            if xin is not None:
                shape.ocr_confidence = xin
            shapes.append(shape)

        t_done = time.time()
        return AutoLabelingResult(shapes, replace=False), {
            "读图": t_loaded - t0,
            "裁剪+识别": t_done - t_loaded,
            "总": t_done - t0,
        }

    def ocr(self, img):
        """给视频工作台用：跟 PPOCRv6Wrapper 一样的返回格式。"""
        bgr = self._load_bgr(img)
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        text = self.pipe.run(
            [pil], max_new_tokens=self.max_new_tokens, batch_size=1
        )[0]
        if not text:
            return [[]]
        box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return [[[box.tolist(), (text, 1.0)]]]

    def set_auto_labeling_filter_classes(self, classes):
        self.filter_classes = classes

    def set_auto_labeling_conf(self, value):
        self.ocr_threshold = float(value)

    def unload(self):
        self.pipe = None
