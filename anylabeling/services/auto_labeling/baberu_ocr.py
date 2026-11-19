"""Baberu OCR：漫画 / 硬字幕用的中日英字符级 OCR（DINOv2 + 6 层 decoder）。

只做识别、不做检测：输入是一块已经裁好的文字区域（气泡 / 字幕块），
输出那上面的文字。三个 ONNX 图（vision -> decoder prefill -> decoder step，
KV cache）在纯 numpy 里串起来跑，只需要 onnxruntime + numpy + pillow，
不依赖 torch / transformers。

模型来源：https://huggingface.co/genshiai-daichi/baberu-ocr （Apache-2.0）
"""

import os
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from PyQt5.QtCore import QPointF

from .model import Model
from .types import AutoLabelingResult

# DINOv2 / ImageNet 预处理（整块缩到 224，不做中心裁剪）
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_PAST = [f"past_k{i}" for i in range(6)] + [f"past_v{i}" for i in range(6)]


class _Vocab:
    """字符级词表：id 0..3 是 <pad>/<bos>/<eos>/<unk>，id>=4 是 charset[id-4]。"""

    def __init__(self, vocab_json):
        with open(vocab_json, "r", encoding="utf-8") as fp:
            charset = __import__("json").load(fp)
        self.id2ch = {0: "", 1: "", 2: "", 3: ""}
        for i, ch in enumerate(charset):
            self.id2ch[i + 4] = ch
        self.bos, self.eos = 1, 2
        self.vocab_size = len(charset) + 4
        # 正常字符（字母/数字/假名/汉字），用来判断"同一个字连着重复几次就该掐"
        self.content_ids = {
            i + 4
            for i, ch in enumerate(charset)
            if len(ch) == 1
            and ch not in "ーｰ〜~"
            and unicodedata.category(ch)[0] in "LN"
        }

    def decode(self, ids):
        return "".join(self.id2ch.get(i, "") for i in ids if i >= 4)


class BaberuOCR(Model):
    """Baberu OCR：对已有框 / 整块图做文字识别。"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "vision_model_path",
            "prefill_model_path",
            "decode_model_path",
            "vocab_path",
        ]
        # 只有识别，没有检测：不给"大框拆分"那类按钮
        widgets = [
            "button_run",
            "button_recog_selected",
            "button_filter_classes",
            "input_conf",
            "edit_conf",
        ]
        output_modes = {
            "rectangle": "Rectangle",
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message=None):
        super().__init__(model_config, on_message)
        # anylabeling/services/auto_labeling/baberu_ocr.py → 项目根
        self.root = Path(__file__).resolve().parents[3]

        def path(key):
            p = Path(str(self.config.get(key, "")))
            if p.is_absolute():
                return p
            gen = self.root / p
            if gen.exists():
                return gen
            # 兜底：YAML 同目录下找
            return Path(str(self.config.get("config_file", ""))).parent / p

        vision_path = path("vision_model_path")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3      # 别让 ORT 的警告刷满日志
        # 线程数：4 核机器上开 2 反而比开满快（ORT 自己定的话经常抢核）
        threads = int(self.config.get("threads", 0) or 0)
        if threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        yao_gpu = bool(self.config.get("use_gpu"))
        if yao_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.on_message(f"载入 Baberu OCR：{vision_path.name}")

        def _jian(p, prov):
            return ort.InferenceSession(str(p), so, providers=prov)

        try:
            self.vision = _jian(vision_path, providers)
            self.prefill = _jian(path("prefill_model_path"), providers)
            self.step = _jian(path("decode_model_path"), providers)
        except Exception as e:  # noqa
            if providers[0] != "CUDAExecutionProvider":
                raise
            # 显存不够 / CUDA 组件缺失：退回 CPU，别让整个模型加载失败
            self.on_message(f"GPU 起不来（{e}），改用 CPU")
            providers = ["CPUExecutionProvider"]
            self.vision = _jian(vision_path, providers)
            self.prefill = _jian(path("prefill_model_path"), providers)
            self.step = _jian(path("decode_model_path"), providers)
        self.on_message(f"Baberu 后端：{self.vision.get_providers()[0]}")
        self.vocab = _Vocab(str(path("vocab_path")))

        self.max_new_tokens = int(self.config.get("max_new_tokens", 256))
        self.repetition_penalty = float(self.config.get("repetition_penalty", 1.2))
        self.max_content_run = int(self.config.get("max_content_run", 12))
        self.max_symbol_run = int(self.config.get("max_symbol_run", 16))
        self.ocr_threshold = float(self.config.get("ocr_threshold", 0.0))
        self.replace = False
        self.filter_classes = None
        # 先空跑一次：不然第一张图要替 cudnn 选卷积算法，白白多等一秒多
        if self.config.get("warmup", True):
            try:
                self._ocr(Image.new("RGB", (224, 224), "white"))
            except Exception:  # noqa
                pass
        self.on_message("Baberu OCR 载入完成")

    # ---------------------------------------------------------------- 推理
    @staticmethod
    def _preprocess(pil):
        img = pil.convert("RGB").resize((224, 224), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255.0 - _MEAN) / _STD
        return x.transpose(2, 0, 1)[None]

    def _ocr(self, pil):
        """一块图 -> (文字, 置信度)"""
        v = self.vocab
        vis = self.vision.run(
            ["vision_embeds"], {"pixel_values": self._preprocess(pil)}
        )[0]
        # fp16 那版在 GPU 上吐出来是 float16，后面 prefill 只吃 float32
        vis = np.asarray(vis, np.float32)
        out = self.prefill.run(
            None,
            {"vision_embeds": vis, "input_ids": np.array([[v.bos]], np.int64)},
        )
        logits = out[0][0, -1].astype(np.float64)
        present = out[1:]
        seq = [v.bos]
        toks = []
        probs = []
        pos = vis.shape[1] + 1

        for _ in range(self.max_new_tokens):
            if self.repetition_penalty != 1.0:
                for tid in set(seq):
                    s = logits[tid]
                    logits[tid] = (
                        s * self.repetition_penalty
                        if s < 0
                        else s / self.repetition_penalty
                    )
            # 同一个字 / 同一个符号连着蹦太多次：掐掉
            last = toks[-1] if toks else 0
            if last in v.content_ids:
                cap = self.max_content_run
            elif last > 3:
                cap = self.max_symbol_run
            else:
                cap = 0
            if cap:
                run = 0
                for t in reversed(toks):
                    if t == last:
                        run += 1
                    else:
                        break
                if run >= cap:
                    logits[last] = -np.inf

            # softmax 取最大概率当这一字的置信度
            z = logits - logits.max()
            p = np.exp(z)
            total = p.sum()
            nxt = int(np.argmax(logits))
            probs.append(float(p[nxt] / total) if total > 0 else 0.0)

            if nxt == v.eos:
                break
            toks.append(nxt)
            seq.append(nxt)
            if len(toks) >= self.max_new_tokens:
                break
            feed = {
                "input_ids": np.array([[nxt]], np.int64),
                "position_ids": np.array([[pos]], np.int64),
            }
            feed.update({nm: t for nm, t in zip(_PAST, present)})
            out = self.step.run(None, feed)
            logits = out[0][0, -1].astype(np.float64)
            present = out[1:]
            pos += 1

        text = v.decode(toks)
        score = float(np.mean(probs)) if probs else 0.0
        return text, score

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _points(box):
        pts = np.asarray(box, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        if pts.shape[0] < 4:
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32
            )
        return pts

    def _crop(self, bgr, box):
        pts = self._points(box)
        x1 = max(0, int(np.floor(pts[:, 0].min())))
        y1 = max(0, int(np.floor(pts[:, 1].min())))
        x2 = min(bgr.shape[1], int(np.ceil(pts[:, 0].max())) + 1)
        y2 = min(bgr.shape[0], int(np.ceil(pts[:, 1].max())) + 1)
        if x2 <= x1 or y2 <= y1:
            return Image.new("RGB", (1, 1), "white")
        return Image.fromarray(
            cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        )

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

    # ---------------------------------------------------------------- 对外接口
    def predict_shapes(self, image, filename=None):
        """纯识别模型，没有检测能力：点「运行」不画任何框。

        要出文字用「识别选中」/「识别全部」，那两个走 predict_shapes_from_boxes，
        只往已有的框上回填文字，不会凭空多出一个盖住整张图的框。
        """
        return AutoLabelingResult([], replace=False)

    def predict_shapes_from_boxes(self, image, boxes, image_path=None):
        """对给定的每个框，裁出来识别（不跑检测）。"""
        t0 = time.time()
        if image is None or not boxes:
            return AutoLabelingResult([], replace=False), {}
        bgr = self._load_bgr(image, image_path)
        if bgr is None or bgr.size == 0:
            return AutoLabelingResult([], replace=False), {}
        t_loaded = time.time()

        # 每个框独立跑一遍。
        # 注意：CUDA 后端下同一个 session 不能被多线程同时 run，
        # 否则会报 "illegal memory access" 直接把程序带崩。显卡上只能串行，
        # 反正 fp32 版一框 0.1 秒，串行也够快。
        crops = [self._crop(bgr, self._points(b)) for b in boxes]
        workers = int(self.config.get("workers", 1) or 1)
        if self.vision.get_providers()[0] == "CUDAExecutionProvider":
            workers = 1
        if len(crops) > 1 and workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=min(workers, len(crops))
            ) as pool:
                jieguo = list(pool.map(self._ocr, crops))
        else:
            jieguo = [self._ocr(c) for c in crops]

        from anylabeling.views.labeling.shape import Shape

        shapes = []
        for box, (text, score) in zip(boxes, jieguo):
            pts = self._points(box)
            if score < self.ocr_threshold:
                text = ""

            shape = Shape(
                label="OCR",
                score=None,
                shape_type="rectangle",
                description=text,
            )
            shape.points = [QPointF(float(x), float(y)) for x, y in pts]
            shape.ocr_confidence = score
            shapes.append(shape)

        t_done = time.time()
        return AutoLabelingResult(shapes, replace=False), {
            "读图": t_loaded - t0,
            "裁剪+识别": t_done - t_loaded,
            "总": t_done - t0,
        }

    def ocr(self, img):
        """给视频工作台用：跟 PPOCRv6Wrapper 一样的返回格式。

        传进来的本来就是一整块字幕区域，所以整块当一个框返回：
        [[ [四点框, (文字, 分数)], ... ]]
        """
        bgr = self._load_bgr(img)
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        text, score = self._ocr(
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        )
        if score < self.ocr_threshold:
            return [[]]
        box = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
        )
        return [[[box.tolist(), (text, score)]]]

    def set_auto_labeling_filter_classes(self, classes):
        self.filter_classes = classes

    def set_auto_labeling_conf(self, value):
        self.ocr_threshold = float(value)

    def unload(self):
        self.vision = None
        self.prefill = None
        self.step = None
        self.vocab = None
