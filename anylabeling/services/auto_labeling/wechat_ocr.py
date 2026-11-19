"""微信 OCR（独立版）—— 不开微信、不登录，直接用微信自带的 wxocr 引擎。

两种跑法，优先第一种：

1. 常驻（默认）：用 wcocr.pyd 直连 mmmojo，自己拉起一个
   ``Weixin.exe --type=wxocr`` 子进程当服务，模型只加载一次。
   之后每张 1~2 秒，不再付冷启动的钱。
2. 一次一批：调 ocr-cli.exe，把整批图喂进去。进程跑完就退，
   每次都要重新解压运行时 + 加载模型（单张 6~19 秒）。

运行文件全在 models\\wechat_ocr\\ 里（路径必须纯英文）。
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QPointF

from .model import Model
from .types import AutoLabelingResult


def _ping_jun(kuai):
    """这一块图的置信度：引擎给的 rate 取平均"""
    if not kuai:
        return None
    return float(sum(float(k.get("rate", 0.0)) for k in kuai) / len(kuai))


def _ping_hang(kuai):
    """把同一行里的碎块拼成一句"""
    if not kuai:
        return ""
    kuai = sorted(kuai, key=lambda k: (k["top"], k["left"]))
    hang = []
    for k in kuai:
        gao = max(1.0, k["bottom"] - k["top"])
        if hang:
            shang = hang[-1]
            shang_gao = max(1.0, shang["bottom"] - shang["top"])
            if (
                abs(k["top"] - shang["top"]) < min(gao, shang_gao) * 0.6
                and k["left"] >= shang["left"] - gao
                and k["left"] - shang["right"] < gao * 3
            ):
                jian = k["left"] - shang["right"]
                sep = " " if jian > gao * 0.5 else ""
                shang["text"] += sep + k["text"]
                shang["right"] = max(shang["right"], k["right"])
                shang["bottom"] = max(shang["bottom"], k["bottom"])
                continue
        hang.append(dict(k))
    return "\n".join(h["text"] for h in hang)


class WechatOCR(Model):
    """微信 OCR 独立版：只认字，框要外部给。"""

    class Meta:
        required_config_names = ["type", "name", "display_name", "ocrcli_dir"]
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
        root = Path(__file__).resolve().parents[3]

        d = Path(str(self.config["ocrcli_dir"]))
        if not d.is_absolute():
            d = root / d
        self.dir = d
        self.exe = d / "ocr-cli.exe"
        self.runtime = d / "runtime"
        self.plugin = d / "runtime" / "ocr"
        self.tmp = d / "tmp"
        self.wxocr_dll = self.plugin / "wxocr.dll"
        for p in (self.runtime / "Weixin.exe", self.wxocr_dll):
            if not p.exists():
                raise FileNotFoundError(f"微信 OCR 组件缺失：{p}")
        self.tmp.mkdir(parents=True, exist_ok=True)

        # 版本子目录（CLI 只认 4.1.x，目录名是伪装的）
        ban = sorted(
            [p for p in self.runtime.iterdir() if p.is_dir() and p.name.startswith("4.")]
        )
        self.version_dir = ban[-1] if ban else None

        self.workers = max(1, int(self.config.get("workers", 2)))
        self.ocr_threshold = float(self.config.get("ocr_threshold", 0.0))
        self.replace = False
        self.filter_classes = None
        self.suo = threading.Lock()

        self.wcocr = None
        if self.config.get("changzhu", True):
            dll, ver = str(self.wxocr_dll), str(self.version_dir or self.runtime)
            if self.config.get("yong_xitong_weixin"):
                zhao = self._zhao_xitong()
                if zhao is None:
                    self.on_message("没找到系统微信或它的 OCR 插件")
                else:
                    dll, ver = str(zhao[0]), str(zhao[1])
                    self.on_message(f"用系统微信引擎：{zhao[1].name}")
            try:
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
                import wcocr  # noqa

                wcocr.init(dll, ver)
                self.wcocr = wcocr
                self.on_message("微信 OCR 常驻引擎已就绪（只冷启动这一次）")
            except Exception as e:  # noqa
                self.on_message(
                    f"常驻引擎起不来（{e}）"
                    + ("——微信要先打开" if self.config.get("yong_xitong_weixin") else "")
                )

        self.on_message(f"微信 OCR 就绪：{self.dir}")

    @staticmethod
    def _zhao_xitong():
        """找系统装的微信和它的 OCR 插件，返回 (wxocr.dll, 版本目录)"""
        huan = os.environ
        jia = [
            Path("F:/A01/Weixin"),
            Path("C:/Program Files/Tencent/Weixin"),
            Path("C:/Program Files (x86)/Tencent/Weixin"),
        ]
        if huan.get("LOCALAPPDATA"):
            jia.append(Path(huan["LOCALAPPDATA"]) / "Tencent" / "Weixin")
        ban = []
        for root in jia:
            if not root.is_dir():
                continue
            for p in root.iterdir():
                try:
                    if p.is_dir() and (p / "Weixin.dll").exists():
                        ban.append(p)
                except Exception:  # noqa
                    continue
        if not ban:
            return None
        ban.sort(key=lambda x: x.name)
        ver = ban[-1]

        chajian = []
        for base in filter(None, [huan.get("APPDATA"), str(Path("F:/Users"))]):
            if base == "F:/Users":
                base = "F:/Users/" + os.environ.get("USERNAME", "")
            root = Path(base) / "Tencent" / "xwechat" / "XPlugin" / "Plugins" / "WeChatOcr"
            if not root.is_dir():
                continue
            for p in root.iterdir():
                dll = p / "extracted" / "wxocr.dll"
                if dll.exists():
                    chajian.append(dll)
        if not chajian:
            return None
        chajian.sort(key=lambda x: str(x))
        return chajian[-1], ver

    # ------------------------------------------------------------ 工具
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

    def _load_bgr(self, image, image_path=None):
        if image_path and os.path.isfile(image_path):
            bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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

    def _cai(self, bgr, box):
        pts = self._points(box)
        x1 = max(0, int(np.floor(pts[:, 0].min())))
        y1 = max(0, int(np.floor(pts[:, 1].min())))
        x2 = min(bgr.shape[1], int(np.ceil(pts[:, 0].max())) + 1)
        y2 = min(bgr.shape[0], int(np.ceil(pts[:, 1].max())) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return bgr[y1:y2, x1:x2]

    # ------------------------------------------------------------ 跑引擎
    def _pao(self, pian_lu):
        """一批图片路径 -> {规范化路径: (文字, 置信度)}"""
        if not pian_lu:
            return {}
        if self.wcocr is not None:
            return self._pao_changzhu(pian_lu)
        return self._pao_cli(pian_lu)

    def _pao_changzhu(self, pian_lu):
        """常驻：逐张喂给已经起好的引擎，每张只付识别时间。"""
        out = {}
        with self.suo:
            for p in pian_lu:
                key = str(p).replace("\\", "/").lower()
                try:
                    r = self.wcocr.ocr(str(p))
                except Exception:  # noqa
                    # 引擎进程被关掉 / 崩了：按原样重建一次
                    try:
                        self.wcocr.init(
                            str(self.wxocr_dll),
                            str(self.version_dir or self.runtime),
                        )
                        r = self.wcocr.ocr(str(p))
                    except Exception:  # noqa
                        continue
                if not r or r.get("errcode", 0) != 0:
                    continue
                kuai = [
                    k for k in (r.get("ocr_response") or [])
                    if float(k.get("rate", 0)) >= self.ocr_threshold
                ]
                out[key] = (_ping_hang(kuai), _ping_jun(kuai))
        return out

    def _pao_cli(self, pian_lu):
        """退回模式：一次一批调 ocr-cli.exe（每批都要冷启动）。"""
        gong = self.tmp / f"_ysg_{os.getpid()}_{int(time.time() * 1000) % 100000}"
        gong.mkdir(parents=True, exist_ok=True)
        try:
            lie = gong / "list.txt"
            chu = gong / "out.jsonl"
            with open(lie, "w", encoding="utf-8") as f:
                for p in pian_lu:
                    f.write(str(p).replace("\\", "/") + "\n")

            env = dict(os.environ)
            env["TEMP"] = str(self.tmp)
            env["TMP"] = str(self.tmp)
            cmd = [
                str(self.exe),
                "--input-list", str(lie),
                "--wechat-dir", str(self.runtime),
                "--wechat-ocr-dir", str(self.plugin),
                "--workers", str(1 if len(pian_lu) <= 2 else self.workers),
                "--output", str(chu),
            ]
            subprocess.run(
                cmd, env=env, timeout=max(180, 30 * len(pian_lu)),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            out = {}
            if chu.exists():
                with open(chu, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:  # noqa
                            continue
                        r = rec.get("results") or {}
                        if r.get("errcode", 0) != 0:
                            continue
                        kuai = [
                            k for k in (r.get("ocr_response") or [])
                            if float(k.get("rate", 0)) >= self.ocr_threshold
                        ]
                        out[self._gui(str(rec.get("image_path", "")))] = (
                            _ping_hang(kuai),
                            _ping_jun(kuai),
                        )
            return out
        finally:
            shutil.rmtree(gong, ignore_errors=True)

    def _ocr_yuanshi(self, tu_lu):
        """调一次引擎，返回原始碎块（带像素坐标）"""
        with self.suo:
            try:
                r = self.wcocr.ocr(str(tu_lu))
            except Exception:  # noqa
                try:
                    self.wcocr.init(
                        str(self.wxocr_dll),
                        str(self.version_dir or self.runtime),
                    )
                    r = self.wcocr.ocr(str(tu_lu))
                except Exception:  # noqa
                    return []
        if not r or r.get("errcode", 0) != 0:
            return []
        return [
            k for k in (r.get("ocr_response") or [])
            if float(k.get("rate", 0)) >= self.ocr_threshold
        ]

    @staticmethod
    def _gui(lu):
        s = (lu or "").replace("\\", "/")
        while "//" in s:
            s = s.replace("//", "/")
        return s.lower()

    # ------------------------------------------------------------ 对外
    def predict_shapes(self, image, filename=None):
        """纯识别模型：点「运行」不画框。"""
        return AutoLabelingResult([], replace=False)

    def predict_shapes_from_boxes(self, image, boxes, image_path=None):
        t0 = time.time()
        if image is None or not boxes:
            return AutoLabelingResult([], replace=False), {}
        bgr = self._load_bgr(image, image_path)
        if bgr is None or bgr.size == 0:
            return AutoLabelingResult([], replace=False), {}
        t_loaded = time.time()

        gong = self.tmp / "_ysg_work"
        gong.mkdir(parents=True, exist_ok=True)

        lu_lie = []
        for i, box in enumerate(boxes):
            kuai = self._cai(bgr, box)
            p = gong / f"{i:04d}.png"
            if kuai is None or kuai.size == 0:
                kuai = np.full((8, 8, 3), 255, np.uint8)
            cv2.imencode(".png", kuai)[1].tofile(str(p))
            lu_lie.append(p)
        jieguo = self._pao(lu_lie)
        wenben = [
            jieguo.get(str(lu_lie[i]).replace("\\", "/").lower(), ("", None))
            for i in range(len(lu_lie))
        ]

        from anylabeling.views.labeling.shape import Shape

        shapes = []
        for i, box in enumerate(boxes):
            pts = self._points(box)
            text, xin = wenben[i] if i < len(wenben) else ("", None)
            shape = Shape(
                label="OCR", score=None, shape_type="rectangle", description=text
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
        gong = self.tmp / "_ysg_work"
        gong.mkdir(parents=True, exist_ok=True)
        p = gong / "one.png"
        cv2.imencode(".png", bgr)[1].tofile(str(p))
        text, _xin = self._pao([p]).get(
            str(p).replace("\\", "/").lower(), ("", None)
        )
        if not text:
            return [[]]
        box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return [[[box.tolist(), (text, 1.0)]]]

    def set_auto_labeling_filter_classes(self, classes):
        self.filter_classes = classes

    def set_auto_labeling_conf(self, value):
        self.ocr_threshold = float(value)

    def unload(self):
        try:
            if self.wcocr is not None:
                self.wcocr.destroy()
        except Exception:  # noqa
            pass
        self.wcocr = None
