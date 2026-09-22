import cv2
import os
import os.path as osp
import shutil
import tempfile
import numpy as np

from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QScrollArea,
    QSizePolicy,
    QGroupBox,
    QTextEdit,
)

from anylabeling.views.labeling.chatbot.style import ChatbotDialogStyle
from anylabeling.views.labeling.logger import logger

# 防止 ONNX GPU 推理时 cuDNN 算法搜索超时
os.environ.setdefault("ORT_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC")

# 全局OCR模型（单例，只加载一次）
_ocr_model = None
_ocr_model_path = None  # 当前加载的V6模型路径

# OCR路径历史记录文件
OCR_PATH_HISTORY_FILE = osp.join(osp.expanduser("~"), ".ocr_path_history.json")


def load_ocr_path_history():
    """加载OCR路径历史记录"""
    import json
    try:
        if osp.exists(OCR_PATH_HISTORY_FILE):
            with open(OCR_PATH_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return [p for p in history if osp.exists(p)]
    except Exception as e:
        logger.warning(f"Failed to load OCR path history: {e}")
    return []


def save_ocr_path_history(ocr_path):
    """保存OCR路径到历史记录"""
    import json
    try:
        history = load_ocr_path_history()
        if ocr_path in history:
            history.remove(ocr_path)
        history.insert(0, ocr_path)
        history = history[:10]
        with open(OCR_PATH_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save OCR path history: {e}")


class PPOCRv6Wrapper:
    """包装 PP-OCRv6 的 TextSystem，提供与 ONNXPaddleOcr.ocr() 兼容的接口"""
    def __init__(self, text_system):
        self.text_system = text_system

    def ocr(self, img):
        dt_boxes, rec_res, scores = self.text_system(img)
        if dt_boxes is None:
            return None
        result = []
        for box, (text, score) in zip(dt_boxes, rec_res):
            result.append([box.tolist(), (text, score)])
        return [result]


class _OnnxYoloWrapper:
    """包装 ONNX 检测模型，提供轻量推理接口（跳过 Shape 创建，只返检测结果）"""
    def __init__(self, model):
        import numpy as np
        self.model = model
        classes = model.config.get("classes", [])
        if isinstance(classes, dict):
            classes = list(classes.values())
        self.names = {i: name for i, name in enumerate(classes)}
        self._last_boxes = np.array([])

    def __call__(self, frame, conf=0.25, classes=None, verbose=False):
        # 设置当前帧的阈值和过滤
        self.model.conf_thres = conf
        if classes is not None:
            self.model.filter_classes = classes
        # 轻量推理：直接走 preprocess→inference→postprocess
        blob = self.model.preprocess(frame, upsample_mode="letterbox")
        outputs = self.model.inference(blob)
        boxes, class_ids, scores, _, _ = self.model.postprocess(outputs)
        self._last_boxes = boxes if boxes.size else np.array([])
        return [self]

    @property
    def boxes(self):
        class _Boxes:
            def __init__(self, boxes):
                self._b = boxes
            def __len__(self):
                return len(self._b) if hasattr(self._b, '__len__') else 0
        return _Boxes(self._last_boxes)


def _load_model_from_yaml(yaml_path):
    """从 YAML 加载模型，与主界面相同的加载逻辑"""
    import yaml
    if not yaml_path or not osp.isfile(yaml_path):
        return None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['config_file'] = osp.abspath(yaml_path)
    config_dir = osp.dirname(yaml_path)
    for key in ['model_path', 'det_model_path', 'rec_model_path', 'rec_char_dict_path']:
        if key in config and not osp.isabs(str(config[key])):
            config[key] = osp.abspath(osp.join(config_dir, config[key]))
    model_type = config.get('type', '')
    def _on_message(msg):
        logger.info(f"[{model_type}] {msg}")
    # 类型 → 模块、类名 映射（与 model_manager 一致）
    TYPE_MAP = {
        "yolov5": ("anylabeling.services.auto_labeling.yolov5", "YOLOv5"),
        "yolov8": ("anylabeling.services.auto_labeling.yolov8", "YOLOv8"),
        "yolov9": ("anylabeling.services.auto_labeling.yolov9", "YOLOv9"),
        "yolov10": ("anylabeling.services.auto_labeling.yolov10", "YOLOv10"),
        "yolo11": ("anylabeling.services.auto_labeling.yolo11", "YOLO11"),
        "yolo12": ("anylabeling.services.auto_labeling.yolo12", "YOLO12"),
        "yolo26": ("anylabeling.services.auto_labeling.yolo26", "YOLO26"),
        "doclayout_yolo": ("anylabeling.services.auto_labeling.doclayout_yolo", "DocLayoutYOLO"),
        "gold_yolo": ("anylabeling.services.auto_labeling.gold_yolo", "GoldYOLO"),
        "rtdetr": ("anylabeling.services.auto_labeling.rtdetr", "RTDETR"),
        "rtdetrv2": ("anylabeling.services.auto_labeling.rtdetrv2", "RTDETRv2"),
        "rfdetr": ("anylabeling.services.auto_labeling.rfdetr", "RFDETR"),
        "ppocr_v4": ("anylabeling.services.auto_labeling.ppocr_v4", "PPOCRv4"),
        "ppocr_v6": ("anylabeling.services.auto_labeling.ppocr_v6", "PPOCRv6"),
    }
    try:
        if model_type in TYPE_MAP:
            module_path, class_name = TYPE_MAP[model_type]
            from importlib import import_module
            module = import_module(module_path)
            model_cls = getattr(module, class_name)
            model = model_cls(config, _on_message)
            logger.info(f"Model loaded from YAML: {model_type}")
            return model
    except Exception as e:
        logger.error(f"Failed to load model from YAML {yaml_path}: {e}")
        import traceback; traceback.print_exc()
    return None


def get_ocr_model(ocr_path=None):
    """获取OCR模型单例 - PP-OCRv6"""
    global _ocr_model, _ocr_model_path

    if ocr_path and ocr_path != _ocr_model_path:
        _ocr_model = None
        _ocr_model_path = None

    if _ocr_model is None:
        try:
            import yaml, onnxruntime as ort
            from anylabeling.services.auto_labeling.utils.ppocr_utils.text_system import TextSystem
            from anylabeling.services.auto_labeling.ppocr_v4 import Args

            if ocr_path:
                final_ocr_path = ocr_path
            else:
                final_ocr_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))), "ppocr_v6")

            if not osp.exists(final_ocr_path):
                logger.error(f"OCR path not found: {final_ocr_path}")
                return None

            yaml_file = None
            for f in os.listdir(final_ocr_path):
                if f.endswith('.yaml') or f.endswith('.yml'):
                    yaml_file = osp.join(final_ocr_path, f)
                    break
            if not yaml_file:
                logger.error(f"No YAML config found in: {final_ocr_path}")
                return None

            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            config['config_file'] = yaml_file
            config_dir = osp.dirname(yaml_file)
            for key in ['det_model_path', 'rec_model_path', 'rec_char_dict_path']:
                if key in config and not osp.isabs(config[key]):
                    config[key] = osp.abspath(osp.join(config_dir, config[key]))

            # GPU / providers
            providers = ["CPUExecutionProvider"]
            try:
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT", "device_id": 0})]
            except:
                pass
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.enable_mem_pattern = True

            det_net = ort.InferenceSession(config['det_model_path'], providers=providers, sess_options=sess_opts)
            rec_net = ort.InferenceSession(config['rec_model_path'], providers=providers, sess_options=sess_opts)

            args = Args(
                use_onnx=True, use_gpu=('CUDA' in str(providers)),
                use_xpu=False, use_npu=False, ir_optim=True,
                use_tensorrt=False, min_subgraph_size=15, precision="fp32",
                gpu_mem=500, gpu_id=0, page_num=0,
                det_algorithm="DB", det_model=det_net,
                det_limit_side_len=config.get('det_limit_side_len', 736),
                det_limit_type=config.get('det_limit_type', 'min'),
                det_box_type="quad",
                det_db_thresh=config.get('det_db_thresh', 0.2),
                det_db_box_thresh=config.get('det_db_box_thresh', 0.45),
                det_db_unclip_ratio=config.get('det_db_unclip_ratio', 1.4),
                max_batch_size=10, use_dilation=False,
                det_db_score_mode="fast",
                det_east_score_thresh=0.8, det_east_cover_thresh=0.1,
                det_east_nms_thresh=0.2,
                det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2,
                det_pse_thresh=0, det_pse_box_thresh=0.85,
                det_pse_min_area=16, det_pse_scale=1,
                det_fce_box_type="poly",
                rec_algorithm="SVTR", rec_model=rec_net,
                rec_char_dict_path=config['rec_char_dict_path'],
                rec_batch_num=config.get('rec_batch_num', 6),
                rec_image_shape=config.get('rec_image_shape', "3,48,320"),
                max_text_length=25, use_space_char=True,
                use_angle_cls=config.get('use_angle_cls', False),
                drop_score=config.get('drop_score', 0.5),
                cls_model=None, cls_batch_num=6, cls_image_shape="3,48,192",
                cls_thresh=0.9, label_list=['0', '180'],
                use_onnx_det_output_tensors=True,
                use_onnx_rec_output_tensors=False,
                draw_img_save_dir="./inference_results_cls",
                save_crop_res=False,
            )

            text_system = TextSystem(args)
            _ocr_model = PPOCRv6Wrapper(text_system)
            _ocr_model_path = final_ocr_path
            logger.info(f"PP-OCRv6 loaded from: {final_ocr_path}")

            save_ocr_path_history(final_ocr_path)
        except Exception as e:
            logger.error(f"Failed to load PP-OCRv6: {e}")
            import traceback
            traceback.print_exc()
            _ocr_model = None
            _ocr_model_path = None
    return _ocr_model


def format_srt_time(ms):
    """将毫秒转换为SRT时间格式 HH:MM:SS,mmm"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_ass_time(ms):
    """将毫秒转换为ASS时间格式 H:MM:SS.cc (centiseconds)"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    centiseconds = (ms % 1000) // 10  # ASS使用厘秒（百分之一秒）
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


# ASS文件头模板
ASS_TEMPLATE = """[Script Info]
; Script generated by X-AnyLabeling
Title: Default Aegisub file
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,新兰圆-B,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,138,1
Style: 旁白-黑字白边,新兰圆-B,40,&H00000000,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,1,80,10,5,1
Style: 字幕对比用,新兰圆-B,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,138,1
Style: 姑妄,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 频道logo,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 版权,新兰圆-B,70,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,4,10,10,10,1
Style: 声明字幕,新兰圆-B,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,80,10,15,1
Style: 男1,新兰圆-B,40,&H00EDA52D,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,1,60,10,5,1
Style: 男2,新兰圆-B,40,&H00BFAA0D,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,1,85,10,5,1
Style: 男3,新兰圆-B,40,&H00389F30,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,4,2,1,85,10,5,1
Style: 男4,新兰圆-B,40,&H00940064,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,4,2,1,85,10,5,1
Style: 男5,新兰圆-B,40,&H00093B72,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,4,2,1,85,10,5,1
Style: 女1,新兰圆-B,40,&H00F300EA,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,4,2,1,60,10,5,1
Style: 女2,新兰圆-B,40,&H004508D8,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,1,60,10,5,1
Style: 女3,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,70,10,0,1
Style: 女4,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女5,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女6,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女7,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女8,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女9,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 女10,新兰圆-B,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,10,1
Style: 旁边 竖向,@新兰圆-B,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,-90,1,2,2,2,10,10,10,1
Style: 动漫字幕,新兰圆-B,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 01谷歌,新兰圆-B,70,&H000A51F2,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,125,1
Style: 02popogo,新兰圆-B,30,&H000987E8,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,125,1
Style: 03腾讯,新兰圆-B,50,&H00EC340B,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,135,1
Style: 04彩云,新兰圆-B,30,&H00BDEC0B,&H00A4FE0D,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,125,1
Style: 05百度,新兰圆-B,100,&H000AF211,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,125,1
Style: 06有道,新兰圆-B,50,&H000809EF,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,125,1
Style: 07搜狗,新兰圆-B,60,&H00F10ACF,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,1,10,10,135,1
Style: 内嵌水印,新兰圆-B,50,&HFFFFFFFF,&H000000FF,&H89000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: 内嵌水印竖,新兰圆-B,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: 内嵌水印竖1,@新兰圆-B,30,&HFFFFFFFF,&H000000FF,&H89000000,&H00000000,0,0,0,0,100,100,0,-90,1,2,0,2,10,10,10,1
Style: 内嵌水印竖着2,@新兰圆-B,50,&HFFFFFFFF,&H000000FF,&H89000000,&H00000000,0,0,0,0,100,100,0,-180,1,2,0,2,10,10,10,1
Style: 内嵌水印竖3,@新兰圆-B,30,&HFFFFFFFF,&H000000FF,&H89000000,&H00000000,0,0,0,0,100,100,0,-90,1,2,0,2,10,10,10,1
Style: 内嵌水印2,新兰圆-B,50,&HFFFFFFFF,&H000000FF,&H89000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: 旁白-黑字白边1BUG,新兰圆-B,50,&H00000000,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,4,0,0,1,1
Style: 内嵌水印透明,新兰圆-B,40,&HD0FFFFFF,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,0,1,1,0,5,10,10,10,1
Style: 内嵌水印透明1,新兰圆-B,100,&HE60000FF,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,0,1,1,0,5,10,10,10,1
Style: 内嵌水印透明2,新兰圆-B,40,&HD0F92E01,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,0,1,1,0,5,10,10,10,1
Style: 内嵌水印透明3,新兰圆-B,40,&HD0B901F7,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,0,1,1,0,5,10,10,10,1
Style: 内嵌水印透明黑边,新兰圆-B,40,&H9FFFFFFF,&H000000FF,&H2C000000,&H00000000,0,0,0,0,100,100,0,0,1,0.5,0,1,10,10,10,1
Style: 女1常用暗粉色,新兰圆-B,40,&H00C200D1,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,4,2,1,60,10,5,1
Style: 女2暗绿色,新兰圆-B,40,&H00117D0B,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,3,2,1,90,10,5,1
Style: 旁白-黑字白边垂直,@新兰圆-B,40,&H00000000,&H000000FF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,-90,1,50,0,5,80,10,5,1
Style: 字幕式水印横,新兰圆-B,70,&HE00000FE,&H000000FF,&HE7000000,&HFF0000FE,0,0,0,0,100,100,0,0,1,2,2,1,10,10,138,1
Style: 字幕式水印竖,新兰圆-B,50,&HCB0000FF,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,90,1,1,0,2,50,10,100,1
Style: 骂傻逼专用,新兰圆-B,55,&HD00000FF,&H000000FF,&HFFFFFFFF,&H0021212D,0,0,0,0,100,100,0,0,1,1,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_msg_box_style,
    get_progress_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)
from anylabeling.views.labeling.widgets import Popup


# 默认OCR过滤规则（支持正则表达式）
DEFAULT_OCR_FILTER_RULES = [
    r'^\s*$',  # 只有空格或空白
    r'^[「」『』【】〖〗《》〈〉（）\(\)\[\]{}\.。，,、；;：:！!？\?…·\-—_\s\.。\・\･]+$',  # 只有标点符号
    r'^\.+$',  # 只有英文点
    r'^。+$',  # 只有中文句号
    r'^…+$',  # 只有省略号
    r'^[\.。…·]+$',  # 只有各种点号组合
]


# OCR过滤规则配置文件路径
OCR_FILTER_RULES_FILE = osp.join(osp.expanduser("~"), ".ocr_filter_rules.json")


def load_ocr_filter_rules():
    """从配置文件加载OCR过滤规则"""
    import json
    try:
        if osp.exists(OCR_FILTER_RULES_FILE):
            with open(OCR_FILTER_RULES_FILE, 'r', encoding='utf-8') as f:
                rules = json.load(f)
                if isinstance(rules, list) and len(rules) > 0:
                    return rules
    except Exception as e:
        logger.warning(f"Failed to load OCR filter rules: {e}")
    return DEFAULT_OCR_FILTER_RULES.copy()


def save_ocr_filter_rules(rules):
    """保存OCR过滤规则到配置文件"""
    import json
    try:
        with open(OCR_FILTER_RULES_FILE, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR filter rules saved to {OCR_FILTER_RULES_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save OCR filter rules: {e}")


class OcrFilterDialog(QDialog):
    """OCR过滤规则编辑对话框"""
    
    def __init__(self, parent=None, rules=None):
        super().__init__(parent)
        # 优先使用传入的规则，否则从配置文件加载
        self.rules = rules if rules else load_ocr_filter_rules()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("OCR过滤规则")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setStyleSheet(get_msg_box_style())
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # 说明
        hint_label = QLabel("每行一条规则，支持正则表达式\n匹配的OCR结果将被跳过不生成")
        hint_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(hint_label)
        
        # 规则编辑区
        self.rules_edit = QTextEdit()
        self.rules_edit.setPlaceholderText("每行一条过滤规则...\n支持正则表达式，如:\n^[。.…]+$  匹配只有点号的文本\n^\\s*$  匹配空白文本")
        self.rules_edit.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, Monaco, monospace;
                font-size: 13px;
                line-height: 1.5;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        # 加载规则
        self.rules_edit.setPlainText("\n".join(self.rules))
        layout.addWidget(self.rules_edit)
        
        # 预设按钮
        preset_layout = QHBoxLayout()
        
        reset_btn = QPushButton("恢复默认")
        reset_btn.setToolTip("恢复为默认过滤规则")
        reset_btn.clicked.connect(self.reset_to_default)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self.rules_edit.clear())
        
        test_btn = QPushButton("测试规则")
        test_btn.setToolTip("测试当前规则是否有语法错误")
        test_btn.clicked.connect(self.test_rules)
        
        preset_layout.addWidget(reset_btn)
        preset_layout.addWidget(clear_btn)
        preset_layout.addStretch()
        preset_layout.addWidget(test_btn)
        layout.addLayout(preset_layout)
        
        # 测试结果显示
        self.test_result_label = QLabel("")
        self.test_result_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.test_result_label)
        
        # 确定取消按钮
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet(get_ok_btn_style())
        ok_btn.clicked.connect(self.on_accept)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def reset_to_default(self):
        """恢复默认规则"""
        self.rules_edit.setPlainText("\n".join(DEFAULT_OCR_FILTER_RULES))
        
    def test_rules(self):
        """测试规则语法"""
        import re
        rules = self.get_rules()
        errors = []
        for i, rule in enumerate(rules, 1):
            try:
                re.compile(rule)
            except re.error as e:
                errors.append(f"第{i}行: {e}")
        
        if errors:
            self.test_result_label.setText("❌ " + "; ".join(errors))
            self.test_result_label.setStyleSheet("color: red; font-size: 12px;")
        else:
            self.test_result_label.setText(f"✓ {len(rules)}条规则语法正确")
            self.test_result_label.setStyleSheet("color: green; font-size: 12px;")
    
    def get_rules(self):
        """获取规则列表"""
        text = self.rules_edit.toPlainText()
        rules = [line.strip() for line in text.split("\n") if line.strip()]
        return rules
    
    def on_accept(self):
        """确定按钮点击，保存规则"""
        rules = self.get_rules()
        save_ocr_filter_rules(rules)
        self.accept()


class RegionSelectLabel(QLabel):
    """可以用鼠标框选区域的Label，支持8个控制点调整"""
    HANDLE_SIZE = 8  # 控制点大小
    
    # 控制点位置常量
    HANDLE_NONE = 0
    HANDLE_TL = 1  # 左上
    HANDLE_TR = 2  # 右上
    HANDLE_BL = 3  # 左下
    HANDLE_BR = 4  # 右下
    HANDLE_T = 5   # 上中
    HANDLE_B = 6   # 下中
    HANDLE_L = 7   # 左中
    HANDLE_R = 8   # 右中
    HANDLE_MOVE = 9  # 移动整个选区
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None
        self.start_point = None
        self.is_selecting = False
        self.is_adjusting = False
        self.active_handle = self.HANDLE_NONE
        self.drag_start_rect = None
        self.drag_start_pos = None
        self.scale_factor = 1.0
        self.on_selection_changed = None
        self.setMouseTracking(True)
    
    def set_image(self, pixmap):
        self.setPixmap(pixmap)
        self.selection_rect = None
    
    def _get_handle_rects(self):
        """获取8个控制点的矩形区域"""
        if not self.selection_rect:
            return {}
        
        r = self.selection_rect
        s = self.HANDLE_SIZE
        hs = s // 2
        
        return {
            self.HANDLE_TL: QRect(r.left() - hs, r.top() - hs, s, s),
            self.HANDLE_TR: QRect(r.right() - hs, r.top() - hs, s, s),
            self.HANDLE_BL: QRect(r.left() - hs, r.bottom() - hs, s, s),
            self.HANDLE_BR: QRect(r.right() - hs, r.bottom() - hs, s, s),
            self.HANDLE_T: QRect(r.center().x() - hs, r.top() - hs, s, s),
            self.HANDLE_B: QRect(r.center().x() - hs, r.bottom() - hs, s, s),
            self.HANDLE_L: QRect(r.left() - hs, r.center().y() - hs, s, s),
            self.HANDLE_R: QRect(r.right() - hs, r.center().y() - hs, s, s),
        }
    
    def _get_handle_at(self, pos):
        """获取鼠标位置的控制点"""
        handles = self._get_handle_rects()
        for handle, rect in handles.items():
            if rect.contains(pos):
                return handle
        # 检查是否在选区内（移动）
        if self.selection_rect and self.selection_rect.contains(pos):
            return self.HANDLE_MOVE
        return self.HANDLE_NONE
    
    def _update_cursor(self, handle):
        """根据控制点更新鼠标样式"""
        from PyQt5.QtCore import Qt
        cursors = {
            self.HANDLE_TL: Qt.SizeFDiagCursor,
            self.HANDLE_BR: Qt.SizeFDiagCursor,
            self.HANDLE_TR: Qt.SizeBDiagCursor,
            self.HANDLE_BL: Qt.SizeBDiagCursor,
            self.HANDLE_T: Qt.SizeVerCursor,
            self.HANDLE_B: Qt.SizeVerCursor,
            self.HANDLE_L: Qt.SizeHorCursor,
            self.HANDLE_R: Qt.SizeHorCursor,
            self.HANDLE_MOVE: Qt.SizeAllCursor,
        }
        if handle in cursors:
            self.setCursor(cursors[handle])
        else:
            self.setCursor(Qt.CrossCursor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            handle = self._get_handle_at(pos)
            
            if handle != self.HANDLE_NONE and self.selection_rect:
                # 开始调整
                self.is_adjusting = True
                self.active_handle = handle
                self.drag_start_rect = QRect(self.selection_rect)
                self.drag_start_pos = pos
            else:
                # 开始新选区
                self.start_point = pos
                self.is_selecting = True
                self.selection_rect = QRect(self.start_point, self.start_point)
            
            self.update()
            self._notify_selection_changed()
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        
        if self.is_adjusting and self.drag_start_rect:
            # 调整选区
            dx = pos.x() - self.drag_start_pos.x()
            dy = pos.y() - self.drag_start_pos.y()
            r = QRect(self.drag_start_rect)
            
            if self.active_handle == self.HANDLE_MOVE:
                r.translate(dx, dy)
            elif self.active_handle == self.HANDLE_TL:
                r.setTopLeft(r.topLeft() + QPoint(dx, dy))
            elif self.active_handle == self.HANDLE_TR:
                r.setTopRight(r.topRight() + QPoint(dx, dy))
            elif self.active_handle == self.HANDLE_BL:
                r.setBottomLeft(r.bottomLeft() + QPoint(dx, dy))
            elif self.active_handle == self.HANDLE_BR:
                r.setBottomRight(r.bottomRight() + QPoint(dx, dy))
            elif self.active_handle == self.HANDLE_T:
                r.setTop(r.top() + dy)
            elif self.active_handle == self.HANDLE_B:
                r.setBottom(r.bottom() + dy)
            elif self.active_handle == self.HANDLE_L:
                r.setLeft(r.left() + dx)
            elif self.active_handle == self.HANDLE_R:
                r.setRight(r.right() + dx)
            
            self.selection_rect = r.normalized()
            self.update()
            self._notify_selection_changed()
        elif self.is_selecting and self.start_point:
            # 绘制新选区
            self.selection_rect = QRect(self.start_point, pos).normalized()
            self.update()
            self._notify_selection_changed()
        else:
            # 更新鼠标样式
            handle = self._get_handle_at(pos)
            self._update_cursor(handle)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_selecting:
                self.is_selecting = False
                if self.selection_rect and self.selection_rect.width() < 10 and self.selection_rect.height() < 10:
                    self.selection_rect = None
            
            self.is_adjusting = False
            self.active_handle = self.HANDLE_NONE
            self.drag_start_rect = None
            self.drag_start_pos = None
            
            self.update()
            self._notify_selection_changed()
    
    def _notify_selection_changed(self):
        """通知选区变化"""
        if self.on_selection_changed:
            if self.selection_rect:
                x = int(self.selection_rect.x() / self.scale_factor)
                y = int(self.selection_rect.y() / self.scale_factor)
                w = int(self.selection_rect.width() / self.scale_factor)
                h = int(self.selection_rect.height() / self.scale_factor)
                self.on_selection_changed(x, y, w, h)
            else:
                self.on_selection_changed(None, None, None, None)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect:
            painter = QPainter(self)
            # 绘制选区边框
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            # 半透明填充
            painter.fillRect(self.selection_rect, QColor(255, 0, 0, 30))
            
            # 绘制8个控制点
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 0, 0))
            for rect in self._get_handle_rects().values():
                painter.drawRect(rect)
    
    def get_selection(self):
        """返回选择的区域"""
        return self.selection_rect
    
    def clear_selection(self):
        self.selection_rect = None
        self.update()
        self._notify_selection_changed()


class RegionSelectDialog(QDialog):
    """区域选择对话框 - 支持视频预览"""
    def __init__(self, parent=None, frame=None, video_capture=None, total_frames=0, fps=30, current_region=None):
        super().__init__(parent)
        self.frame = frame
        self.video_capture = video_capture
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame_idx = 0
        self.selected_region = None
        self.scale_factor = 1.0
        self.user_scale = 1.0  # 用户缩放比例
        self.initial_region = current_region  # 保存初始区域，用于显示
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("选择检测区域")
        # 去掉问号，添加最大化按钮
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(get_msg_box_style())
        layout = QVBoxLayout()
        
        # 顶部工具栏：提示文字 + 缩放按钮
        top_layout = QHBoxLayout()
        hint_label = QLabel("用鼠标框选要检测变化的区域（不选则检测整个画面）")
        top_layout.addWidget(hint_label)
        top_layout.addStretch()
        
        # 缩放按钮
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.setToolTip("缩小")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.setToolTip("放大")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        
        self.zoom_fit_btn = QPushButton("适应")
        self.zoom_fit_btn.setToolTip("适应窗口")
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)
        
        self.zoom_100_btn = QPushButton("100%")
        self.zoom_100_btn.setToolTip("原始大小")
        self.zoom_100_btn.clicked.connect(self.zoom_100)
        
        top_layout.addWidget(self.zoom_out_btn)
        top_layout.addWidget(self.zoom_label)
        top_layout.addWidget(self.zoom_in_btn)
        top_layout.addWidget(self.zoom_fit_btn)
        top_layout.addWidget(self.zoom_100_btn)
        
        layout.addLayout(top_layout)
        
        # 图片显示区域
        self.image_label = RegionSelectLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setMinimumSize(640, 400)
        
        # 显示图片
        if self.frame is not None:
            self.display_frame()
            # 如果有初始区域，显示它
            if self.initial_region:
                self.set_initial_region(self.initial_region)
        
        layout.addWidget(self.scroll_area)
        
        # 视频进度条（如果有视频）
        if self.video_capture is not None and self.total_frames > 0:
            from PyQt5.QtWidgets import QSlider
            
            slider_layout = QHBoxLayout()
            
            self.frame_label = QLabel("帧: 0")
            self.frame_label.setMinimumWidth(80)
            
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.valueChanged.connect(self.on_slider_changed)
            
            self.time_label = QLabel("00:00:00")
            self.time_label.setMinimumWidth(60)
            
            slider_layout.addWidget(self.frame_label)
            slider_layout.addWidget(self.frame_slider)
            slider_layout.addWidget(self.time_label)
            
            layout.addLayout(slider_layout)
        
        # 选区信息显示
        self.selection_info_label = QLabel("选区: 未选择")
        self.selection_info_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.selection_info_label)
        
        # 设置回调
        self.image_label.on_selection_changed = self.on_selection_changed
        
        # 历史区域选择
        history_layout = QHBoxLayout()
        history_label = QLabel("历史区域:")
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(250)
        self.history_combo.addItem("-- 选择历史区域 --")
        self.load_region_history()
        self.history_combo.currentIndexChanged.connect(self.on_history_selected)
        
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_combo)
        history_layout.addStretch()
        layout.addLayout(history_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("清除选区")
        clear_btn.clicked.connect(self.clear_selection)
        
        ok_button = QPushButton("确定")
        ok_button.setStyleSheet(get_ok_btn_style())
        ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("取消")
        cancel_button.setStyleSheet(get_cancel_btn_style())
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(900, 700)
    
    def on_selection_changed(self, x, y, w, h):
        """选区变化时更新显示"""
        if x is not None:
            self.selection_info_label.setText(f"选区: 位置({x}, {y})  大小 {w} x {h}")
            self.selection_info_label.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
        else:
            self.selection_info_label.setText("选区: 未选择")
            self.selection_info_label.setStyleSheet("color: #666; font-size: 12px;")
    
    def zoom_in(self):
        """放大"""
        self.user_scale = min(3.0, self.user_scale * 1.25)
        self.display_frame()
        self.zoom_label.setText(f"{int(self.user_scale * 100)}%")
    
    def zoom_out(self):
        """缩小"""
        self.user_scale = max(0.1, self.user_scale / 1.25)
        self.display_frame()
        self.zoom_label.setText(f"{int(self.user_scale * 100)}%")
    
    def zoom_fit(self):
        """适应窗口"""
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        scroll_w = self.scroll_area.width() - 20
        scroll_h = self.scroll_area.height() - 20
        self.user_scale = min(scroll_w / w, scroll_h / h)
        self.display_frame()
        self.zoom_label.setText(f"{int(self.user_scale * 100)}%")
    
    def zoom_100(self):
        """原始大小"""
        self.user_scale = 1.0
        self.display_frame()
        self.zoom_label.setText("100%")
    
    def on_slider_changed(self, value):
        """进度条改变时更新帧"""
        if self.video_capture is None:
            return
        
        self.current_frame_idx = value
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, value)
        ret, frame = self.video_capture.read()
        if ret:
            self.frame = frame
            self.display_frame()
        
        # 更新标签
        self.frame_label.setText(f"帧: {value}")
        time_sec = value / self.fps if self.fps > 0 else 0
        hours = int(time_sec // 3600)
        minutes = int((time_sec % 3600) // 60)
        seconds = int(time_sec % 60)
        self.time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
    def display_frame(self):
        """显示视频帧"""
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        
        # 使用用户缩放比例
        self.scale_factor = self.user_scale
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        if self.scale_factor != 1.0:
            display_frame = cv2.resize(self.frame, (new_w, new_h))
        else:
            display_frame = self.frame
            
        # 转换为QPixmap
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.image_label.set_image(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        self.image_label.scale_factor = self.scale_factor  # 同步缩放比例
    
    def set_initial_region(self, region):
        """设置初始选区（用于显示之前选择的区域）"""
        if region and len(region) == 4:
            x, y, w, h = region
            # 转换为显示坐标
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            # 设置选区
            self.image_label.selection_rect = QRect(sx, sy, sw, sh)
            self.image_label.update()
            self.image_label._notify_selection_changed()
        
    def clear_selection(self):
        self.image_label.clear_selection()
        
    def get_region(self):
        """获取选择的区域（原始图片坐标）"""
        rect = self.image_label.get_selection()
        if rect is None:
            return None
            
        # 转换回原始图片坐标
        x = int(rect.x() / self.scale_factor)
        y = int(rect.y() / self.scale_factor)
        w = int(rect.width() / self.scale_factor)
        h = int(rect.height() / self.scale_factor)
        
        # 保存到历史记录
        self.save_region_history(x, y, w, h)
        
        return (x, y, w, h)
    
    def load_region_history(self):
        """加载区域历史记录"""
        import json
        self.region_history_file = osp.join(osp.expanduser("~"), ".region_history.json")
        try:
            if osp.exists(self.region_history_file):
                with open(self.region_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    for item in history:
                        x, y, w, h = item['x'], item['y'], item['w'], item['h']
                        self.history_combo.addItem(f"{w}x{h} - ({x}, {y})", item)
        except Exception as e:
            logger.warning(f"Failed to load region history: {e}")
    
    def save_region_history(self, x, y, w, h):
        """保存区域到历史记录"""
        import json
        try:
            history = []
            if osp.exists(self.region_history_file):
                with open(self.region_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            new_item = {'x': x, 'y': y, 'w': w, 'h': h}
            
            # 移除重复项
            history = [item for item in history if not (item['x'] == x and item['y'] == y and item['w'] == w and item['h'] == h)]
            history.insert(0, new_item)
            
            # 只保留最近10个
            history = history[:10]
            
            with open(self.region_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save region history: {e}")
    
    def on_history_selected(self, index):
        """从历史记录选择区域"""
        if index <= 0:
            return
        item = self.history_combo.itemData(index)
        if item:
            x, y, w, h = item['x'], item['y'], item['w'], item['h']
            # 转换为显示坐标
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            # 设置选区
            self.image_label.selection_rect = QRect(sx, sy, sw, sh)
            self.image_label.update()
            self.image_label._notify_selection_changed()
            # 重置下拉框
            self.history_combo.setCurrentIndex(0)


class FrameExtractionDialog(QDialog):
    def __init__(self, parent=None, total_frames=0, fps=0, first_frame=None, video_capture=None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.fps = fps
        self.first_frame = first_frame  # 用于区域选择
        self.video_capture = video_capture  # 视频捕获对象
        self.detection_region = None  # 检测区域
        self.video_duration = total_frames / fps if fps > 0 else 0  # 视频时长（秒）
        # 只阻塞当前应用，不阻塞其他应用（如资源管理器）
        self.setWindowModality(Qt.ApplicationModal)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("帧提取设置")
        self.setStyleSheet(get_msg_box_style())
        layout = QVBoxLayout()

        # Interval input with unit selector
        interval_layout = QHBoxLayout()
        interval_label = QLabel("提取间隔:")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, max(1, self.total_frames))
        self.interval_spin.setValue(1)
        self.interval_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        self.interval_spin.setMinimumWidth(80)
        self.interval_spin.valueChanged.connect(self.update_estimated_count)
        
        # 单位选择
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["帧", "秒", "分钟"])
        self.unit_combo.setMinimumWidth(70)
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        
        # 预计提取数量显示
        self.count_label = QLabel()
        self.count_label.setStyleSheet("color: #8B008B; font-size: 18px; font-weight: bold;")
        self.count_label.setMinimumWidth(80)
        
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addWidget(self.unit_combo)
        interval_layout.addStretch()
        interval_layout.addWidget(QLabel("预计:"))
        interval_layout.addWidget(self.count_label)

        # Prefix input
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("文件名前缀:")
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setStyleSheet(
            ChatbotDialogStyle.get_settings_edit_style()
        )
        self.prefix_edit.setText("frame_")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_edit)

        # Sequence length input
        seq_layout = QHBoxLayout()
        seq_label = QLabel("序列长度:")
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(3, 10)
        self.seq_spin.setValue(5)
        self.seq_spin.setMinimumWidth(100)
        self.seq_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        seq_layout.addWidget(seq_label)
        seq_layout.addWidget(self.seq_spin)

        # 时间轴命名选项
        self.timestamp_naming_checkbox = QCheckBox("时间轴命名")
        self.timestamp_naming_checkbox.setChecked(False)
        self.timestamp_naming_checkbox.setToolTip("使用时间轴格式命名文件\n如: 0_01_16_266.jpg")
        self.timestamp_naming_checkbox.stateChanged.connect(self.on_timestamp_naming_changed)

        # 检测区域选择（场景变化和YOLO共用）
        region_layout = QHBoxLayout()
        self.region_btn = QPushButton("选择检测区域")
        self.region_btn.setToolTip("选择检测区域（场景变化和YOLO共用）")
        self.region_btn.clicked.connect(self.select_region)
        
        self.region_label = QLabel("全画面")
        self.region_label.setStyleSheet("color: gray;")
        
        region_layout.addWidget(self.region_btn)
        region_layout.addWidget(self.region_label)
        region_layout.addStretch()

        # 场景变化检测选项
        scene_layout = QHBoxLayout()
        self.scene_checkbox = QCheckBox("场景变化检测")
        self.scene_checkbox.setChecked(True)  # 默认开启
        self.scene_checkbox.stateChanged.connect(self.on_scene_changed)
        self.scene_checkbox.setToolTip("使用像素差异快速检测画面变化\n作为第一步粗筛")
        
        # 检测间隔
        self.detect_interval_label = QLabel("间隔:")
        self.detect_interval_spin = QSpinBox()
        self.detect_interval_spin.setRange(1, 30)
        self.detect_interval_spin.setValue(1)  # 默认1帧
        self.detect_interval_spin.setSuffix("帧")
        self.detect_interval_spin.setToolTip("每N帧检测一次场景变化\n值越大速度越快，但可能漏检")
        self.detect_interval_spin.setEnabled(True)  # 默认启用
        self.detect_interval_spin.setMinimumWidth(70)
        self.detect_interval_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        
        self.threshold_label = QLabel("阈值:")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 100)
        self.threshold_spin.setValue(70)  # 默认70
        self.threshold_spin.setToolTip("场景变化灵敏度\n值越大越敏感")
        self.threshold_spin.setEnabled(True)  # 默认启用
        self.threshold_spin.setMinimumWidth(50)
        self.threshold_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        
        scene_layout.addWidget(self.scene_checkbox)
        scene_layout.addWidget(self.detect_interval_label)
        scene_layout.addWidget(self.detect_interval_spin)
        scene_layout.addStretch()
        scene_layout.addWidget(self.threshold_label)
        scene_layout.addWidget(self.threshold_spin)

        # YOLO 检测选项 - 第一行
        yolo_layout = QHBoxLayout()
        self.yolo_checkbox = QCheckBox("YOLO检测")
        self.yolo_checkbox.setChecked(True)  # 默认开启
        self.yolo_checkbox.stateChanged.connect(self.on_yolo_changed)
        
        self.yolo_model_btn = QPushButton("选择模型")
        self.yolo_model_btn.setToolTip("选择 YOLO YAML 配置文件")
        self.yolo_model_btn.setEnabled(True)  # 默认启用
        self.yolo_model_btn.clicked.connect(self.select_yolo_model)
        
        # 模型历史记录下拉框
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setMinimumWidth(150)
        self.yolo_model_combo.setMaximumWidth(200)
        self.yolo_model_combo.setEnabled(True)  # 默认启用
        self.yolo_model_combo.addItem("-- 历史模型 --")
        self.load_model_history()
        self.yolo_model_combo.currentIndexChanged.connect(self.on_model_history_selected)
        
        self.yolo_conf_label = QLabel("置信度:")
        self.yolo_conf_spin = QSpinBox()
        self.yolo_conf_spin.setRange(1, 100)
        self.yolo_conf_spin.setValue(50)  # 默认50
        self.yolo_conf_spin.setToolTip("检测置信度阈值 (%)")
        self.yolo_conf_spin.setEnabled(True)  # 默认启用
        self.yolo_conf_spin.setMinimumWidth(60)
        self.yolo_conf_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        
        yolo_layout.addWidget(self.yolo_checkbox)
        yolo_layout.addWidget(self.yolo_model_btn)
        yolo_layout.addWidget(self.yolo_model_combo)
        yolo_layout.addStretch()
        yolo_layout.addWidget(self.yolo_conf_label)
        yolo_layout.addWidget(self.yolo_conf_spin)
        
        # YOLO 标签过滤行
        yolo_filter_layout = QHBoxLayout()
        self.yolo_classes_label = QLabel("只检测类别:")
        self.yolo_classes_edit = QLineEdit()
        self.yolo_classes_edit.setText("changfangtiao")  # 默认值
        self.yolo_classes_edit.setPlaceholderText("留空检测所有，多个用逗号分隔")
        self.yolo_classes_edit.setStyleSheet(
            ChatbotDialogStyle.get_settings_edit_style()
        )
        self.yolo_classes_edit.setEnabled(True)  # 默认启用
        self.yolo_classes_edit.setMinimumWidth(350)
        
        yolo_filter_layout.addWidget(self.yolo_classes_label)
        yolo_filter_layout.addWidget(self.yolo_classes_edit)
        
        # 提示标签
        self.detect_hint_label = QLabel("提示: 同时勾选时，需要场景变化且检测到目标才提取")
        self.detect_hint_label.setStyleSheet("color: #888; font-size: 11px;")
        
        # 快速模式
        self.fast_mode_checkbox = QCheckBox("快速模式 (仅场景变化时运行YOLO)")
        self.fast_mode_checkbox.setChecked(True)  # 默认开启
        self.fast_mode_checkbox.setToolTip("开启后只在场景变化时才运行YOLO检测\n大幅提升处理速度")
        
        # OCR去重选项 - 独立一行
        ocr_layout = QHBoxLayout()
        self.ocr_checkbox = QCheckBox("OCR去重")
        self.ocr_checkbox.setChecked(True)  # 默认开启
        self.ocr_checkbox.stateChanged.connect(self.on_ocr_changed)
        self.ocr_checkbox.setToolTip("使用OCR识别字幕文字\n相似度超过阈值的文字会被去重")
        
        # OCR过滤规则按钮
        self.ocr_filter_btn = QPushButton("过滤规则")
        self.ocr_filter_btn.setToolTip("设置OCR过滤规则\n匹配的文本将被跳过")
        self.ocr_filter_btn.setEnabled(True)  # 默认启用
        self.ocr_filter_btn.clicked.connect(self.edit_ocr_filter_rules)
        self.ocr_filter_rules = load_ocr_filter_rules()  # 从配置文件加载过滤规则
        
        self.ocr_similarity_label = QLabel("相似度:")
        self.ocr_similarity_spin = QSpinBox()
        self.ocr_similarity_spin.setRange(50, 100)
        self.ocr_similarity_spin.setValue(70)  # 默认70%
        self.ocr_similarity_spin.setSuffix("%")
        self.ocr_similarity_spin.setToolTip("文字相似度超过此值则认为是同一句\n值越大越严格，去重越少")
        self.ocr_similarity_spin.setEnabled(True)  # 默认启用
        self.ocr_similarity_spin.setMinimumWidth(60)
        self.ocr_similarity_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        
        # 优先保存长文本选项
        self.ocr_prefer_longer_checkbox = QCheckBox("优先长文本")
        self.ocr_prefer_longer_checkbox.setChecked(True)  # 默认开启
        self.ocr_prefer_longer_checkbox.setToolTip("当新文本包含旧文本时，用新文本替换旧文本\nOCR不会凭空产生不存在的文字，所以长文本更可靠")
        self.ocr_prefer_longer_checkbox.setEnabled(True)  # 默认启用
        
        ocr_layout.addWidget(self.ocr_checkbox)
        ocr_layout.addWidget(self.ocr_filter_btn)
        ocr_layout.addWidget(self.ocr_prefer_longer_checkbox)
        ocr_layout.addStretch()
        ocr_layout.addWidget(self.ocr_similarity_label)
        ocr_layout.addWidget(self.ocr_similarity_spin)
        
        # OCR路径选择行
        ocr_path_layout = QHBoxLayout()
        self.ocr_path_label = QLabel("加载模型:")
        self.ocr_path_btn = QPushButton("选择YAML")
        self.ocr_path_btn.setToolTip("选择 PP-OCRv6 的 YAML 配置文件")
        self.ocr_path_btn.setEnabled(True)
        self.ocr_path_btn.clicked.connect(self.select_ocr_path)
        
        # OCR路径历史记录下拉框
        self.ocr_path_combo = QComboBox()
        self.ocr_path_combo.setMinimumWidth(200)
        self.ocr_path_combo.setMaximumWidth(350)
        self.ocr_path_combo.setEnabled(True)
        self.ocr_path_combo.addItem("-- 历史路径 --")
        self.load_ocr_path_history()
        self.ocr_path_combo.currentIndexChanged.connect(self.on_ocr_path_history_selected)
        
        self.ocr_path = None  # 存储OCR路径
        
        ocr_path_layout.addWidget(self.ocr_path_label)
        ocr_path_layout.addWidget(self.ocr_path_btn)
        ocr_path_layout.addWidget(self.ocr_path_combo)
        ocr_path_layout.addStretch()
        
        # 调试模式
        self.debug_checkbox = QCheckBox("调试模式 (生成YOLO检测图和场景检测图)")
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.setToolTip("开启后会生成额外的调试文件夹:\n- YOLO检测结果图片和TXT\n- 场景检测图片")
        
        self.yolo_model_path = None  # 存储模型路径

        # ===== 图片提取区 =====
        self.extract_group = QGroupBox("图片提取")
        self.extract_group.setCheckable(True)
        self.extract_group.setChecked(False)  # 默认禁用
        self.extract_group.toggled.connect(self.on_extract_group_toggled)
        extract_layout = QVBoxLayout()
        extract_layout.addLayout(prefix_layout)
        extract_layout.addLayout(seq_layout)
        extract_layout.addLayout(interval_layout)
        extract_layout.addWidget(self.timestamp_naming_checkbox)
        self.extract_group.setLayout(extract_layout)
        layout.addWidget(self.extract_group)

        # ===== 硬字幕提取功能区 =====
        self.subtitle_group = QGroupBox("硬字幕提取")
        self.subtitle_group.setCheckable(True)
        self.subtitle_group.setChecked(True)  # 默认启用
        self.subtitle_group.toggled.connect(self.on_subtitle_group_toggled)
        subtitle_layout = QVBoxLayout()
        subtitle_layout.addLayout(region_layout)
        subtitle_layout.addLayout(scene_layout)
        subtitle_layout.addLayout(yolo_layout)
        subtitle_layout.addLayout(yolo_filter_layout)
        subtitle_layout.addLayout(ocr_layout)
        subtitle_layout.addLayout(ocr_path_layout)
        subtitle_layout.addWidget(self.detect_hint_label)
        subtitle_layout.addWidget(self.fast_mode_checkbox)
        subtitle_layout.addWidget(self.debug_checkbox)
        self.subtitle_group.setLayout(subtitle_layout)
        layout.addWidget(self.subtitle_group)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.setStyleSheet(get_ok_btn_style())
        cancel_button = QPushButton("取消")
        cancel_button.setStyleSheet(get_cancel_btn_style())

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(580, 320)
        
        # 初始化预计数量显示
        self.update_estimated_count()

    def on_unit_changed(self, index):
        """单位改变时更新预计数量"""
        self.update_estimated_count()

    def on_scene_changed(self, state):
        """场景检测选项改变时"""
        is_scene = state == Qt.Checked
        self.threshold_spin.setEnabled(is_scene)
        self.detect_interval_spin.setEnabled(is_scene)
        self.update_estimated_count()

    def on_yolo_changed(self, state):
        """YOLO检测选项改变时"""
        is_yolo = state == Qt.Checked
        self.yolo_model_btn.setEnabled(is_yolo)
        self.yolo_model_combo.setEnabled(is_yolo)
        self.yolo_conf_spin.setEnabled(is_yolo)
        self.yolo_classes_edit.setEnabled(is_yolo)
        self.update_estimated_count()
    
    def on_ocr_changed(self, state):
        """OCR去重选项改变时"""
        is_ocr = state == Qt.Checked
        self.ocr_similarity_spin.setEnabled(is_ocr)
        self.ocr_filter_btn.setEnabled(is_ocr)
        self.ocr_prefer_longer_checkbox.setEnabled(is_ocr)
        self.ocr_path_btn.setEnabled(is_ocr)
        self.ocr_path_combo.setEnabled(is_ocr)
    
    def on_extract_group_toggled(self, checked):
        """图片提取区域启用/禁用（与硬字幕提取互斥）"""
        if checked:
            # 启用图片提取时，禁用硬字幕提取
            self.subtitle_group.setChecked(False)
        self.prefix_edit.setEnabled(checked and not self.timestamp_naming_checkbox.isChecked())
        self.seq_spin.setEnabled(checked and not self.timestamp_naming_checkbox.isChecked())
        self.interval_spin.setEnabled(checked)
        self.unit_combo.setEnabled(checked)
        self.timestamp_naming_checkbox.setEnabled(checked)
        self.update_estimated_count()
    
    def on_timestamp_naming_changed(self, state):
        """时间轴命名选项改变时"""
        is_timestamp = state == Qt.Checked
        # 时间轴命名时禁用前缀和序列长度
        self.prefix_edit.setEnabled(not is_timestamp and self.extract_group.isChecked())
        self.seq_spin.setEnabled(not is_timestamp and self.extract_group.isChecked())
    
    def on_subtitle_group_toggled(self, checked):
        """硬字幕提取区域启用/禁用（与图片提取互斥）"""
        if checked:
            # 启用硬字幕提取时，禁用图片提取
            self.extract_group.setChecked(False)
        self.region_btn.setEnabled(checked)
        # 场景检测相关
        self.scene_checkbox.setEnabled(checked)
        if checked and self.scene_checkbox.isChecked():
            self.threshold_spin.setEnabled(True)
            self.detect_interval_spin.setEnabled(True)
        else:
            self.threshold_spin.setEnabled(False)
            self.detect_interval_spin.setEnabled(False)
        # YOLO相关
        self.yolo_checkbox.setEnabled(checked)
        if checked and self.yolo_checkbox.isChecked():
            self.yolo_model_btn.setEnabled(True)
            self.yolo_model_combo.setEnabled(True)
            self.yolo_conf_spin.setEnabled(True)
            self.yolo_classes_edit.setEnabled(True)
        else:
            self.yolo_model_btn.setEnabled(False)
            self.yolo_model_combo.setEnabled(False)
            self.yolo_conf_spin.setEnabled(False)
            self.yolo_classes_edit.setEnabled(False)
        # OCR相关
        self.ocr_checkbox.setEnabled(checked)
        if checked and self.ocr_checkbox.isChecked():
            self.ocr_similarity_spin.setEnabled(True)
            self.ocr_filter_btn.setEnabled(True)
            self.ocr_prefer_longer_checkbox.setEnabled(True)
            self.ocr_path_btn.setEnabled(True)
            self.ocr_path_combo.setEnabled(True)
        else:
            self.ocr_similarity_spin.setEnabled(False)
            self.ocr_filter_btn.setEnabled(False)
            self.ocr_prefer_longer_checkbox.setEnabled(False)
            self.ocr_path_btn.setEnabled(False)
            self.ocr_path_combo.setEnabled(False)
        # 其他
        self.fast_mode_checkbox.setEnabled(checked)
        self.debug_checkbox.setEnabled(checked)
        self.update_estimated_count()
    
    def edit_ocr_filter_rules(self):
        """编辑OCR过滤规则"""
        dialog = OcrFilterDialog(self, self.ocr_filter_rules)
        if dialog.exec_() == QDialog.Accepted:
            self.ocr_filter_rules = dialog.get_rules()
    
    def load_ocr_path_history(self):
        """加载OCR路径历史记录"""
        history = load_ocr_path_history()
        for path in history:
            # 在文件夹里找 YAML 文件
            yaml_in_folder = None
            if osp.isdir(path):
                for f in os.listdir(path):
                    if f.endswith('.yaml') or f.endswith('.yml'):
                        yaml_in_folder = osp.join(path, f)
                        break
            if yaml_in_folder:
                self.ocr_path_combo.addItem(osp.basename(yaml_in_folder), yaml_in_folder)
            elif osp.isfile(path) and (path.endswith('.yaml') or path.endswith('.yml')):
                self.ocr_path_combo.addItem(osp.basename(path), path)
        if history:
            self.ocr_path = history[0] if osp.isfile(history[0]) else None
    
    def select_ocr_path(self):
        """选择OCR配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 PP-OCRv6 YAML 配置文件",
            "",
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if file_path:
            self.ocr_path = file_path
            self.ocr_path_combo.setItemText(0, osp.basename(file_path))
            self.ocr_path_combo.setCurrentIndex(0)
            save_ocr_path_history(file_path)
            while self.ocr_path_combo.count() > 1:
                self.ocr_path_combo.removeItem(1)
            self.load_ocr_path_history()
    
    def on_ocr_path_history_selected(self, index):
        """从历史记录选择OCR路径"""
        if index <= 0:
            return
        ocr_path = self.ocr_path_combo.itemData(index)
        if ocr_path and osp.exists(ocr_path):
            self.ocr_path = ocr_path
            self.ocr_path_combo.setItemText(0, osp.basename(ocr_path))
            self.ocr_path_combo.setCurrentIndex(0)
    
    def load_model_history(self):
        """加载模型历史记录"""
        import json
        self.model_history_file = osp.join(osp.expanduser("~"), ".yolo_model_history.json")
        try:
            if osp.exists(self.model_history_file):
                with open(self.model_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    for path in history:
                        if osp.exists(path):
                            self.yolo_model_combo.addItem(osp.basename(path), path)
        except Exception as e:
            logger.warning(f"Failed to load model history: {e}")
    
    def save_model_history(self, model_path):
        """保存模型到历史记录"""
        import json
        try:
            history = []
            if osp.exists(self.model_history_file):
                with open(self.model_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 移除重复项，添加到最前面
            if model_path in history:
                history.remove(model_path)
            history.insert(0, model_path)
            
            # 只保留最近10个
            history = history[:10]
            
            with open(self.model_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model history: {e}")
    
    def on_model_history_selected(self, index):
        """从历史记录选择模型"""
        if index <= 0:
            return
        model_path = self.yolo_model_combo.itemData(index)
        if model_path and osp.exists(model_path):
            self.yolo_model_path = model_path
            self.yolo_model_combo.setItemText(0, osp.basename(model_path))
            self.yolo_model_combo.setCurrentIndex(0)

    def select_yolo_model(self):
        """选择YOLO模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO YAML 配置文件",
            "",
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if file_path:
            self.yolo_model_path = file_path
            # 更新下拉框显示
            filename = osp.basename(file_path)
            self.yolo_model_combo.setItemText(0, filename)
            self.yolo_model_combo.setCurrentIndex(0)
            # 保存到历史记录
            self.save_model_history(file_path)
            # 刷新下拉框历史
            while self.yolo_model_combo.count() > 1:
                self.yolo_model_combo.removeItem(1)
            self.load_model_history()

    def select_region(self):
        """打开区域选择对话框"""
        dialog = RegionSelectDialog(
            self, 
            self.first_frame, 
            self.video_capture, 
            self.total_frames, 
            self.fps,
            self.detection_region  # 传递当前已设置的区域
        )
        if dialog.exec_():
            region = dialog.get_region()
            if region:
                x, y, w, h = region
                self.detection_region = region
                self.region_label.setText(f"({x},{y}) {w}x{h}")
                self.region_label.setStyleSheet("color: green;")
            else:
                self.detection_region = None
                self.region_label.setText("全画面")
                self.region_label.setStyleSheet("color: gray;")
        
        # 重置视频位置到开头
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_estimated_count(self):
        """更新预计提取数量"""
        # 硬字幕提取模式无法预估数量
        if self.subtitle_group.isChecked():
            self.count_label.setText("? 张")
            return
            
        # 图片提取模式，计算预计数量
        interval_value = self.interval_spin.value()
        unit_index = self.unit_combo.currentIndex()
        
        # 根据单位计算帧间隔
        if unit_index == 0:  # 帧
            frame_interval = interval_value
        elif unit_index == 1:  # 秒
            frame_interval = int(interval_value * self.fps)
        else:  # 分钟
            frame_interval = int(interval_value * 60 * self.fps)
        
        # 计算预计提取数量
        if frame_interval > 0:
            estimated_count = (self.total_frames + frame_interval - 1) // frame_interval
        else:
            estimated_count = self.total_frames
        
        self.count_label.setText(f"{estimated_count} 张")

    def get_frame_interval(self):
        """获取实际的帧间隔"""
        interval_value = self.interval_spin.value()
        unit_index = self.unit_combo.currentIndex()
        
        if unit_index == 0:  # 帧
            return interval_value
        elif unit_index == 1:  # 秒
            return max(1, int(interval_value * self.fps))
        else:  # 分钟
            return max(1, int(interval_value * 60 * self.fps))

    def get_values(self):
        # 解析标签过滤列表
        classes_text = self.yolo_classes_edit.text().strip()
        if classes_text:
            yolo_classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        else:
            yolo_classes = None  # None 表示检测所有类别
        
        # 判断是否使用硬字幕模式（硬字幕提取启用且场景检测或YOLO检测开启时）
        subtitle_enabled = self.subtitle_group.isChecked()
        use_subtitle_mode = subtitle_enabled and (self.scene_checkbox.isChecked() or self.yolo_checkbox.isChecked())
        
        return (
            self.get_frame_interval(),  # 返回实际的帧间隔
            self.prefix_edit.text(),
            self.seq_spin.value(),
            use_subtitle_mode,  # 硬字幕模式时使用时间轴格式
            subtitle_enabled and self.scene_checkbox.isChecked(),  # 场景检测
            self.threshold_spin.value(),  # 场景检测阈值
            self.detect_interval_spin.value(),  # 检测间隔（每N帧检测一次）
            self.detection_region,  # 检测区域 (x, y, w, h) 或 None
            subtitle_enabled and self.yolo_checkbox.isChecked(),  # YOLO检测
            self.yolo_model_path,  # YOLO模型路径
            self.yolo_conf_spin.value() / 100.0,  # YOLO置信度 (0-1)
            yolo_classes,  # YOLO类别过滤列表
            self.debug_checkbox.isChecked(),  # 调试模式
            self.fast_mode_checkbox.isChecked(),  # 快速模式
            subtitle_enabled and self.ocr_checkbox.isChecked(),  # OCR去重
            self.ocr_similarity_spin.value() / 100.0,  # OCR相似度阈值 (0-1)
            self.ocr_filter_rules,  # OCR过滤规则列表（正则表达式）
            self.ocr_prefer_longer_checkbox.isChecked(),  # 优先保存长文本
            self.extract_group.isChecked(),  # 图片提取启用
            subtitle_enabled,  # 硬字幕提取启用
            self.timestamp_naming_checkbox.isChecked(),  # 时间轴命名
            self.ocr_path,  # OCR路径
        )


def extract_frames_from_video(self, input_file, out_dir):
    temp_video_path = None
    video_capture = None
    opened_successfully = False
    ffmpeg_path = None

    try:
        input_file_str = str(input_file)

        # Load video directly
        video_capture = cv2.VideoCapture(input_file_str)
        if video_capture.isOpened():
            opened_successfully = True
        else:
            video_capture.release()
            logger.warning(
                f"Loading video failed. Trying temporary file workaround."
            )

            try:
                with open(input_file, "rb") as f:
                    video_data = f.read()
                _, ext = osp.splitext(input_file)
                suffix = ext if ext else ".mp4"
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                )
                temp_video_path = temp_file.name
                temp_file.write(video_data)
                temp_file.close()
                logger.debug(
                    f"Writing video data to temporary file: {temp_video_path}"
                )

                video_capture = cv2.VideoCapture(temp_video_path)
                if video_capture.isOpened():
                    opened_successfully = True
                else:
                    video_capture.release()
                    logger.error(
                        f"Failed to open video via temporary file: {temp_video_path}"
                    )
            except Exception as e:
                logger.error(f"Error during temporary file workaround: {e}")
                if video_capture:
                    video_capture.release()

        if not opened_successfully:
            popup = Popup(
                f"Failed to open video file: {osp.basename(input_file)}",
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return None

        # --- Proceed with frame extraction settings ---
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # Handle cases where fps might be 0 or invalid
        if not fps or fps <= 0:
            logger.warning(
                f"Invalid or zero FPS ({fps}) detected for video. Defaulting FPS to 30 for calculations."
            )
            fps = 30.0  # Assign a default FPS
        logger.info(
            f"Video opened: Total Frames ~{total_frames}, FPS ~{fps:.2f}"
        )

        # 读取第一帧用于区域选择
        ret, first_frame = video_capture.read()
        if ret:
            # 重置到开头
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            first_frame = None

        # 处理UI事件，让拖放操作完成，释放资源管理器
        QApplication.processEvents()
        
        dialog = FrameExtractionDialog(self, total_frames, fps, first_frame, video_capture)
        if not dialog.exec_():
            logger.info(
                "Frame extraction cancelled by user in settings dialog."
            )
            # video_capture is released in the outer finally block
            return None

        interval, prefix, seq_len, use_timestamp, use_scene_detect, scene_threshold, detect_interval, detection_region, use_yolo, yolo_model_path, yolo_conf, yolo_classes, debug_mode, fast_mode, use_ocr, ocr_similarity, ocr_filter_list, ocr_prefer_longer, extract_enabled, subtitle_enabled, extract_timestamp_naming, ocr_path = dialog.get_values()
        
        # 检查是否至少启用了一个功能
        if not extract_enabled and not subtitle_enabled:
            popup = Popup(
                "请至少启用一个功能（图片提取或硬字幕提取）",
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return None
        
        # 验证YOLO设置
        yolo_model = None
        yolo_class_ids = []
        if use_yolo:
            if not yolo_model_path:
                popup = Popup(
                    "请先选择YOLO YAML配置文件",
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, position="center")
                return None
            raw_model = _load_model_from_yaml(yolo_model_path)
            if raw_model is None:
                popup = Popup(
                    f"加载YOLO模型失败",
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, position="center")
                return None
            yolo_model = _OnnxYoloWrapper(raw_model)
            classes = raw_model.config.get("classes", [])
            if isinstance(classes, dict):
                classes = list(classes.values())
            if classes:
                name_to_id = {str(v).lower(): k for k, v in enumerate(classes)}
                if yolo_classes:
                    for cls_name in yolo_classes:
                        cls_lower = cls_name.lower()
                        if cls_lower in name_to_id:
                            yolo_class_ids.append(name_to_id[cls_lower])
                logger.info(f"YOLO ({raw_model.config.get('type','')}) loaded, classes: {classes[:5]}...")
        
        # 获取OCR模型（如果启用了OCR去重）
        ocr_model = None
        if use_ocr:
            if not ocr_path:
                popup = Popup(
                    "请先选择OCR YAML配置文件",
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, position="center")
                return None
            raw_ocr = _load_model_from_yaml(ocr_path)
            if raw_ocr is None or not hasattr(raw_ocr, 'text_system'):
                popup = Popup(
                    "加载OCR模型失败",
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, position="center")
                return None
            ocr_model = PPOCRv6Wrapper(raw_ocr.text_system)
        
        # 创建输出文件夹
        logger.info(f"Creating output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        if osp.exists(out_dir):
            logger.info(f"Output directory created successfully: {out_dir}")
        else:
            logger.error(f"Failed to create output directory: {out_dir}")
        
        # 处理UI事件，让拖放操作完成
        QApplication.processEvents()

        # 统一使用 OpenCV 提取（可以边提取边保存，取消时已提取的保留）
        # Inner try: Handle the actual extraction
        try:
            logger.info(f"Using OpenCV for extraction. Scene detect: {use_scene_detect}, YOLO: {use_yolo}, Region: {detection_region}")
            # --- OpenCV Path ---
            if use_scene_detect or use_yolo:
                # 场景检测或YOLO检测模式，无法预估帧数
                estimated_frames = 0
                if use_yolo:
                    progress_dialog = QProgressDialog(
                        "正在进行YOLO检测...",
                        "取消",
                        0,
                        total_frames,  # 用总帧数作为进度
                        self,
                    )
                else:
                    progress_dialog = QProgressDialog(
                        "正在检测场景变化...",
                        "取消",
                        0,
                        total_frames,  # 用总帧数作为进度
                        self,
                    )
            else:
                estimated_frames = (
                    (total_frames + interval - 1) // interval
                    if total_frames > 0 and interval > 0
                    else 0
                )
                progress_dialog = QProgressDialog(
                    f"正在提取帧... 0/{estimated_frames}",
                    "取消",
                    0,
                    estimated_frames,
                self,
            )
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setWindowTitle("进度")
            progress_dialog.setMinimumWidth(400)
            progress_dialog.setMinimumHeight(150)
            progress_dialog.setStyleSheet(
                get_progress_dialog_style(color="#1d1d1f", height=20)
            )
            progress_dialog.setValue(0)
            progress_dialog.show()

            import time
            start_time = time.time()  # 记录开始时间
            
            frame_count = 0
            saved_frame_count = 1  # 从1开始计数
            extraction_cancelled = False
            prev_frame_gray = None  # 用于场景检测（存储上一帧的灰度图）
            prev_ocr_text = None  # 用于OCR去重（存储上一次识别的文字）
            recent_saved_texts = []  # 最近保存到SRT的文字列表（用于去重，最多5条）
            last_save_frame = -9999  # 上次保存的帧号
            min_save_interval = 0  # OCR模式下精确到帧
            stable_frame = None  # 稳定帧
            stable_frame_count = 0
            frame_to_save = None  # 要保存的帧
            stable_start_frame = 0  # 稳定帧开始的帧号（用于VideoSubFinder格式）
            
            # 延迟保存相关变量（用于正确计算结束时间）
            pending_save_frame = None  # 待保存的帧
            pending_start_frame = 0  # 待保存帧的开始时间（帧号）
            last_subtitle_end_frame = 0  # 上一句字幕消失的帧号
            subtitle_visible = False  # 当前是否有字幕显示
            
            # SRT字幕生成相关
            srt_entries = []  # 存储SRT条目 [(index, start_time, end_time, text), ...]
            pending_ocr_text = None  # 待保存的OCR文本
            
            while True:
                if progress_dialog.wasCanceled():
                    logger.info(
                        "Frame extraction cancelled by user (OpenCV)."
                    )
                    extraction_cancelled = True
                    break

                if not video_capture.isOpened():
                    logger.warning(
                        "Video capture became unopened during OpenCV processing."
                    )
                    break

                ret, frame = video_capture.read()
                if not ret:
                    break

                should_save = False
                scene_changed = False
                yolo_detected = False
                diff_ratio = 0.0
                current_ocr_text = None
                
                # 获取检测区域帧
                if detection_region:
                    rx, ry, rw, rh = detection_region
                    h, w = frame.shape[:2]
                    rx = max(0, min(rx, w - 1))
                    ry = max(0, min(ry, h - 1))
                    rw = min(rw, w - rx)
                    rh = min(rh, h - ry)
                    region_frame = frame[ry:ry+rh, rx:rx+rw]
                else:
                    region_frame = frame
                
                # 第一步：场景变化检测（像素差异快速筛选）
                if use_scene_detect:
                    # 检测间隔：每N帧检测一次
                    if frame_count % detect_interval == 0:
                        gray = cv2.cvtColor(region_frame, cv2.COLOR_BGR2GRAY)
                        small = cv2.resize(gray, (128, 64))
                        
                        if prev_frame_gray is None:
                            prev_frame_gray = small
                            stable_start_frame = frame_count
                            scene_changed = True  # 第一帧算变化
                        else:
                            diff = cv2.absdiff(prev_frame_gray, small)
                            diff_ratio = float(np.mean(diff)) / 255.0
                            
                            # 阈值映射：阈值越大越敏感
                            # 阈值1 -> 需要10%差异（不敏感）
                            # 阈值30 -> 需要3%差异
                            # 阈值50 -> 需要1.5%差异
                            # 阈值100 -> 需要0.1%差异（非常敏感）
                            required_diff = 0.10 - (scene_threshold / 100.0) * 0.099
                            
                            if diff_ratio > required_diff:
                                scene_changed = True
                                prev_frame_gray = small
                else:
                    scene_changed = True
                
                # 第二步：YOLO检测（使用选择的区域）
                yolo_results = None
                yolo_actually_ran = False  # 标记YOLO是否实际运行了
                if use_yolo and yolo_model is not None:
                    # 快速模式：只在场景变化时才运行YOLO
                    # 但如果字幕正在显示，需要定期检测字幕是否消失
                    should_run_yolo = True
                    if fast_mode and use_scene_detect:
                        should_run_yolo = scene_changed
                        # 字幕显示中时，每10帧检测一次是否消失
                        if subtitle_visible and frame_count % 10 == 0:
                            should_run_yolo = True
                    
                    if should_run_yolo:
                        yolo_actually_ran = True
                        # 使用选择的区域进行YOLO检测
                        if yolo_class_ids:
                            yolo_results = yolo_model(region_frame, conf=yolo_conf, classes=yolo_class_ids, verbose=False)
                        else:
                            yolo_results = yolo_model(region_frame, conf=yolo_conf, verbose=False)
                        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                            yolo_detected = True
                
                # 第三步：OCR去重（只在需要保存时才运行OCR）
                ocr_skip = False
                ocr_filtered = False  # 标记是否被过滤规则跳过（需要留空档）
                if use_ocr and ocr_model is not None:
                    # 先判断前面的条件是否满足
                    preliminary_save = False
                    if use_yolo and use_scene_detect:
                        preliminary_save = scene_changed and yolo_detected
                    elif use_yolo:
                        preliminary_save = yolo_detected
                    elif use_scene_detect:
                        preliminary_save = scene_changed
                    
                    if preliminary_save:
                        # 运行OCR识别
                        try:
                            ocr_result = ocr_model.ocr(region_frame)
                            current_ocr_text = ""
                            if ocr_result and ocr_result[0]:
                                texts = [line[1][0] if isinstance(line[1], tuple) else line[1] for line in ocr_result[0]]
                                current_ocr_text = " ".join(texts)
                            
                            # 去除空格后的文字
                            current_ocr_clean = current_ocr_text.replace(" ", "").replace("　", "").strip()
                            
                            # OCR结果为空或只有空格 → 跳过
                            if not current_ocr_clean:
                                ocr_skip = True
                                ocr_filtered = True  # 标记为过滤跳过（需要断点）
                                prev_ocr_text = ""  # 更新记录，作为断点
                                logger.info(f"Frame {frame_count}: OCR SKIP (empty text)")
                            
                            # 过滤只有省略号的文本（如「......」、「...」、「…」、「。。。」等）
                            # 注意：只过滤纯省略号，不过滤包含其他标点的文本
                            if not ocr_skip:
                                import re
                                # 移除括号和空格后，检查是否只剩下省略号类字符
                                # 括号：「」『』【】〖〗《》〈〉（）()[]{}
                                bracket_pattern = r'[「」『』【】〖〗《》〈〉（）\(\)\[\]{}\s]'
                                text_without_brackets = re.sub(bracket_pattern, '', current_ocr_clean)
                                # 省略号类字符：…、...、。。。、・・・、···等
                                ellipsis_pattern = r'^[\.。…·\・\･]+$'
                                if text_without_brackets and re.match(ellipsis_pattern, text_without_brackets):
                                    ocr_skip = True
                                    ocr_filtered = True  # 标记为过滤跳过（需要断点）
                                    prev_ocr_text = current_ocr_text  # 更新记录，作为断点
                                    logger.info(f"Frame {frame_count}: OCR SKIP (ellipsis only: {current_ocr_text})")
                            
                            # 用户自定义过滤规则（正则表达式）
                            if not ocr_skip and ocr_filter_list:
                                for filter_pattern in ocr_filter_list:
                                    try:
                                        if re.match(filter_pattern, current_ocr_clean):
                                            ocr_skip = True
                                            ocr_filtered = True  # 标记为过滤跳过（需要断点）
                                            prev_ocr_text = current_ocr_text  # 更新记录，作为断点
                                            logger.info(f"Frame {frame_count}: OCR SKIP (regex match: '{current_ocr_text}' ~ '{filter_pattern}')")
                                            break
                                    except re.error:
                                        pass  # 忽略无效的正则表达式
                            
                            # 相似度去重
                            if not ocr_skip:
                                # 计算文字相似度（忽略空格，使用字符集合比较）
                                def text_similarity(s1, s2):
                                    if not s1 and not s2:
                                        return 1.0
                                    if not s1 or not s2:
                                        return 0.0
                                    
                                    # 日文小写假名转大写假名的映射
                                    small_to_large = str.maketrans(
                                        'ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ',
                                        'あいうえおつやゆよわアイウエオツヤユヨワ'
                                    )
                                    
                                    # 去除空格并统一假名大小写
                                    s1_clean = s1.replace(" ", "").replace("　", "").translate(small_to_large)
                                    s2_clean = s2.replace(" ", "").replace("　", "").translate(small_to_large)
                                    if not s1_clean and not s2_clean:
                                        return 1.0
                                    if not s1_clean or not s2_clean:
                                        return 0.0
                                    
                                    # 如果一个是另一个的子串，认为是相同的
                                    if s1_clean in s2_clean or s2_clean in s1_clean:
                                        return 1.0
                                    
                                    # 计算相同字符数（不考虑位置）
                                    from collections import Counter
                                    c1 = Counter(s1_clean)
                                    c2 = Counter(s2_clean)
                                    common = sum((c1 & c2).values())
                                    longer = max(len(s1_clean), len(s2_clean))
                                    return common / longer
                                
                                # 检查是否是子串关系（用于优先保存长文本）
                                def is_substring_relation(s1, s2):
                                    """检查s1和s2是否有子串关系，返回 (是否子串, 较长的文本)"""
                                    if not s1 or not s2:
                                        return False, None
                                    
                                    # 日文小写假名转大写假名的映射
                                    small_to_large = str.maketrans(
                                        'ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ',
                                        'あいうえおつやゆよわアイウエオツヤユヨワ'
                                    )
                                    
                                    s1_clean = s1.replace(" ", "").replace("　", "").translate(small_to_large)
                                    s2_clean = s2.replace(" ", "").replace("　", "").translate(small_to_large)
                                    
                                    if s1_clean in s2_clean:
                                        return True, s2  # s2更长
                                    if s2_clean in s1_clean:
                                        return True, s1  # s1更长
                                    return False, None
                                
                                # 比较两个文本的长度（去除空格后）
                                def get_clean_length(s):
                                    if not s:
                                        return 0
                                    return len(s.replace(" ", "").replace("　", ""))
                                
                                # 和上一次识别的文字比较
                                similarity = text_similarity(current_ocr_text, prev_ocr_text) if prev_ocr_text else 0.0
                                # 也和最近保存的5条文字比较（防止漏检后重复保存）
                                max_similarity = similarity
                                matched_saved_idx = -1  # 匹配到的已保存文本索引
                                matched_text = prev_ocr_text  # 匹配到的文本
                                for idx, saved_text in enumerate(recent_saved_texts):
                                    sim = text_similarity(current_ocr_text, saved_text)
                                    if sim > max_similarity:
                                        max_similarity = sim
                                        matched_saved_idx = idx
                                        matched_text = saved_text
                                
                                # 相似度超过阈值认为是同一句
                                if max_similarity >= ocr_similarity:
                                    # 检查是否启用了优先保存长文本
                                    if ocr_prefer_longer:
                                        current_len = get_clean_length(current_ocr_text)
                                        matched_len = get_clean_length(matched_text)
                                        
                                        # 当前文本更长，替换之前的文本
                                        if current_len > matched_len:
                                            if matched_saved_idx >= 0:
                                                old_text = recent_saved_texts[matched_saved_idx]
                                                recent_saved_texts[matched_saved_idx] = current_ocr_text
                                                # 同时更新SRT条目中的文本
                                                for i, (idx, start, end, text) in enumerate(srt_entries):
                                                    if text == old_text:
                                                        srt_entries[i] = (idx, start, end, current_ocr_text)
                                                        logger.info(f"Frame {frame_count}: OCR REPLACE (longer: {matched_len}->{current_len}) | '{old_text[:30]}...' -> '{current_ocr_text[:30]}...'")
                                                        break
                                            else:
                                                # 匹配的是prev_ocr_text，更新最后一条SRT条目
                                                if srt_entries:
                                                    last_idx, last_start, last_end, last_text = srt_entries[-1]
                                                    if text_similarity(last_text, prev_ocr_text) >= ocr_similarity:
                                                        srt_entries[-1] = (last_idx, last_start, last_end, current_ocr_text)
                                                        logger.info(f"Frame {frame_count}: OCR REPLACE last (longer: {matched_len}->{current_len}) | '{last_text[:30]}...' -> '{current_ocr_text[:30]}...'")
                                            prev_ocr_text = current_ocr_text
                                            ocr_skip = True
                                            logger.info(f"Frame {frame_count}: OCR SKIP after replace (similarity={max_similarity:.2f})")
                                        else:
                                            # 当前文本更短或相同，跳过
                                            ocr_skip = True
                                            logger.info(f"Frame {frame_count}: OCR SKIP (shorter/equal: {current_len}<={matched_len}, similarity={max_similarity:.2f})")
                                    else:
                                        ocr_skip = True
                                        logger.info(f"Frame {frame_count}: OCR SKIP (similarity={max_similarity:.2f} >= {ocr_similarity:.2f})")
                                else:
                                    # 文字变化，更新记录
                                    prev_ocr_text = current_ocr_text
                                    stable_start_frame = frame_count
                                    logger.info(f"Frame {frame_count}: OCR SAVE (similarity={max_similarity:.2f} < {ocr_similarity:.2f}) | {current_ocr_text}")
                        except Exception as e:
                            logger.warning(f"OCR error at frame {frame_count}: {e}")
                
                # 每10帧处理一次UI事件，避免完全阻塞
                if frame_count % 10 == 0:
                    QApplication.processEvents()
                
                # 检测字幕消失（用于精确结束时间）
                # 只有在YOLO实际运行了且没检测到字幕时，才认为字幕消失
                subtitle_disappeared = False
                if use_yolo and yolo_actually_ran and subtitle_visible and not yolo_detected:
                    # 之前有字幕，现在YOLO检测不到了 → 字幕消失
                    subtitle_disappeared = True
                    last_subtitle_end_frame = frame_count - 1
                    subtitle_visible = False
                    logger.info(f"Frame {frame_count}: Subtitle disappeared")
                
                # 更新字幕可见状态
                if yolo_detected:
                    subtitle_visible = True
                
                # 判断是否保存
                if use_yolo and use_scene_detect:
                    # 两者都启用：需要场景变化 AND YOLO检测到目标
                    should_save = scene_changed and yolo_detected
                elif use_yolo:
                    # 只启用YOLO
                    should_save = yolo_detected
                elif use_scene_detect:
                    # 只启用场景检测
                    should_save = scene_changed
                else:
                    # 普通间隔模式
                    if frame_count % interval == 0:
                        should_save = True
                
                # OCR去重：如果文字相同则不保存
                if ocr_skip:
                    should_save = False
                    # 如果是被过滤的内容（不是相似度去重），需要结束上一句字幕并留出空档
                    if ocr_filtered and pending_save_frame is not None and use_timestamp:
                        # 检测到被过滤的字幕（如省略号），先保存上一句
                        # 结束时间是被过滤字幕出现之前
                        end_frame = frame_count - 1
                        duration_ms = int((end_frame - pending_start_frame) * 1000 / fps)
                        
                        if duration_ms >= 300:
                            start_ms = int(pending_start_frame * 1000 / fps)
                            start_hours = start_ms // 3600000
                            start_minutes = (start_ms % 3600000) // 60000
                            start_seconds = (start_ms % 60000) // 1000
                            start_milliseconds = start_ms % 1000
                            
                            end_ms = int(end_frame * 1000 / fps)
                            end_hours = end_ms // 3600000
                            end_minutes = (end_ms % 3600000) // 60000
                            end_seconds = (end_ms % 60000) // 1000
                            end_milliseconds = end_ms % 1000
                            
                            # 获取图像尺寸信息
                            h, w = pending_save_frame.shape[:2]
                            size_info = f"000000{0:04d}{0:04d}{w:04d}{h:04d}"
                            
                            # 组合VideoSubFinder格式文件名
                            timestamp_str = f"{start_hours}_{start_minutes:02d}_{start_seconds:02d}_{start_milliseconds:03d}__{end_hours}_{end_minutes:02d}_{end_seconds:02d}_{end_milliseconds:03d}_{size_info}"
                            frame_filename = osp.join(out_dir, f"{timestamp_str}.jpeg")
                            
                            try:
                                cv2.imwrite(frame_filename, pending_save_frame)
                                saved_frame_count += 1
                                logger.info(f"Saved (before filtered): {timestamp_str}")
                                # 添加SRT条目（只有持续时间>=300ms才添加）
                                if use_ocr and pending_ocr_text and (end_ms - start_ms) >= 300:
                                    srt_entries.append((len(srt_entries) + 1, start_ms, end_ms, pending_ocr_text))
                                    # 更新最近保存的文字列表（最多5条）
                                    recent_saved_texts.append(pending_ocr_text)
                                    if len(recent_saved_texts) > 5:
                                        recent_saved_texts.pop(0)
                            except Exception as e:
                                logger.error(f"Error writing frame {frame_filename}: {e}")
                        
                        pending_save_frame = None
                        pending_ocr_text = None
                    
                    # 注意：不在这里更新 stable_start_frame
                    # 让下一次 OCR SAVE 时自动更新，这样新字幕的开始时间才是正确的
                
                # 字幕消失时，立即保存待保存的帧（使用消失时刻作为结束时间）
                if subtitle_disappeared and pending_save_frame is not None and use_timestamp:
                    # 计算时长（毫秒）
                    duration_ms = int((last_subtitle_end_frame - pending_start_frame) * 1000 / fps)
                    
                    # 最小时长过滤：< 300ms 的不保存（可能是误检或字幕淡出）
                    if duration_ms < 300:
                        logger.info(f"Frame {frame_count}: SKIP (duration={duration_ms}ms < 300ms)")
                        pending_save_frame = None
                        stable_start_frame = frame_count
                    else:
                        # 计算开始时间
                        start_ms = int(pending_start_frame * 1000 / fps)
                        start_hours = start_ms // 3600000
                        start_minutes = (start_ms % 3600000) // 60000
                        start_seconds = (start_ms % 60000) // 1000
                        start_milliseconds = start_ms % 1000
                        
                        # 计算结束时间（字幕消失的那一帧）
                        end_ms = int(last_subtitle_end_frame * 1000 / fps)
                        end_hours = end_ms // 3600000
                        end_minutes = (end_ms % 3600000) // 60000
                        end_seconds = (end_ms % 60000) // 1000
                        end_milliseconds = end_ms % 1000
                        
                        # 获取图像尺寸信息
                        h, w = pending_save_frame.shape[:2]
                        size_info = f"000000{0:04d}{0:04d}{w:04d}{h:04d}"
                        
                        # 组合VideoSubFinder格式文件名
                        timestamp_str = f"{start_hours}_{start_minutes:02d}_{start_seconds:02d}_{start_milliseconds:03d}__{end_hours}_{end_minutes:02d}_{end_seconds:02d}_{end_milliseconds:03d}_{size_info}"
                        frame_filename = osp.join(out_dir, f"{timestamp_str}.jpeg")
                        
                        try:
                            cv2.imwrite(frame_filename, pending_save_frame)
                            saved_frame_count += 1
                            logger.info(f"Saved (subtitle disappeared): {timestamp_str}")
                            # 添加SRT条目（只有持续时间>=300ms才添加）
                            if use_ocr and pending_ocr_text and (end_ms - start_ms) >= 300:
                                srt_entries.append((len(srt_entries) + 1, start_ms, end_ms, pending_ocr_text))
                                # 更新最近保存的文字列表（最多5条）
                                recent_saved_texts.append(pending_ocr_text)
                                if len(recent_saved_texts) > 5:
                                    recent_saved_texts.pop(0)
                        except Exception as e:
                            logger.error(f"Error writing frame {frame_filename}: {e}")
                        
                        pending_save_frame = None  # 已保存，清空
                        pending_ocr_text = None
                        # 重要：更新stable_start_frame，这样下一句字幕的开始时间才正确
                        stable_start_frame = frame_count
                
                # 调试模式：在检测到时立即保存（不等合并）
                if debug_mode:
                    # 调试模式用简单时间戳格式（当前帧时间）
                    debug_ms = int(frame_count * 1000 / fps)
                    debug_hours = debug_ms // 3600000
                    debug_minutes = (debug_ms % 3600000) // 60000
                    debug_seconds = (debug_ms % 60000) // 1000
                    debug_milliseconds = debug_ms % 1000
                    debug_filename = f"{debug_hours}_{debug_minutes:02d}_{debug_seconds:02d}_{debug_milliseconds:03d}"
                    
                    # YOLO检测到目标时保存
                    if yolo_detected and yolo_results is not None:
                        yolo_debug_dir = out_dir + "_debug_yolo"
                        os.makedirs(yolo_debug_dir, exist_ok=True)
                        
                        # 绘制检测框
                        vis_frame = frame.copy()
                        boxes = yolo_results[0].boxes
                        h, w = frame.shape[:2]
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls_id = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            cls_name = yolo_model.names[cls_id]
                            
                            if detection_region:
                                rx, ry, _, _ = detection_region
                                x1, y1, x2, y2 = x1 + rx, y1 + ry, x2 + rx, y2 + ry
                            
                            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label = f"{cls_name} {conf:.2f}"
                            cv2.putText(vis_frame, label, (int(x1), int(y1) - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        vis_filename = osp.join(yolo_debug_dir, f"{debug_filename}.jpg")
                        cv2.imwrite(vis_filename, vis_frame)
                        
                        # 保存YOLO格式TXT
                        txt_filename = osp.join(yolo_debug_dir, f"{debug_filename}.txt")
                        with open(txt_filename, 'w') as f:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cls_id = int(box.cls[0].cpu().numpy())
                                if detection_region:
                                    rx, ry, _, _ = detection_region
                                    x1, y1, x2, y2 = x1 + rx, y1 + ry, x2 + rx, y2 + ry
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                box_w = (x2 - x1) / w
                                box_h = (y2 - y1) / h
                                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                    
                    # 场景变化时保存
                    if scene_changed and use_scene_detect:
                        scene_debug_dir = out_dir + "_debug_scene"
                        os.makedirs(scene_debug_dir, exist_ok=True)
                        
                        # 保存原始帧
                        scene_filename = osp.join(scene_debug_dir, f"{debug_filename}.jpg")
                        cv2.imwrite(scene_filename, frame)
                        
                        # 保存差异信息
                        try:
                            txt_filename = osp.join(scene_debug_dir, f"{debug_filename}.txt")
                            with open(txt_filename, 'w', encoding='utf-8') as f:
                                f.write(f"Diff Ratio: {diff_ratio:.4f}\n")
                        except Exception as e:
                            logger.warning(f"Failed to save scene debug: {e}")
                    
                    # OCR去重时保存识别结果
                    if use_ocr and current_ocr_text is not None and should_save:
                        ocr_debug_dir = out_dir + "_debug_ocr"
                        os.makedirs(ocr_debug_dir, exist_ok=True)
                        
                        try:
                            txt_filename = osp.join(ocr_debug_dir, f"{debug_filename}.txt")
                            with open(txt_filename, 'w', encoding='utf-8') as f:
                                f.write(f"OCR Text: {current_ocr_text}\n")
                        except Exception as e:
                            logger.warning(f"Failed to save OCR debug: {e}")
                
                # 更新进度
                if use_yolo or use_scene_detect:
                    progress_dialog.setValue(frame_count)
                    
                    # 计算运行时间和剩余时间
                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                    
                    if frame_count > 0:
                        # 估算剩余时间
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        remaining_frames = total_frames - frame_count
                        remaining_sec = remaining_frames / fps_actual if fps_actual > 0 else 0
                        remaining_str = f"{int(remaining_sec // 60):02d}:{int(remaining_sec % 60):02d}"
                    else:
                        remaining_str = "--:--"
                    
                    progress_dialog.setLabelText(
                        f"已检测 {frame_count}/{total_frames} 帧，提取 {saved_frame_count - 1} 张\n"
                        f"已用时: {elapsed_str}  剩余: {remaining_str}"
                    )
                
                if should_save:
                    # 延迟保存策略：检测到新字幕时，先保存上一句字幕
                    if pending_save_frame is not None and use_timestamp:
                        # 如果字幕已经消失过，使用消失时刻作为结束时间
                        # 否则使用当前帧-1作为结束时间
                        if last_subtitle_end_frame > pending_start_frame:
                            end_frame = last_subtitle_end_frame
                        else:
                            end_frame = frame_count - 1 if frame_count > 0 else 0
                        
                        # 计算开始时间
                        start_ms = int(pending_start_frame * 1000 / fps)
                        start_hours = start_ms // 3600000
                        start_minutes = (start_ms % 3600000) // 60000
                        start_seconds = (start_ms % 60000) // 1000
                        start_milliseconds = start_ms % 1000
                        
                        # 计算结束时间
                        end_ms = int(end_frame * 1000 / fps)
                        end_hours = end_ms // 3600000
                        end_minutes = (end_ms % 3600000) // 60000
                        end_seconds = (end_ms % 60000) // 1000
                        end_milliseconds = end_ms % 1000
                        
                        # 获取图像尺寸信息
                        h, w = pending_save_frame.shape[:2]
                        size_info = f"000000{0:04d}{0:04d}{w:04d}{h:04d}"
                        
                        # 组合VideoSubFinder格式文件名
                        timestamp_str = f"{start_hours}_{start_minutes:02d}_{start_seconds:02d}_{start_milliseconds:03d}__{end_hours}_{end_minutes:02d}_{end_seconds:02d}_{end_milliseconds:03d}_{size_info}"
                        frame_filename = osp.join(out_dir, f"{timestamp_str}.jpeg")
                        
                        try:
                            cv2.imwrite(frame_filename, pending_save_frame)
                            saved_frame_count += 1
                            logger.info(f"Saved (new subtitle): {timestamp_str}")
                            # 添加SRT条目（只有持续时间>=300ms才添加）
                            if use_ocr and pending_ocr_text and (end_ms - start_ms) >= 300:
                                srt_entries.append((len(srt_entries) + 1, start_ms, end_ms, pending_ocr_text))
                                # 更新最近保存的文字列表（最多5条）
                                recent_saved_texts.append(pending_ocr_text)
                                if len(recent_saved_texts) > 5:
                                    recent_saved_texts.pop(0)
                        except Exception as e:
                            logger.error(f"Error writing frame {frame_filename}: {e}")
                    
                    # 记录当前帧为待保存（等字幕消失或下一句字幕出现时再保存）
                    save_frame = frame_to_save if frame_to_save is not None else frame
                    pending_save_frame = save_frame.copy()
                    pending_start_frame = stable_start_frame
                    pending_ocr_text = current_ocr_text  # 记录当前OCR文本
                    last_subtitle_end_frame = 0  # 重置消失时刻
                    
                    # 重置
                    frame_to_save = None
                    stable_start_frame = frame_count
                    
                    if not use_timestamp:
                        # 非时间轴模式（图片提取模式），直接保存
                        if extract_enabled:
                            if extract_timestamp_naming:
                                # 时间轴命名格式: 0_01_16_266.jpg
                                current_ms = int(frame_count * 1000 / fps)
                                ts_hours = current_ms // 3600000
                                ts_minutes = (current_ms % 3600000) // 60000
                                ts_seconds = (current_ms % 60000) // 1000
                                ts_milliseconds = current_ms % 1000
                                base_filename = f"{ts_hours}_{ts_minutes:02d}_{ts_seconds:02d}_{ts_milliseconds:03d}"
                            else:
                                # 序号命名格式: frame_00001.jpg
                                base_filename = f"{prefix}{str(saved_frame_count).zfill(seq_len)}"
                            frame_filename = osp.join(out_dir, f"{base_filename}.jpg")
                            try:
                                cv2.imwrite(frame_filename, save_frame)
                                saved_frame_count += 1
                            except Exception as e:
                                logger.error(f"Error writing frame {frame_filename}: {e}")
                    
                    if not use_scene_detect and not use_yolo:
                        progress_dialog.setValue(saved_frame_count - 1)
                        
                        # 计算运行时间和剩余时间
                        elapsed = time.time() - start_time
                        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                        
                        if saved_frame_count > 1:
                            fps_actual = (saved_frame_count - 1) / elapsed if elapsed > 0 else 0
                            remaining_frames = estimated_frames - (saved_frame_count - 1)
                            remaining_sec = remaining_frames / fps_actual if fps_actual > 0 else 0
                            remaining_str = f"{int(remaining_sec // 60):02d}:{int(remaining_sec % 60):02d}"
                        else:
                            remaining_str = "--:--"
                        
                        progress_dialog.setLabelText(
                            f"正在提取帧... {saved_frame_count - 1}/{estimated_frames}\n"
                            f"已用时: {elapsed_str}  剩余: {remaining_str}"
                        )

                frame_count += 1
                # 非YOLO模式下每帧处理UI事件
                if not use_yolo:
                    QApplication.processEvents()

            # 循环结束后，保存最后一句字幕（如果有）
            if pending_save_frame is not None and use_timestamp:
                # 最后一句字幕的结束时间是视频最后一帧
                end_frame = frame_count
                
                # 计算开始时间
                start_ms = int(pending_start_frame * 1000 / fps)
                start_hours = start_ms // 3600000
                start_minutes = (start_ms % 3600000) // 60000
                start_seconds = (start_ms % 60000) // 1000
                start_milliseconds = start_ms % 1000
                
                # 计算结束时间
                end_ms = int(end_frame * 1000 / fps)
                end_hours = end_ms // 3600000
                end_minutes = (end_ms % 3600000) // 60000
                end_seconds = (end_ms % 60000) // 1000
                end_milliseconds = end_ms % 1000
                
                # 获取图像尺寸信息
                h, w = pending_save_frame.shape[:2]
                size_info = f"000000{0:04d}{0:04d}{w:04d}{h:04d}"
                
                # 组合VideoSubFinder格式文件名
                timestamp_str = f"{start_hours}_{start_minutes:02d}_{start_seconds:02d}_{start_milliseconds:03d}__{end_hours}_{end_minutes:02d}_{end_seconds:02d}_{end_milliseconds:03d}_{size_info}"
                frame_filename = osp.join(out_dir, f"{timestamp_str}.jpeg")
                
                try:
                    cv2.imwrite(frame_filename, pending_save_frame)
                    saved_frame_count += 1
                    logger.info(f"Saved last subtitle frame: {timestamp_str}")
                    # 添加SRT条目（只有持续时间>=300ms才添加）
                    if use_ocr and pending_ocr_text and (end_ms - start_ms) >= 300:
                        srt_entries.append((len(srt_entries) + 1, start_ms, end_ms, pending_ocr_text))
                        # 更新最近保存的文字列表（最多5条）
                        recent_saved_texts.append(pending_ocr_text)
                        if len(recent_saved_texts) > 5:
                            recent_saved_texts.pop(0)
                except Exception as e:
                    logger.error(f"Error writing last frame {frame_filename}: {e}")

            progress_dialog.close()
            
            # 生成SRT字幕文件（保存在输出文件夹外面，和文件夹同级）
            if use_ocr and srt_entries:
                # 使用输出文件夹的名称作为字幕文件名
                out_dir_name = osp.basename(out_dir)
                
                # 生成SRT文件
                srt_filename = osp.join(osp.dirname(out_dir), f"{out_dir_name}.srt")
                try:
                    with open(srt_filename, 'w', encoding='utf-8') as f:
                        for idx, start, end, text in srt_entries:
                            f.write(f"{idx}\n")
                            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
                            f.write(f"{text}\n\n")
                    logger.info(f"SRT file saved: {srt_filename} ({len(srt_entries)} entries)")
                except Exception as e:
                    logger.error(f"Error writing SRT file: {e}")
                
                # 生成ASS文件
                ass_filename = osp.join(osp.dirname(out_dir), f"{out_dir_name}.ass")
                try:
                    with open(ass_filename, 'w', encoding='utf-8-sig') as f:
                        f.write(ASS_TEMPLATE)
                        for idx, start, end, text in srt_entries:
                            start_ass = format_ass_time(start)
                            end_ass = format_ass_time(end)
                            f.write(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n")
                    logger.info(f"ASS file saved: {ass_filename} ({len(srt_entries)} entries)")
                except Exception as e:
                    logger.error(f"Error writing ASS file: {e}")

            if extraction_cancelled:
                logger.warning(
                    f"Extraction cancelled. Frames saved so far (OpenCV): {saved_frame_count - 1}"
                )
            else:
                logger.info(
                    f"OpenCV extraction finished. Saved frames: {saved_frame_count - 1}"
                )

            # --- Common success return ---
            return out_dir

        # Except block for the extraction phase
        except Exception as extraction_e:
            logger.exception(
                f"An unexpected error occurred during frame extraction logic: {extraction_e}"
            )
            popup = Popup(
                f"提取帧时发生错误: {extraction_e}",
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return None

    # Except block for the *outer* try (opening/setup phase)
    except Exception as opening_e:
        logger.exception(
            f"An unexpected error occurred during video opening/setup: {opening_e}"
        )
        popup = Popup(
            f"打开视频时发生错误: {opening_e}",
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return None

    # Finally block for the *outer* try (always runs)
    finally:
        # Release capture if it exists and is opened (mainly for OpenCV path or if ffmpeg failed early)
        if video_capture is not None and video_capture.isOpened():
            logger.info("Releasing video capture resource.")
            video_capture.release()
        # Clean up the temporary file if created
        if temp_video_path and osp.exists(temp_video_path):
            try:
                logger.debug(
                    f"Removing temporary video file: {temp_video_path}"
                )
                os.remove(temp_video_path)
            except OSError as e:
                logger.error(
                    f"Error removing temporary file {temp_video_path}: {e}"
                )


def open_video_file(self, video_path=None, zimu_lujing=None):
    """打开视频 -> 视频工作台窗口（画面逐帧 / 时间轴 / 参数区）

    zimu_lujing 给了就顺手把这份字幕也载进来（同时拖视频 + 字幕时用）。
    """
    if not self.may_continue():
        return

    from anylabeling.views.labeling.widgets.video_work_dialog import (
        dakai_video_gongzuotai,
    )

    dakai_video_gongzuotai(self, video_path, zimu_lujing)