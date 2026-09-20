# -*- coding: utf-8 -*-
"""实时推理窗口

帧源两种：
    桌面区域抓取 / 整块显示器抓取

用「自动标注」面板当前已加载的模型逐帧推理，结果直接画在画面上，
同时按帧落盘成软件自己的 JSON（图片 + 同名 .json），目录规则和「从剪贴板粘贴」一致：

    <软件根目录>\\实时推理\\<YYYYMMDD_HHMMSS>\\

不做追踪、不做计数。
"""

import json
import os
import os.path as osp
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QMutex, QMutexLocker, QRect, QRectF, QThread, QTimer
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from anylabeling.services.auto_labeling.utils.general import (
    calculate_rotation_theta,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_highlight_button_style,
    get_normal_button_style,
)

# ==================== 需要调的值都在这里，只改这一行 ====================
MULU_MINGZI = "实时推理"          # 输出根目录名（在软件根目录下）
ZHUOMIAN_FPS = 15                # 抓屏帧率
ZHEN_CHA_YUZHI = 0.02            # 帧差阈值：低于它认为和上一帧一样，跳过不存（设 0 = 全存）
TUPIAN_ZHILIANG = 95             # 存图 JPG 质量
KONGXIAN_HAOMIAO = 2             # 推理线程没帧可取时的等待毫秒（防止空转烧 CPU）
JIAO_BEN_HAO = "1.0.0"           # 写进 JSON 的 version
RENGONG_BIANJI = False           # 写进 JSON 的 manually_edited（False=AI 产出，文件列表不标橙色）
# ====================================================================

# 这几类模型跑不了实时（rmbg 是抠图，不是检测）
_BU_ZHI_CHI_SHI_SHI = ("rmbg",)

# 这几类模型要 PIL 图像输入（实时帧直接转 PIL 喂进去，不落盘）
_PIL_SHU_RU = ("rfdetr", "dfine", "rio_detr", "yoloe")

# 画框配色（只在拿不到主界面配色时的备用），正常都用软件里设置的那套
_YANSE_BIAO = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#fabebe",
]


def shuchu_genmulu():
    """输出根目录：软件根目录下的「实时推理」

    与「从剪贴板粘贴」（label_widget.py 的 paste_image_from_clipboard）
    用的是同一套定位方式：
        开发环境  .../anylabeling/views/labeling/widgets/本文件 → 取到 anylabeling 的上一级
        打包环境  取 exe 所在目录
    """
    import sys

    if getattr(sys, "frozen", False):
        application_dir = osp.dirname(sys.executable)
    else:
        application_dir = osp.dirname(
            osp.dirname(
                osp.dirname(osp.dirname(osp.abspath(__file__)))
            )
        )
    return osp.join(osp.dirname(application_dir), MULU_MINGZI)


def zhentu_zhuan_qimage(zhen):
    """RGB ndarray → QImage（带拷贝，避免 numpy 缓冲被回收）

    末尾必须转成 Format_RGB32：模型内部经 qimage2ndarray 取像素，
    它只认 8/16/32/64 位图，24 位的 RGB888 会直接抛异常，
    导致每帧都推理失败（拿到空结果）。
    """
    gao, kuan = zhen.shape[:2]
    return (
        QImage(zhen.data, kuan, gao, 3 * kuan, QImage.Format_RGB888)
        .copy()
        .convertToFormat(QImage.Format_RGB32)
    )


def zhentu_zhuan_pil(zhen):
    """RGB ndarray → PIL.Image（rfdetr/dfine/rio_detr/yoloe 只吃 PIL）"""
    return Image.fromarray(zhen)


def qimage_zhuan_zhentu(tupian):
    """QImage → RGB ndarray"""
    tupian = tupian.convertToFormat(QImage.Format_RGB888)
    kuan = tupian.width()
    gao = tupian.height()
    if kuan <= 0 or gao <= 0:
        return None
    zhi = tupian.constBits()
    zhi.setsize(tupian.byteCount())
    huanchong = np.frombuffer(zhi, dtype=np.uint8)
    huanchong = huanchong.reshape((gao, tupian.bytesPerLine()))
    return huanchong[:, : kuan * 3].reshape((gao, kuan, 3)).copy()


def shifou_keyong(moxing):
    """模型能不能跑实时（判断依据和裁切检测一致：看模块名）"""
    mokuaiming = f"{type(moxing).__module__}".lower()
    for guanjianzi in _BU_ZHI_CHI_SHI_SHI:
        if guanjianzi in mokuaiming:
            return False, mokuaiming
    return True, mokuaiming


class _TuiliXiancheng(QThread):
    """推理线程：队列只留最新一帧，永远处理最新画面"""

    tuili_wancheng = pyqtSignal(object, object)   # 帧(RGB ndarray), shapes 列表

    def __init__(self, moxing, parent=None):
        super().__init__(parent)
        self.moxing = moxing
        self.duilie = deque(maxlen=1)
        self.suo = QMutex()
        self._yunxing = False
        self.mokuai_biao = (
            f"{type(moxing).__module__}.{type(moxing).__name__}".lower()
        )

    def fang_zhen(self, zhen):
        with QMutexLocker(self.suo):
            self.duilie.append(zhen)

    def run(self):
        self._yunxing = True
        logger.info("实时推理：推理线程启动")
        while self._yunxing:
            zhen = None
            with QMutexLocker(self.suo):
                if self.duilie:
                    zhen = self.duilie.pop()
            if zhen is None:
                # 没帧可推理时睡一下，不能空转（空转会吃满一个核）
                self.msleep(KONGXIAN_HAOMIAO)
                continue
            try:
                shapes = self._tui(zhen)
            except Exception as cuowu:  # noqa
                logger.error(f"实时推理单帧失败: {cuowu}")
                shapes = []
            self.tuili_wancheng.emit(zhen, shapes)
        logger.info("实时推理：推理线程结束")

    def _tui(self, zhen):
        """喂一帧给模型。

        输入兼容规则和裁切检测里的 _call_model_on_crop 一致：
            comic_text_detector 只认数组(BGR)，其余模型吃 QImage。
        """
        if "comic_text_detector" in self.mokuai_biao:
            bgr = cv2.cvtColor(zhen, cv2.COLOR_RGB2BGR).copy()
            jieguo = self.moxing.predict_shapes(bgr, None)
        elif any(k in self.mokuai_biao for k in _PIL_SHU_RU):
            # 这批模型要 PIL 图像：内存直接转，不落盘
            jieguo = self.moxing.predict_shapes(zhentu_zhuan_pil(zhen), None)
        else:
            jieguo = self.moxing.predict_shapes(zhentu_zhuan_qimage(zhen), None)
        return list(getattr(jieguo, "shapes", None) or [])

    def ting(self):
        self._yunxing = False
        self.wait(3000)


class _QuyuXuanzeFuGai(QtWidgets.QWidget):
    """全屏遮罩：拖鼠标框选桌面区域"""

    xuanqu_wancheng = pyqtSignal(object, object)   # QScreen, QRect(屏幕内坐标)

    def __init__(self):
        super().__init__(
            None,
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.qidian = None
        self.zhongdian = None

        quyu = QRect()
        for pingmu in QtWidgets.QApplication.screens():
            quyu = quyu.united(pingmu.geometry())
        self.setGeometry(quyu)

    def paintEvent(self, shijian):  # noqa
        huabi = QPainter(self)
        huabi.fillRect(self.rect(), QColor(0, 0, 0, 110))
        if self.qidian is None or self.zhongdian is None:
            huabi.setPen(QColor(255, 255, 255))
            huabi.setFont(QFont("Microsoft YaHei", 14))
            huabi.drawText(
                self.rect(),
                Qt.AlignCenter,
                "拖动鼠标框选桌面区域，Esc 取消",
            )
            return

        kuang = QRect(self.qidian, self.zhongdian).normalized()
        huabi.setCompositionMode(QPainter.CompositionMode_Clear)
        huabi.fillRect(kuang, Qt.transparent)
        huabi.setCompositionMode(QPainter.CompositionMode_SourceOver)
        huabi.setPen(QPen(QColor(0, 255, 0), 2))
        huabi.drawRect(kuang)
        huabi.setPen(QColor(255, 255, 255))
        huabi.setFont(QFont("Microsoft YaHei", 10))
        huabi.drawText(
            kuang.topLeft() + QtCore.QPoint(4, -6),
            f"{kuang.width()} x {kuang.height()}",
        )

    def mousePressEvent(self, shijian):
        if shijian.button() == Qt.LeftButton:
            self.qidian = shijian.pos()
            self.zhongdian = shijian.pos()
            self.update()

    def mouseMoveEvent(self, shijian):
        if self.qidian is not None:
            self.zhongdian = shijian.pos()
            self.update()

    def mouseReleaseEvent(self, shijian):
        if shijian.button() != Qt.LeftButton or self.qidian is None:
            return
        kuang = QRect(self.qidian, self.zhongdian).normalized()
        self.close()
        if kuang.width() < 8 or kuang.height() < 8:
            return

        # 遮罩本地坐标 → 全局坐标 → 换算成所在屏幕内的坐标
        quanju = kuang.translated(self.geometry().topLeft())
        pingmu = QtWidgets.QApplication.screenAt(quanju.center())
        if pingmu is None:
            pingmu = QtWidgets.QApplication.primaryScreen()
        if pingmu is None:
            return
        pingmu_quyu = pingmu.geometry()
        xiangdui = QRect(
            quanju.x() - pingmu_quyu.x(),
            quanju.y() - pingmu_quyu.y(),
            quanju.width(),
            quanju.height(),
        )
        self.xuanqu_wancheng.emit(pingmu, xiangdui)

    def keyPressEvent(self, shijian):
        if shijian.key() == Qt.Key_Escape:
            self.close()


class RealtimeInferDialog(QtWidgets.QDialog):
    """实时推理窗口"""

    zhuangtai_gaibian = pyqtSignal(bool)   # True=正在实时推理

    def __init__(self, zidong_biaozhu, parent=None):
        super().__init__(parent)
        self.zidong_biaozhu = zidong_biaozhu      # 自动标注面板（模型从这里取）
        self.moshi = ""                           # "" / zhuomian / xianshiqi
        self.zhuomian_pingmu = None
        self.zhuomian_quyu = None
        self.shuchu_mulu = ""
        self.leibiao = []                         # 类别名，顺序即 YOLO 索引
        self.yanse_huancun = {}                   # 标签名 → 框颜色（跟主界面一致）
        self.tuili = None
        self.zhuomian_dingshi = None
        self.zhengzai = False
        self.shang_yizhen = None
        self.yicun_shu = 0
        self.zhen_lv = 0.0
        self.shangci_shijian = time.time()
        self.zuihou_zhen = None                   # 最后一帧原图（窗口缩放时重画用）
        self.zuihou_shapes = []                   # 最后一帧的检测结果
        self._jian_jiemian()

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _jian_jiemian(self):
        self.setWindowTitle("实时推理")
        self.setMinimumSize(760, 540)
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
        )

        zong = QtWidgets.QVBoxLayout(self)

        di_yi_hang = QtWidgets.QHBoxLayout()
        di_yi_hang.addWidget(QtWidgets.QLabel("帧源:"))

        self.kuang_xuan_anniu = QtWidgets.QPushButton("框选区域…")
        self.kuang_xuan_anniu.clicked.connect(self._kuang_xuan)
        di_yi_hang.addWidget(self.kuang_xuan_anniu)

        self.xuan_xianshiqi_anniu = QtWidgets.QPushButton("选择显示器…")
        self.xuan_xianshiqi_anniu.clicked.connect(self._xuan_xianshiqi)
        self.xuan_xianshiqi_anniu.setToolTip("整块显示器画面实时推理")
        di_yi_hang.addWidget(self.xuan_xianshiqi_anniu)

        di_yi_hang.addStretch()
        di_yi_hang.addWidget(QtWidgets.QLabel("输出:"))
        self.mulu_biaoqian = QtWidgets.QLabel("未开始")
        self.mulu_biaoqian.setStyleSheet("color:#555;")
        di_yi_hang.addWidget(self.mulu_biaoqian)
        zong.addLayout(di_yi_hang)

        di_er_hang = QtWidgets.QHBoxLayout()
        self.kaishi_anniu = QtWidgets.QPushButton("开始")
        self.kaishi_anniu.setStyleSheet(get_highlight_button_style())
        self.kaishi_anniu.clicked.connect(self._qiehuan_kaishi)
        di_er_hang.addWidget(self.kaishi_anniu)

        self.tiaoguo_xiangsi = QtWidgets.QCheckBox("跳过相似帧")
        self.tiaoguo_xiangsi.setChecked(True)
        self.tiaoguo_xiangsi.setToolTip(
            "相邻两帧几乎一样时不存，避免存出成百上千张重复图"
        )
        di_er_hang.addWidget(self.tiaoguo_xiangsi)

        di_er_hang.addStretch()
        self.zhuangtai_biaoqian = QtWidgets.QLabel("状态: 未开始")
        di_er_hang.addWidget(self.zhuangtai_biaoqian)
        zong.addLayout(di_er_hang)

        self.huamian = QtWidgets.QLabel("先选帧源，再点「开始」")
        self.huamian.setAlignment(Qt.AlignCenter)
        self.huamian.setMinimumSize(640, 380)
        self.huamian.setStyleSheet("background:#1e1e1e; color:#bbbbbb;")
        # 窗口一改大小就按新尺寸重画，画面跟着自适应
        self.huamian.installEventFilter(self)
        zong.addWidget(self.huamian, 1)

        self._shuaxin_anniu()

    def eventFilter(self, duixiang, shijian):
        """画面控件尺寸变了（拉窗口/最大化）→ 拿最后一帧按新尺寸重画"""
        if duixiang is self.huamian and shijian.type() == QtCore.QEvent.Resize:
            if self.zuihou_zhen is not None:
                self._huamian_xianshi(self.zuihou_zhen, self.zuihou_shapes)
        return super().eventFilter(duixiang, shijian)

    def _shuaxin_anniu(self):
        """高亮当前帧源那个按钮（按钮始终可点，点谁用谁）"""
        anniu_biao = {
            "zhuomian": self.kuang_xuan_anniu,
            "xianshiqi": self.xuan_xianshiqi_anniu,
        }
        for moshi, anniu in anniu_biao.items():
            anniu.setStyleSheet(
                get_highlight_button_style()
                if self.moshi == moshi
                else get_normal_button_style()
            )

    def _kuang_xuan(self):
        if self.moshi != "zhuomian" and self.zhengzai:
            self._tingzhi("已停止（切换帧源）")
        self.moshi = "zhuomian"
        self._shuaxin_anniu()
        self.fugai = _QuyuXuanzeFuGai()
        self.fugai.xuanqu_wancheng.connect(self._shoudao_quyu)
        self.fugai.show()
        self.fugai.raise_()
        self.fugai.activateWindow()

    def _shoudao_quyu(self, pingmu, quyu):
        self.zhuomian_pingmu = pingmu
        self.zhuomian_quyu = quyu
        self.huamian.setText(
            f"桌面区域: {quyu.x()},{quyu.y()}  {quyu.width()}x{quyu.height()}"
        )
        logger.info(
            f"实时推理：选中桌面区域 {quyu.x()},{quyu.y()} "
            f"{quyu.width()}x{quyu.height()}"
        )

    def _xuan_xianshiqi(self):
        """弹出显示器清单，选哪块就整屏推理"""
        caidan = QtWidgets.QMenu(self)
        zhupingmu = QtWidgets.QApplication.primaryScreen()
        for xuhao, pingmu in enumerate(
            QtWidgets.QApplication.screens(), start=1
        ):
            quyu = pingmu.geometry()
            mingzi = f"显示器 {xuhao}   {quyu.width()} x {quyu.height()}"
            if pingmu is zhupingmu:
                mingzi += "   (主显示器)"
            dongzuo = caidan.addAction(mingzi)
            dongzuo.triggered.connect(
                lambda _=False, p=pingmu: self._shoudao_xianshiqi(p)
            )

        dian = self.xuan_xianshiqi_anniu.mapToGlobal(
            QtCore.QPoint(0, self.xuan_xianshiqi_anniu.height())
        )
        if not caidan.exec_(dian):
            return   # 没选就什么都不改

        if self.moshi != "xianshiqi" and self.zhengzai:
            self._tingzhi("已停止（切换帧源）")
        self.moshi = "xianshiqi"
        self._shuaxin_anniu()

    def _shoudao_xianshiqi(self, pingmu):
        quyu = pingmu.geometry()
        # grabWindow 要的是「屏幕内」坐标，整屏就是 0,0 起
        self.zhuomian_pingmu = pingmu
        self.zhuomian_quyu = QRect(0, 0, quyu.width(), quyu.height())
        self.huamian.setText(f"显示器整屏: {quyu.width()}x{quyu.height()}")
        logger.info(f"实时推理：选中显示器整屏 {quyu.width()}x{quyu.height()}")

    # ------------------------------------------------------------------
    # 起停
    # ------------------------------------------------------------------
    def _qiehuan_kaishi(self):
        if self.zhengzai:
            self._tingzhi("已停止")
        else:
            self._kaishi()

    def _kaishi(self):
        peizhi = self.zidong_biaozhu.model_manager.loaded_model_config or {}
        moxing = peizhi.get("model")
        if moxing is None:
            QtWidgets.QMessageBox.warning(
                self, "提示", "请先在自动标注面板加载一个模型"
            )
            return

        keyong, mokuaiming = shifou_keyong(moxing)
        if not keyong:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                f"这个模型不能用于实时推理：\n{mokuaiming}\n\n"
                "它每帧都要写一次临时文件，跟不上实时。",
            )
            return

        if self.zhuomian_pingmu is None or self.zhuomian_quyu is None:
            if self.moshi == "xianshiqi":
                tishi = "请先选择显示器"
            else:
                tishi = "请先框选桌面区域"
            QtWidgets.QMessageBox.warning(self, "提示", tishi)
            return

        shijian = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.shuchu_mulu = osp.join(shuchu_genmulu(), shijian)
        try:
            os.makedirs(self.shuchu_mulu, exist_ok=True)
        except OSError as cuowu:
            QtWidgets.QMessageBox.critical(
                self, "错误", f"建不了输出目录：\n{cuowu}"
            )
            return

        self.leibiao = [str(x) for x in (peizhi.get("classes") or [])]
        self.mulu_biaoqian.setText(shijian)
        self.mulu_biaoqian.setToolTip(self.shuchu_mulu)

        self.tuili = _TuiliXiancheng(moxing, self)
        self.tuili.tuili_wancheng.connect(self._shoudao_jieguo)
        self.tuili.start()

        self.zhuomian_dingshi = QTimer(self)
        self.zhuomian_dingshi.timeout.connect(self._zhua_zhuomian)
        self.zhuomian_dingshi.start(int(1000 / ZHUOMIAN_FPS))

        self.zhengzai = True
        self.shang_yizhen = None
        self.yicun_shu = 0
        self.zhen_lv = 0.0
        self.shangci_shijian = time.time()
        # 每次开始都重新取一遍配色，中途在软件里改过颜色也能跟上
        self.yanse_huancun.clear()
        self.kaishi_anniu.setText("停止")
        self.zhuangtai_gaibian.emit(True)
        self._gengxin_zhuangtai()
        logger.info(f"实时推理：开始，输出到 {self.shuchu_mulu}")

    def _tingzhi(self, jieshu_wenben="已停止"):
        if self.zhuomian_dingshi is not None:
            self.zhuomian_dingshi.stop()
            self.zhuomian_dingshi = None
        if self.tuili is not None:
            self.tuili.ting()
            self.tuili = None
        self.zhengzai = False
        self.kaishi_anniu.setText("开始")
        self.zhuangtai_gaibian.emit(False)
        self.zhuangtai_biaoqian.setText(
            f"{jieshu_wenben} | 本次已存 {self.yicun_shu} 帧"
        )
        logger.info(f"实时推理：{jieshu_wenben}，共存 {self.yicun_shu} 帧")

    # ------------------------------------------------------------------
    # 取帧
    # ------------------------------------------------------------------
    def _zhua_zhuomian(self):
        if not self.zhengzai or self.tuili is None:
            return
        if self.zhuomian_pingmu is None or self.zhuomian_quyu is None:
            return
        quyu = self.zhuomian_quyu
        try:
            tupian = self.zhuomian_pingmu.grabWindow(
                0, quyu.x(), quyu.y(), quyu.width(), quyu.height()
            )
        except Exception as cuowu:  # noqa
            logger.error(f"实时推理：抓屏失败 {cuowu}")
            return
        if tupian is None or tupian.isNull():
            return
        zhen = qimage_zhuan_zhentu(tupian.toImage())
        if zhen is not None:
            self.tuili.fang_zhen(zhen)

    # ------------------------------------------------------------------
    # 结果：显示 + 落盘
    # ------------------------------------------------------------------
    def _shoudao_jieguo(self, zhen, shapes):
        if not self.zhengzai:
            return
        xianzai = time.time()
        jian_ge = xianzai - self.shangci_shijian
        self.shangci_shijian = xianzai
        if jian_ge > 0:
            self.zhen_lv = 0.85 * self.zhen_lv + 0.15 * (1.0 / jian_ge)

        self._huamian_xianshi(zhen, shapes)
        if self._luopan(zhen, shapes):
            self.yicun_shu += 1
        self._gengxin_zhuangtai()

    def _huamian_xianshi(self, zhen, shapes):
        # 记住最后一帧，窗口缩放时靠它按新尺寸重画
        self.zuihou_zhen = zhen
        self.zuihou_shapes = shapes
        gao, kuan = zhen.shape[:2]
        yuanshi = QPixmap.fromImage(
            QImage(zhen.data, kuan, gao, 3 * kuan, QImage.Format_RGB888)
        )
        suofang = yuanshi.scaled(
            self.huamian.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        bi = suofang.width() / kuan if kuan else 1.0

        huabi = QPainter(suofang)
        huabi.setRenderHint(QPainter.Antialiasing)
        for xingzhuang in shapes:
            dian = getattr(xingzhuang, "points", None)
            if not dian:
                continue
            pings = [(p.x() * bi, p.y() * bi) for p in dian]
            biaoqian = str(getattr(xingzhuang, "label", "") or "")
            yanse = self._qu_kuang_yanse(biaoqian)
            huabi.setPen(QPen(yanse, 2))

            # 三点及以上的都按多边形画：OBB 的 shape_type 是 "rotation"（4 个角点），
            # 以前只认 "polygon"，旋转框就被当成轴对齐矩形画了
            if len(pings) >= 3:
                huabi.drawPolygon(
                    QtGui.QPolygonF(
                        [QtCore.QPointF(x, y) for x, y in pings]
                    )
                )
                zuo_biao = pings[0]
            else:
                xs = [p[0] for p in pings]
                ys = [p[1] for p in pings]
                kuang = QRectF(
                    min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                )
                huabi.drawRect(kuang)
                zuo_biao = (kuang.left(), kuang.top())

            if biaoqian:
                huabi.setFont(QFont("Microsoft YaHei", 9))
                zihao = huabi.fontMetrics()
                kuan_t = zihao.width(biaoqian) + 8
                gao_t = zihao.height()
                beijing = QRectF(
                    zuo_biao[0], max(0.0, zuo_biao[1] - gao_t), kuan_t, gao_t
                )
                huabi.fillRect(beijing, QColor(0, 0, 0, 160))
                huabi.setPen(yanse)
                huabi.drawText(beijing, Qt.AlignCenter, biaoqian)
        huabi.end()
        self.huamian.setPixmap(suofang)

    def _qu_kuang_yanse(self, biaoqian):
        """标签名 → 框颜色：直接问主界面要，和软件里设的配色完全一致

        优先级和主界面画框（shape.py 默认态）一样：
            高亮态独立边框色 > 默认态独立边框色 > 标签色
        问不到（比如窗口已销毁）才退回自带的备用配色。
        """
        yanse = self.yanse_huancun.get(biaoqian)
        if yanse is not None:
            return yanse

        zhu = getattr(self.zidong_biaozhu, "parent", None)
        if zhu is not None and hasattr(zhu, "_get_rgb_by_label"):
            try:
                from anylabeling.views.labeling.shape import Shape

                zhu_yanse = None
                if Shape.highlighting_enabled:
                    zhu_yanse = zhu._get_border_rgb_by_label(biaoqian)
                if zhu_yanse is None:
                    zhu_yanse = zhu._get_default_border_color_by_label(biaoqian)
                if zhu_yanse is None:
                    zhu_yanse = zhu._get_rgb_by_label(biaoqian)
                if zhu_yanse:
                    if isinstance(zhu_yanse, (list, tuple)):
                        yanse = QColor(
                            int(zhu_yanse[0]), int(zhu_yanse[1]), int(zhu_yanse[2])
                        )
                    else:
                        yanse = QColor(str(zhu_yanse))
            except Exception as cuowu:  # noqa
                logger.error(f"实时推理：取主界面框颜色失败，改用备用配色 {cuowu}")

        if yanse is None:
            yanse = QColor(
                _YANSE_BIAO[self._leibie_suoyin(biaoqian) % len(_YANSE_BIAO)]
            )
        self.yanse_huancun[biaoqian] = yanse
        return yanse

    def _leibie_suoyin(self, biaoqian):
        """类别名 → 索引（只用来决定框的颜色），没见过的排到后面"""
        if biaoqian in self.leibiao:
            return self.leibiao.index(biaoqian)
        self.leibiao.append(biaoqian)
        logger.info(f"实时推理：出现新类别 {biaoqian}，配色索引 {len(self.leibiao) - 1}")
        return len(self.leibiao) - 1

    def _luopan(self, zhen, shapes):
        """存一帧：原图（不带框）+ 同名 JSON 标注。真正存了才返回 True"""
        if not self.shuchu_mulu:
            return False
        if self.tiaoguo_xiangsi.isChecked() and not self._zhen_bu_tong(zhen):
            return False

        gao, kuan = zhen.shape[:2]
        if kuan <= 0 or gao <= 0:
            return False

        shijian_biao = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tupian_lujing = osp.join(self.shuchu_mulu, f"{shijian_biao}.jpg")
        biaozhu_lujing = osp.join(self.shuchu_mulu, f"{shijian_biao}.json")

        xingzhuang_liebiao = []
        for xingzhuang in shapes:
            yige = self._zhuan_json_xingzhuang(xingzhuang, kuan, gao)
            if yige is not None:
                xingzhuang_liebiao.append(yige)

        neirong = {
            "version": JIAO_BEN_HAO,
            "flags": {},
            "shapes": xingzhuang_liebiao,
            "imagePath": osp.basename(tupian_lujing),
            "imageData": None,
            "imageHeight": gao,
            "imageWidth": kuan,
            "manually_edited": RENGONG_BIANJI,
            "description": "",
        }

        try:
            cheng, huanchong = cv2.imencode(
                ".jpg",
                cv2.cvtColor(zhen, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), TUPIAN_ZHILIANG],
            )
            if not cheng:
                logger.error("实时推理：图片编码失败，本帧不存")
                return False
            # 用 tofile 写盘：cv2.imwrite 遇到中文目录（「实时推理」）
            # 会静默返回 False 且不报错，图片会一张都存不下来
            huanchong.tofile(tupian_lujing)
            with open(biaozhu_lujing, "w", encoding="utf-8") as wenjian:
                json.dump(neirong, wenjian, ensure_ascii=False, indent=2)
        except Exception as cuowu:  # noqa
            logger.error(f"实时推理：存帧失败 {cuowu}")
            return False

        self.shang_yizhen = zhen
        return True

    def _zhuan_json_xingzhuang(self, xingzhuang, kuan, gao):
        """一个模型 shape → 软件 JSON 里的一条，坐标用绝对像素"""
        dian = getattr(xingzhuang, "points", None)
        if not dian:
            return None
        pings = [[float(p.x()), float(p.y())] for p in dian]
        if not pings:
            return None

        leixing = str(getattr(xingzhuang, "shape_type", "") or "rectangle")
        fangxiang = None
        if leixing == "rectangle":
            # 软件规范：矩形固定 4 个点 左上 → 右上 → 右下 → 左下
            xs = [p[0] for p in pings]
            ys = [p[1] for p in pings]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            pings = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif leixing == "rotation":
            # OBB 旋转框：模型自己带 direction（弧度），没有就按角点算一个
            fangxiang = getattr(xingzhuang, "direction", None)
            if fangxiang is None:
                fangxiang = calculate_rotation_theta(pings)
            try:
                fangxiang = float(fangxiang)
            except (TypeError, ValueError):
                fangxiang = 0.0

        # 夹到图片范围内
        pings = [
            [min(max(x, 0.0), float(kuan)), min(max(y, 0.0), float(gao))]
            for x, y in pings
        ]

        fenshu = getattr(xingzhuang, "score", None)
        try:
            fenshu = float(fenshu) if fenshu is not None else None
        except (TypeError, ValueError):
            fenshu = None

        yige = {
            "label": str(getattr(xingzhuang, "label", "") or ""),
            "score": fenshu,
            "points": pings,
            "group_id": getattr(xingzhuang, "group_id", None),
            "description": getattr(xingzhuang, "description", None),
            "translation": getattr(xingzhuang, "translation", "") or "",
            "difficult": bool(getattr(xingzhuang, "difficult", False)),
            "shape_type": leixing,
            "flags": getattr(xingzhuang, "flags", None) or {},
            "attributes": getattr(xingzhuang, "attributes", None) or {},
            "kie_linking": getattr(xingzhuang, "kie_linking", None) or [],
            "is_edited": bool(getattr(xingzhuang, "is_edited", False)),
            "is_manually_locked": bool(
                getattr(xingzhuang, "is_manually_locked", False)
            ),
        }
        if fangxiang is not None:
            # 和软件 shape.to_dict() 一样：direction 排在最后，只有 rotation 才有
            yige["direction"] = fangxiang
        return yige

    def _zhen_bu_tong(self, zhen):
        """和上一张存下来的帧比，差得够多才返回 True"""
        if self.shang_yizhen is None:
            return True
        try:
            xiao_xin = cv2.resize(zhen, (64, 64))
            xiao_jiu = cv2.resize(self.shang_yizhen, (64, 64))
            hui_xin = cv2.cvtColor(xiao_xin, cv2.COLOR_RGB2GRAY)
            hui_jiu = cv2.cvtColor(xiao_jiu, cv2.COLOR_RGB2GRAY)
            cha = cv2.absdiff(hui_xin, hui_jiu)
            bilv = float(np.sum(cha)) / (64 * 64 * 255)
            return bilv > ZHEN_CHA_YUZHI
        except Exception as cuowu:  # noqa
            logger.error(f"实时推理：帧差计算失败 {cuowu}")
            return True

    def _gengxin_zhuangtai(self):
        if self.zhengzai:
            self.zhuangtai_biaoqian.setText(
                f"状态: 运行中 | {self.zhen_lv:.1f} FPS | 已存 {self.yicun_shu} 帧"
            )
        else:
            self.zhuangtai_biaoqian.setText("状态: 未开始")

    def closeEvent(self, shijian):
        if self.zhengzai:
            self._tingzhi("窗口关闭")
        else:
            self.zhuangtai_gaibian.emit(False)
        shijian.accept()
