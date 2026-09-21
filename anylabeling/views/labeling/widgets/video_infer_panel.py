# -*- coding: utf-8 -*-
"""视频工作台 —— 实时推理 / 硬字幕提取（右侧两块面板 + 后台跑）

一块设置管两件事：
    勾了 OCR   = 硬字幕提取（按字幕段存图 + SRT/ASS + 时间轴字幕块）
    没勾 OCR   = 实时推理（只检测、画框，存图 + 同名 JSON）

两种跑法，共用同一套单帧处理逻辑（ZhenChuli），所以结果完全一致：
    跟随播放   主线程取到帧就丢进推理线程；线程只处理"最新一帧"，
               跑不过来自然丢帧，画面不卡
    全片扫描   扫描线程自己开 cap 全速读帧，逐帧处理，不播放

界面位置（都在 video_work_dialog.py 的视频工作台里）：
    上块  设置                    -> ShezhiMianban
    下块  OCR 时间轴 + 字幕列表    -> ZimuMianban
    底部  大时间轴的波形上叠字幕块  -> 画在 video_work_dialog.py 的 ShijianZhou 里

检测模型、OCR 模型的手选对话框、SRT/ASS 格式，
全部复用 utils/video.py 里现成的那套，本文件不重复实现。
检测区域直接在 video_work_dialog.py 的视频画面上鼠标拖拽框选。
"""

import os
import os.path as osp
import re
import time
from collections import Counter

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from anylabeling.views.labeling.logger import logger


# =====================================================================
# 用户可调参数（只改这一段）
# =====================================================================
MULU_HOUZHUI = "_字幕提取"      # 输出文件夹：建在视频旁边，名字 = 视频名 + 这个后缀
YANSE_KUANG = "#00E5FF"         # 检测框颜色
YANSE_KUANG_ZI = "#FF453A"      # 检测框上的类别文字颜色
YANSE_ZIMU = "#2F7FE0"          # 字幕块颜色（时间轴上的那种蓝，所有块都一样）
YANSE_ZIMU_ZI = "#FFFFFF"       # 字幕块上的文字颜色（不管选没选中都这个色，不变）
YANSE_ZIMU_XUAN = "#22C55E"     # 选中的字幕块：只沿块内部描一圈绿框，底色文字都不动
YANSE_ZIMU_BIAN_A = "#123E6E"   # 字幕块描边色（奇数块用的一号深蓝）
YANSE_ZIMU_BIAN_B = "#8A4B00"   # 字幕块描边色（偶数块用的二号深棕），两块交替描边
YANSE_ZIMU_QIU = "#FFB300"      # 字幕块左上角的小圆球（有文字的那种块才点一个）
ZIMU_TOUMING_DU = 200           # 字幕块底色不透明度（0-255，越小越透，255 = 实心）
ZIMU_GAO_MOREN = 32             # 时间轴上单行字幕块的默认高度
ZIMU_GAO_ZUI_XIAO = 12          # 单行字幕块最矮
ZIMU_GAO_ZUIDA = 200            # 单行字幕块最高（块是压在波形上的上层，拉多高都不压波形）
ZIMU_ZUI_DUAN_MIAO = 300        # 字幕段最短时长；比这短的丢掉（基本都是误检）
ZHEN_CHA_YUZHI = 0.02           # 跳相似帧的阈值：低于它认为和上一张一样，不存（设 0 = 全存）
JINGZHI_CHA_YUZHI = 0.002       # 静止画面阈值：和上一帧的差异低于它就当画面没动，
                                # 这一帧不重复跑检测和 OCR（画面没动 = 字幕没动，结果必然一样）
TUPIAN_ZHILIANG = 95            # 存图 JPG 质量
JIAO_BEN_HAO = "1.0.0"          # 写进 JSON 的 version
RENGONG_BIANJI = False          # 写进 JSON 的 manually_edited
KONGXIAN_HAO_MIAO = 3           # 推理线程没帧可取时的等待毫秒
JINDU_JIAN_GE_HAO_MIAO = 80     # 扫描时进度最快每多少毫秒报一次
YUJI_ZUI_SHAO_ZHEN = 30         # 至少处理这么多帧才开始估"预计完成"，免得开头几帧乱跳
QUYU_LISHI_TIAO = 10            # 框选区域的历史记多少条
ZIMU_ZHUI_SHU = 5               # OCR 去重时记住最近多少条已存文字
ZISHU_MIAO_BANJIAO = 0.5        # 字幕列表「字/秒」：半角字符算几个字（0.5 = 算半个字）
ZIMU_BIANJI_ZIHAO_MOREN = 13    # 「字幕编辑」编辑框的字号（px），开窗口就是这个大小
ZIMU_BIANJI_ZIHAO_ZUI_XIAO = 9  # Ctrl+滚轮 能缩到多小
ZIMU_BIANJI_ZIHAO_ZUIDA = 48    # Ctrl+滚轮 能放到多大
# =====================================================================


# 类别配色（问不到主界面配色时的备用）
_YANSE_BIAO = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#008080", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#fabebe",
]


# ---------------------------------------------------------------------
# 配色 / 样式（和 video_work_dialog.py 同一套，改主题时两处一起改）
# ---------------------------------------------------------------------
_YANSE_AN = {
    "zhuse": "#0A84FF", "zhuse_hover": "#409CFF",
    "beijing": "#1c1c1e", "beijing2": "#2c2c2e",
    "mian": "#2c2c2e", "mian_hover": "#3a3a3c",
    "biankuang": "#3a3a3c", "biankuang_liang": "#48484a",
    "wenzi": "#f5f5f7", "wenzi_ci": "#aeaeb2",
    "hong": "#FF453A", "qing": "#30D158", "juzi": "#FF9F0A",
}

_YANSE_LIANG = {
    "zhuse": "#0071e3", "zhuse_hover": "#0077ED",
    "beijing": "#ffffff", "beijing2": "#F9F9F9",
    "mian": "#f5f5f7", "mian_hover": "#e5e5e5",
    "biankuang": "#E5E5E5", "biankuang_liang": "#d2d2d7",
    "wenzi": "#1d1d1f", "wenzi_ci": "#86868b",
    "hong": "#FF453A", "qing": "#30D158", "juzi": "#FF9F0A",
}


def _ys():
    return _YANSE_AN if _ZHUTI == "dark" else _YANSE_LIANG


_ZHUTI = "light"


def _ys_zhu_anniu():
    c = _ys()
    return f"""
    QPushButton {{
        background-color: {c['zhuse']};
        color: #ffffff;
        border: 1px solid {c['zhuse']};
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        padding: 0 12px;
        min-height: 28px;
    }}
    QPushButton:hover {{ background-color: {c['zhuse_hover']}; }}
    QPushButton:disabled {{
        background-color: {c['mian']};
        color: {c['wenzi_ci']};
        border-color: {c['biankuang']};
    }}
    """


def _ys_ci_anniu():
    c = _ys()
    return f"""
    QPushButton {{
        background-color: {c['mian']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        font-size: 12px;
        padding: 0 10px;
        min-height: 26px;
    }}
    QPushButton:hover {{ background-color: {c['mian_hover']}; }}
    QPushButton:disabled {{
        color: {c['wenzi_ci']};
        border-color: {c['biankuang']};
    }}
    """


def _ys_shuru():
    c = _ys()
    return f"""
    QLineEdit {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 12px;
        min-height: 22px;
    }}
    QLineEdit:disabled {{ color: {c['wenzi_ci']}; }}
    """


def _ys_bianji_kuang(zihao=None):
    """编辑框样式；zihao 是字号（px），不给就用默认的"""
    if zihao is None:
        zihao = ZIMU_BIANJI_ZIHAO_MOREN
    c = _ys()
    return f"""
    QPlainTextEdit {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 4px 8px;
        font-size: {int(zihao)}px;
        font-weight: bold;
    }}
    QPlainTextEdit:disabled {{ color: {c['wenzi_ci']}; }}
    """


def _ys_xuanxiang():
    c = _ys()
    return f"""
    QCheckBox {{
        color: {c['wenzi']};
        font-size: 12px;
        spacing: 6px;
    }}
    QCheckBox:disabled {{ color: {c['wenzi_ci']}; }}
    QRadioButton {{
        color: {c['wenzi']};
        font-size: 12px;
        spacing: 6px;
    }}
    QRadioButton:disabled {{ color: {c['wenzi_ci']}; }}
    """


def _ys_shuzi():
    c = _ys()
    return f"""
    QSpinBox {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 2px 4px;
        font-size: 12px;
        min-height: 22px;
    }}
    QSpinBox:disabled {{ color: {c['wenzi_ci']}; }}
    """


def _ys_liebiao():
    c = _ys()
    return f"""
    QListWidget {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang']};
        border-radius: 6px;
        font-size: 12px;
        outline: none;
    }}
    QListWidget::item {{ padding: 3px 6px; }}
    QListWidget::item:selected {{
        background-color: {c['zhuse']};
        color: #ffffff;
    }}
    """


def _ys_zimu_biao():
    """字幕列表（ASS 字段表）：带网格、整行选中"""
    c = _ys()
    return f"""
    QTableWidget {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang']};
        border-radius: 6px;
        font-size: 12px;
        gridline-color: {c['biankuang']};
        outline: none;
    }}
    QTableWidget::item {{ padding: 2px 4px; }}
    QTableWidget::item:selected {{
        background-color: {c['zhuse']};
        color: #ffffff;
    }}
    QHeaderView::section {{
        background-color: {c['mian']};
        color: {c['wenzi_ci']};
        border: none;
        border-right: 1px solid {c['biankuang']};
        border-bottom: 1px solid {c['biankuang']};
        padding: 3px 4px;
        font-size: 11px;
    }}
    QTableCornerButton::section {{
        background-color: {c['mian']};
        border: none;
    }}
    """


def _ys_jindu():
    c = _ys()
    return f"""
    QProgressBar {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        font-size: 11px;
        text-align: center;
        min-height: 18px;
        max-height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {c['zhuse']};
        border-radius: 5px;
    }}
    """


def _ys_huadongqu():
    c = _ys()
    return f"""
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['biankuang_liang']};
        min-height: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {c['wenzi_ci']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0; border: none; background: transparent;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    """


def _biaoti_wenben(wenben):
    """小节标题"""
    t = QtWidgets.QLabel(wenben)
    f = t.font()
    f.setBold(True)
    t.setFont(f)
    return t


def _zici_wenben(wenben):
    """次要说明文字"""
    t = QtWidgets.QLabel(wenben)
    t.setObjectName("YsgHint")
    t.setWordWrap(True)
    return t


# ---------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------
def shuchu_mulu(video_lujing):
    """输出目录：视频旁边，名字 = 视频名 + 后缀"""
    juedui = osp.abspath(video_lujing)
    jiceng = osp.dirname(juedui)
    mingzi = osp.splitext(osp.basename(juedui))[0]
    return osp.join(jiceng, f"{mingzi}{MULU_HOUZHUI}")


# ---------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------
def _zhengshu_biao(ms):
    """毫秒 -> VideoSubFinder 用的那串 时_分_秒_毫秒"""
    ms = max(0, int(ms or 0))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    x = ms % 1000
    return f"{h}_{m:02d}_{s:02d}_{x:03d}"


def _zimu_wenjian_ming(qi_ms, zhi_ms, kuan, gao):
    """字幕段图名（VideoSubFinder 格式，带起止时间，后面的工具还认得）"""
    chizi = f"000000{0:04d}{0:04d}{int(kuan):04d}{int(gao):04d}"
    return f"{_zhengshu_biao(qi_ms)}__{_zhengshu_biao(zhi_ms)}_{chizi}"


def _shi_jian_wenben(ms):
    """毫秒 -> 00:00:00.000（字幕列表里显示用）"""
    ms = max(0, int(ms or 0))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    x = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{x:03d}"


def _ass_shi_jian_wenben(ms):
    """毫秒 -> 0:00:00.00（ASS 里那种时间写法，表格里显示用）"""
    ms = max(0, int(ms or 0))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    x = (ms % 1000) // 10
    return f"{h}:{m:02d}:{s:02d}.{x:02d}"


_ZISHU_BU_SUAN = "「」『』（）〈〉《》【】〔〕｛｝"
# 上面的都是成对的括号，摆字幕里只是包一层，算字数的时候不数它


def _chun_wen(wenben):
    """去掉 {...} 行内标签的干净文字（列表 / 时间轴块上只显示这个）"""
    return re.sub(r"\{[^}]*\}", "", str(wenben or "")).strip()


def _zishu_miao(qi_ms, zhi_ms, wenben):
    """字/秒：去掉 {...} 样式标签和空白之后的字数 ÷ 时长秒数

    字数怎么数：全角字（汉字、假名、全角标点）算 1 个，半角字母 / 数字算
    ZISHU_MIAO_BANJIAO 个，成对括号和半角标点不算。时长不到 1 毫秒的按 0 算，
    免得除零。
    """
    miao = (int(zhi_ms) - int(qi_ms)) / 1000.0
    if miao <= 0:
        return 0
    wen = _chun_wen(wenben)
    zi = 0.0
    for ch in wen:
        if ch.isspace() or ch in _ZISHU_BU_SUAN:
            continue
        if ord(ch) > 0x2E7F:
            zi += 1.0
        elif ch.isalnum():
            zi += ZISHU_MIAO_BANJIAO
    return int(zi / miao)


def _haoshi_wenben(miao):
    """秒 -> 12:34（不足一小时）/ 1:02:34"""
    miao = max(0, int(miao or 0))
    h = miao // 3600
    m = (miao % 3600) // 60
    s = miao % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _dui_bi_wenben(s1, s2):
    """两段文字相似度（忽略空格，日文小写假名折成大写，按字符集合比对）"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    xiao_zhuan_da = str.maketrans(
        "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ",
        "あいうえおつやゆよわアイウエオツヤユヨワ",
    )
    a = s1.replace(" ", "").replace("　", "").translate(xiao_zhuan_da)
    b = s2.replace(" ", "").replace("　", "").translate(xiao_zhuan_da)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    gong = sum((Counter(a) & Counter(b)).values())
    return gong / max(len(a), len(b))


def _gan_jing_wenben(s):
    """去掉空格，方便比较长度"""
    if not s:
        return 0
    return len(s.replace(" ", "").replace("　", ""))


# =====================================================================
# 简版时间轴（右侧下块用）：整条铺满，画字幕块 + 播放头，点一下跳过去
# =====================================================================
class ZimuZhou(QtWidgets.QWidget):
    """OCR 时间轴：整片铺满，不缩放，只用来概览字幕分布和点击跳转"""

    tiaozheng = pyqtSignal(int)     # 点了某处 -> 请求跳到这个 ms
    xuan_zhong = pyqtSignal(int)    # 点了某个字幕块 -> 第几条（-1 = 取消）

    CHIDU_GAO = 14
    ZUO_PAD = 6
    ZUI_XIAO_GAO = 46

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(self.ZUI_XIAO_GAO)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setToolTip("点字幕块 = 选中它并跳到段首；点空白 = 跳到这个位置")
        self._shichang = 0
        self._bofangtou = 0
        self._zimu = []             # [(start_ms, end_ms, wenben), ...]
        self._xuanfu = None
        self._xuan_zhong = -1       # 选中的是第几条

    # ---- 外部接口 ----
    def shezhi_shichang(self, ms):
        self._shichang = max(0, int(ms or 0))
        self.update()

    def shezhi_bofangtou(self, ms):
        self._bofangtou = max(0, int(ms or 0))
        self.update()

    def shezhi_zimu(self, zimu):
        self._zimu = list(zimu or [])
        if self._xuan_zhong >= len(self._zimu):
            self._xuan_zhong = -1
        self.update()

    def shezhi_xuan_zhong(self, xu):
        """外部（字幕列表 / 底部时间轴）选了第几条，这边跟着高亮"""
        xu = int(xu)
        if xu < 0 or xu >= len(self._zimu):
            xu = -1
        if xu != self._xuan_zhong:
            self._xuan_zhong = xu
            self.update()

    def qingkong(self):
        self._shichang = 0
        self._bofangtou = 0
        self._zimu = []
        self._xuan_zhong = -1
        self.update()

    def zimu_shu(self):
        return len(self._zimu)

    # ---- 坐标 ----
    def _guidao_qu(self):
        return QtCore.QRect(
            self.ZUO_PAD,
            self.CHIDU_GAO,
            max(0, self.width() - self.ZUO_PAD * 2),
            max(0, self.height() - self.CHIDU_GAO),
        )

    def _ms_to_x(self, ms):
        qu = self._guidao_qu()
        if self._shichang <= 0 or qu.width() <= 0:
            return qu.left()
        bi = float(ms) / float(self._shichang)
        return int(round(qu.left() + bi * qu.width()))

    def _x_to_ms(self, x):
        qu = self._guidao_qu()
        if self._shichang <= 0 or qu.width() <= 0:
            return 0
        bi = (x - qu.left()) / float(qu.width())
        return int(round(max(0.0, min(1.0, bi)) * self._shichang))

    # ---- 绘制 ----
    def paintEvent(self, event):
        c = _ys()
        huabi = QtGui.QPainter(self)
        huabi.fillRect(self.rect(), QtGui.QColor(c["beijing2"]))

        qu = self._guidao_qu()
        jing = QtGui.QColor(c["biankuang_liang"])
        huabi.fillRect(qu, jing)

        if self._shichang <= 0:
            huabi.setPen(QtGui.QColor(c["wenzi_ci"]))
            huabi.drawText(qu, Qt.AlignCenter, "（还没开始提取）")
            huabi.end()
            return

        # 字幕块：底色一律半透明蓝，选中的那块只多描一圈绿框（跟底部大时间轴一致）
        yanse = QtGui.QColor(YANSE_ZIMU)
        yanse.setAlpha(ZIMU_TOUMING_DU)
        for xu, (qi, zhi, _w) in enumerate(self._zimu):
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 <= x1:
                x2 = x1 + 2
            kuai = QtCore.QRect(x1, qu.top() + 2, x2 - x1, qu.height() - 4)
            if xu == self._xuan_zhong:
                huabi.setPen(QtGui.QPen(QtGui.QColor(YANSE_ZIMU_XUAN), 2))
            else:
                huabi.setPen(Qt.NoPen)
            huabi.setBrush(QtGui.QBrush(yanse))
            huabi.drawRect(kuai)
        huabi.setBrush(Qt.NoBrush)

        # 播放头
        x = self._ms_to_x(self._bofangtou)
        huabi.setPen(QtGui.QPen(QtGui.QColor(c["zhuse"]), 2))
        huabi.drawLine(x, qu.top(), x, qu.bottom())

        huabi.end()

    # ---- 鼠标 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._shichang > 0:
            ms = self._x_to_ms(event.pos().x())
            xu = self._zimu_xu_zai_ms(ms)
            if xu >= 0:
                # 点在字幕块上：选中它（列表那边跟着选），并跳到这一段开头
                self.shezhi_xuan_zhong(xu)
                self.xuan_zhong.emit(xu)
                self.tiaozheng.emit(int(self._zimu[xu][0]))
                event.accept()
                return
            self.tiaozheng.emit(ms)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._xuanfu = event.pos().x()
        if self._shichang > 0:
            zai = self._zai_zimu(self._x_to_ms(event.pos().x()))
            self.setToolTip(
                f"{_shi_jian_wenben(zai[0])}  {zai[2][:40]}"
                if zai
                else "点一下跳到这个位置"
            )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._xuanfu = None
        self.setToolTip("单击 = 跳到这个位置；点字幕块 = 选中它")
        return super().leaveEvent(event)

    def _zai_zimu(self, ms):
        for qi, zhi, wenben in self._zimu:
            if qi <= ms <= zhi:
                return (qi, zhi, wenben)
        return None

    def _zimu_xu_zai_ms(self, ms):
        """这个时刻落在第几条字幕上（找不到返回 -1）"""
        for xu, (qi, zhi, _w) in enumerate(self._zimu):
            if qi <= ms <= zhi:
                return xu
        return -1


# =====================================================================
# 单帧处理：检测 -> （可选）OCR 去重 -> 落盘
#   这一段完全照搬 utils/video.py 里 extract_frames_from_video 的循环体，
#   只是把"弹窗+进度对话框"换成了返回值，其余判断一字未改。
# =====================================================================
class ZhenChuli(object):
    """处理一帧；两种跑法共用同一个类，保证结果一致"""

    def __init__(self, canyu):
        c = canyu or {}
        # 区域
        self.quyu = c.get("quyu")                    # (x,y,w,h) 或 None
        # 场景检测
        self.yong_changjing = bool(c.get("changjing"))
        self.changjing_yuzhi = float(c.get("changjing_yuzhi") or 70)
        self.jiance_jiange = max(1, int(c.get("jiance_jiange") or 1))
        # 检测
        self.yong_yolo = bool(c.get("yong_yolo"))
        self.moxing = c.get("moxing")                # _OnnxYoloWrapper 或 None
        self.zhixin_du = float(c.get("zhixin_du") or 0.5)
        self.leibie_id = list(c.get("leibie_id") or [])
        self.leibie_ming = dict(c.get("leibie_ming") or {})
        self.kuaisu = bool(c.get("kuaisu"))
        # OCR
        self.yong_ocr = bool(c.get("ocr"))
        self.ocr_moxing = c.get("ocr_moxing")
        self.ocr_xiangsi = float(c.get("ocr_xiangsi") or 0.7)
        self.ocr_guolv = list(c.get("ocr_guolv") or [])
        self.ocr_youxian_chang = bool(c.get("ocr_youxian_chang"))
        # 输出
        self.shuchu_mulu = c.get("shuchu") or ""
        self.cun_tu = bool(c.get("cun_tu", True))
        self.tiao_xiangsi = bool(c.get("tiao_xiangsi", True))
        self.tiaoshi = bool(c.get("tiaoshi"))
        # 视频信息
        self.fps = float(c.get("fps") or 30.0)
        if self.fps <= 0:
            self.fps = 30.0
        self.kuan = int(c.get("kuan") or 0)
        self.gao = int(c.get("gao") or 0)
        self.zong_zhen = int(c.get("zong_zhen") or 0)

        # ---- 运行中的状态 ----
        self.shang_hui = None            # 上一帧灰度（场景检测）
        self.shang_ocr = None            # 上一次识别的文字
        self.jinqi_wenben = []           # 最近存过的文字（最多 ZIMU_ZHUI_SHU 条）
        self.dai_cun = None              # 待保存的字幕段 {qi_zhen, zhen, kuang, wenben}
        self.zimu_liebiao = []           # 已完成的字幕段 [(qi_ms, zhi_ms, wenben)]
        self.zimu_ke_jiandu = False      # 当前画面里有没有字幕
        self.shang_ci_xiao_shi_zhen = 0  # 上一次字幕消失的帧号
        self.zhen_hao = 0
        self.yi_cun = 0                  # 已存张数
        self.zimao_qi_zhen = 0           # 当前段起点帧号
        self.zimu_you_shuchu = False     # 有没有出过 SRT 条目
        self.shang_cun_zhen = None       # 上一次存下来的帧（逐帧模式跳相似帧用）
        self.jingzhi_jizhun_hui = None   # 静止判定的基准帧（64x64 灰度小图）
        self.shang_zhen_kuang = []       # 上一帧的框（画面没动时直接沿用）

    # ------------------------------------------------------------------
    # 一帧
    # ------------------------------------------------------------------
    def chuli(self, zhen_bgr, zhen_hao):
        """处理一帧

        返回 (框列表, 刚完成的字幕段)
            框列表   [(类别名, 分数, (x1,y1,x2,y2)), ...]  绝对像素
            字幕段   (起ms, 止ms, 文字) 或 None
        """
        self.zhen_hao = int(zhen_hao)

        quyu_zhen = zhen_bgr
        rx = ry = 0
        if self.quyu:
            rx, ry, rw, rh = (int(v) for v in self.quyu)
            h, w = zhen_bgr.shape[:2]
            rx = max(0, min(rx, max(0, w - 1)))
            ry = max(0, min(ry, max(0, h - 1)))
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))
            quyu_zhen = zhen_bgr[ry:ry + rh, rx:rx + rw]

        # ---------- 0. 画面根本没动：直接沿用上一帧，检测和 OCR 都不跑 ----------
        if self._hua_mian_mei_dong(quyu_zhen):
            return list(self.shang_zhen_kuang), None

        # ---------- 1. 场景检测 ----------
        changjing_bian = self._changjing(quyu_zhen)

        # ---------- 2. 检测 ----------
        kuang = []
        yolo_you = False
        yolo_pao_guo = False        # 这一帧 YOLO 实际跑了吗
        if self.yong_yolo and self.moxing is not None:
            pao = True
            if self.kuaisu and self.yong_changjing:
                # 快速模式：平时只在场景变化时跑；字幕还在时每 10 帧确认一次
                pao = changjing_bian
                if self.zimu_ke_jiandu and self.zhen_hao % 10 == 0:
                    pao = True
            if pao:
                yolo_pao_guo = True
                kuang = self._jiance(quyu_zhen, rx, ry)
                yolo_you = len(kuang) > 0

        # ---------- 3. 字幕消失？ ----------
        zi_mu_xiao_shi = False
        if (
            self.yong_yolo
            and yolo_pao_guo
            and self.zimu_ke_jiandu
            and not yolo_you
        ):
            zi_mu_xiao_shi = True
            self.shang_ci_xiao_shi_zhen = self.zhen_hao - 1
            self.zimu_ke_jiandu = False
        if yolo_you:
            self.zimu_ke_jiandu = True

        # ---------- 4. 该不该存 ----------
        if self.yong_yolo and self.yong_changjing:
            ying_cun = changjing_bian and yolo_you
        elif self.yong_yolo:
            ying_cun = yolo_you
        elif self.yong_changjing:
            ying_cun = changjing_bian
        else:
            ying_cun = False

        # ---------- 5. 分两条路走 ----------
        # 存盘一律用裁剪后的区域图；框坐标要跟着减掉区域偏移
        pian_yi = (rx, ry)
        if self.yong_ocr:
            xin_zimu = self._zou_duan_moshi(
                quyu_zhen, kuang, ying_cun, zi_mu_xiao_shi, pian_yi
            )
        else:
            self._zou_zhu_zhen(quyu_zhen, kuang, ying_cun, pian_yi)
            xin_zimu = None

        # ---------- 6. 调试图 ----------
        if self.tiaoshi:
            self._cun_tiaoshi(
                zhen_bgr, quyu_zhen, kuang, self.shang_ocr,
                changjing_bian, zi_mu_xiao_shi,
            )

        self.shang_zhen_kuang = list(kuang)
        return kuang, xin_zimu

    # ------------------------------------------------------------------
    # 画面没动？没动就把这一帧白扔掉
    # ------------------------------------------------------------------
    def _hua_mian_mei_dong(self, quyu_zhen):
        """这一帧和「上次真正处理过的那一帧」是不是一模一样

        字幕只要变了，画面必然跟着变；反过来画面纹丝没动，字幕就是没动。
        所以没动的帧再跑一遍检测 + OCR，结果必定和上一帧相同 —— 纯粹白烧时间。
        判定在 64x64 的缩略图上做，一帧不到一毫秒，换掉一次几十毫秒的识别。

        基准取的是「上次真正处理过的那一帧」，不是紧挨着的上一帧：
        画面慢慢淡入（每帧只变一点点）时，差异会一帧帧累加，
        累到超过阈值就重跑一次，绝不会因为"每帧都变得太少"而整句漏掉。
        """
        try:
            xiao = cv2.resize(
                quyu_zhen, (64, 64), interpolation=cv2.INTER_AREA
            )
            hui = cv2.cvtColor(xiao, cv2.COLOR_BGR2GRAY)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：静止判定失败 {cuowu}")
            return False

        jizhun = self.jingzhi_jizhun_hui
        if jizhun is None or jizhun.shape != hui.shape:
            self.jingzhi_jizhun_hui = hui
            return False
        try:
            cha = float(np.mean(cv2.absdiff(jizhun, hui))) / 255.0
        except Exception:  # noqa
            return False
        if cha < JINGZHI_CHA_YUZHI:
            return True                 # 没动：基准不动，接着跟后面比
        self.jingzhi_jizhun_hui = hui   # 动了：拿这一帧当新基准
        return False

    # ------------------------------------------------------------------
    # 路一：勾了 OCR = 硬字幕提取（按字幕段存图）
    # ------------------------------------------------------------------
    def _zou_duan_moshi(self, quyu_zhen, kuang, ying_cun,
                        zi_mu_xiao_shi, pian_yi):
        xin = None

        # OCR 去重
        wenben = None
        if self.ocr_moxing is not None and ying_cun:
            wenben, tiaoguo, bei_guolv = self._ocr(quyu_zhen)
            if tiaoguo:
                ying_cun = False
            if bei_guolv and self.dai_cun is not None:
                # 被规则过滤的（纯省略号之类）：先给上一句收尾
                duan = self._jie_duan(self.zhen_hao - 1)
                if duan is not None:
                    xin = duan

        # 字幕消失：立刻收尾（用消失那一刻当结束时间）
        if zi_mu_xiao_shi and self.dai_cun is not None:
            duan = self._jie_duan(self.shang_ci_xiao_shi_zhen)
            if duan is not None:
                xin = duan
            self.zimao_qi_zhen = self.zhen_hao

        # 有新字幕了：先把上一句收尾，再把这一帧记成新段的起点
        if ying_cun:
            if self.dai_cun is not None:
                if self.shang_ci_xiao_shi_zhen > self.dai_cun["qi_zhen"]:
                    jie_zhen = self.shang_ci_xiao_shi_zhen
                else:
                    jie_zhen = max(self.dai_cun["qi_zhen"], self.zhen_hao - 1)
                duan = self._jie_duan(jie_zhen)
                if duan is not None:
                    xin = duan
            if self.cun_tu:
                self.dai_cun = {
                    "qi_zhen": int(self.zimao_qi_zhen),
                    "zhen": quyu_zhen.copy(),
                    "kuang": list(kuang),
                    "wenben": wenben,
                    "pian_yi": pian_yi,
                }
            self.shang_ci_xiao_shi_zhen = 0
            self.zimao_qi_zhen = self.zhen_hao
        return xin

    # ------------------------------------------------------------------
    # 路二：没勾 OCR = 实时推理（检测到就存一张，可跳相似帧）
    # ------------------------------------------------------------------
    def _zou_zhu_zhen(self, quyu_zhen, kuang, ying_cun, pian_yi):
        if not ying_cun or not self.cun_tu or not self.shuchu_mulu:
            return
        if self.tiao_xiangsi and not self._zhen_bu_tong(quyu_zhen):
            return
        ms = int(self.zhen_hao * 1000 / self.fps)
        self._cun_yi_zhang(quyu_zhen, kuang, None, ms, ms, pian_yi)
        self.shang_cun_zhen = quyu_zhen.copy()

    def _zhen_bu_tong(self, zhen):
        """和上一张存下来的帧比，差得够多才返回 True（照搬实时推理的 0.02 阈值）"""
        if self.shang_cun_zhen is None:
            return True
        try:
            xin = cv2.resize(zhen, (64, 64))
            jiu = cv2.resize(self.shang_cun_zhen, (64, 64))
            cha = cv2.absdiff(
                cv2.cvtColor(xin, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(jiu, cv2.COLOR_BGR2GRAY),
            )
            return float(np.sum(cha)) / (64 * 64 * 255) > ZHEN_CHA_YUZHI
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：帧差计算失败 {cuowu}")
            return True

    def jie_shu(self):
        """视频跑完了：把最后一段收尾"""
        if self.dai_cun is None:
            return None
        return self._jie_duan(self.zhen_hao)

    # ------------------------------------------------------------------
    # 场景检测
    # ------------------------------------------------------------------
    def _changjing(self, quyu_zhen):
        if not self.yong_changjing:
            return True
        if self.zhen_hao % self.jiance_jiange != 0:
            return False
        gray = cv2.cvtColor(quyu_zhen, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (128, 64))
        if self.shang_hui is None:
            self.shang_hui = small
            self.zimao_qi_zhen = self.zhen_hao
            return True
        cha = cv2.absdiff(self.shang_hui, small)
        bilv = float(np.mean(cha)) / 255.0
        # 阈值越大越敏感：1 -> 差 10% 才算变；100 -> 差 0.1% 就算变
        xuqiu = 0.10 - (self.changjing_yuzhi / 100.0) * 0.099
        if bilv > xuqiu:
            self.shang_hui = small
            return True
        return False

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def _jiance(self, quyu_zhen, rx, ry):
        """跑一次检测，返回 [(类别名, 分数, 绝对像素框), ...]

        走的是和 _OnnxYoloWrapper 完全一样的三步（preprocess / inference /
        postprocess），只是不借它的壳 —— 它的 boxes 只留了框的个数，
        类别和分数拿不到，画框和写 JSON 都不够用。

        postprocess 返回 (bbox, class_ids, scores, masks, keypoints)，
        其中 bbox 已经被 scale_boxes 换算回「传进去那张图」的坐标。
        """
        kuang = []
        m = self.moxing
        if m is None:
            return kuang
        try:
            m.conf_thres = self.zhixin_du
            m.filter_classes = self.leibie_id or None
            blob = m.preprocess(quyu_zhen, upsample_mode="letterbox")
            shuchu = m.inference(blob)
            jieguo = m.postprocess(shuchu)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：检测失败 {cuowu}")
            return kuang

        if not jieguo or len(jieguo) < 3:
            return kuang
        quan, lei, fen = jieguo[0], jieguo[1], jieguo[2]
        if quan is None or len(quan) == 0:
            return kuang
        try:
            quan = np.asarray(quan, dtype=np.float64)
            lei = np.asarray(lei).reshape(-1)
            fen = np.asarray(fen, dtype=np.float64).reshape(-1)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：检测结果解不开 {cuowu}")
            return kuang

        for i in range(len(quan)):
            x1, y1, x2, y2 = (float(v) for v in quan[i][:4])
            lei_id = int(lei[i]) if i < len(lei) else -1
            mingzi = self.leibie_ming.get(lei_id, str(lei_id))
            fenshu = float(fen[i]) if i < len(fen) else 0.0
            kuang.append(
                (mingzi, fenshu, (x1 + rx, y1 + ry, x2 + rx, y2 + ry))
            )
        return kuang

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _ocr(self, quyu_zhen):
        """返回 (文字, 是否跳过, 是否被规则过滤)"""
        try:
            jieguo = self.ocr_moxing.ocr(quyu_zhen)
        except Exception as cuowu:  # noqa
            logger.warning(f"字幕提取：OCR 失败 {cuowu}")
            return None, False, False

        wenben = ""
        if jieguo and jieguo[0]:
            juzi = [
                x[1][0] if isinstance(x[1], tuple) else x[1]
                for x in jieguo[0]
            ]
            wenben = " ".join(juzi)

        ganjing = wenben.replace(" ", "").replace("　", "").strip()
        if not ganjing:
            self.shang_ocr = ""
            return wenben, True, True

        # 纯省略号（「......」「…」「。。。」）不算字幕
        kuohao = r"[「」『』【】〖〗《》〈〉（）\(\)\[\]{}\s]"
        qu_kuohao = re.sub(kuohao, "", ganjing)
        if qu_kuohao and re.match(r"^[\.。…·\・\･]+$", qu_kuohao):
            self.shang_ocr = wenben
            return wenben, True, True

        # 用户自定义过滤规则
        for guize in self.ocr_guolv:
            try:
                if re.match(guize, ganjing):
                    self.shang_ocr = wenben
                    return wenben, True, True
            except re.error:
                continue

        # 相似度去重
        xiangsi = (
            _dui_bi_wenben(wenben, self.shang_ocr) if self.shang_ocr else 0.0
        )
        zui_gao = xiangsi
        mingzhong = -1
        pi_pei = self.shang_ocr
        for xu, jiu in enumerate(self.jinqi_wenben):
            xd = _dui_bi_wenben(wenben, jiu)
            if xd > zui_gao:
                zui_gao = xd
                mingzhong = xu
                pi_pei = jiu

        if zui_gao >= self.ocr_xiangsi:
            if self.ocr_youxian_chang:
                chang = _gan_jing_wenben(wenben)
                jiu_chang = _gan_jing_wenben(pi_pei)
                if chang > jiu_chang:
                    # 这句更长，替换掉之前那条
                    if mingzhong >= 0:
                        jiu_wen = self.jinqi_wenben[mingzhong]
                        self.jinqi_wenben[mingzhong] = wenben
                        for i, (_q, _z, t) in enumerate(self.zimu_liebiao):
                            if t == jiu_wen:
                                self.zimu_liebiao[i] = (_q, _z, wenben)
                                break
                    elif self.zimu_liebiao:
                        q, z, jiu_wen = self.zimu_liebiao[-1]
                        if _dui_bi_wenben(jiu_wen, self.shang_ocr) >= self.ocr_xiangsi:
                            self.zimu_liebiao[-1] = (q, z, wenben)
                    self.shang_ocr = wenben
                    return wenben, True, False
            return wenben, True, False

        self.shang_ocr = wenben
        self.zimao_qi_zhen = self.zhen_hao
        return wenben, False, False

    # ------------------------------------------------------------------
    # 收尾一段：存图 + 存 JSON + 出 SRT 条目
    # ------------------------------------------------------------------
    def _jie_duan(self, jie_zhen):
        """把 dai_cun 这一段封口，返回 (起ms, 止ms, 文字) 或 None"""
        if self.dai_cun is None:
            return None
        qi_zhen = int(self.dai_cun["qi_zhen"])
        jie_zhen = max(qi_zhen, int(jie_zhen))
        qi_ms = int(qi_zhen * 1000 / self.fps)
        zhi_ms = int(jie_zhen * 1000 / self.fps)
        chang = zhi_ms - qi_ms
        zhen = self.dai_cun["zhen"]
        kuang = self.dai_cun["kuang"]
        wenben = self.dai_cun["wenben"]
        pian_yi = self.dai_cun.get("pian_yi", (0, 0))
        self.dai_cun = None

        if chang < ZIMU_ZUI_DUAN_MIAO:
            # 太短，基本是误检或字幕淡出
            return None

        if self.cun_tu and self.shuchu_mulu and zhen is not None:
            self._cun_yi_zhang(zhen, kuang, wenben, qi_ms, zhi_ms, pian_yi)

        if self.yong_ocr and wenben:
            self.zimu_liebiao.append((qi_ms, zhi_ms, wenben))
            self.jinqi_wenben.append(wenben)
            if len(self.jinqi_wenben) > ZIMU_ZHUI_SHU:
                self.jinqi_wenben.pop(0)
            self.zimu_you_shuchu = True
            return (qi_ms, zhi_ms, wenben)
        elif self.yong_ocr:
            # 没识别出文字，也把位置记下来（字幕块照样显示）
            return None
        return None

    def _cun_yi_zhang(self, zhen, kuang, wenben, qi_ms, zhi_ms,
                      pian_yi=(0, 0)):
        """存一张图 + 同名 JSON（图是裁剪后的区域图，框坐标相对这张图）"""
        gao, kuan = zhen.shape[:2]
        if kuan <= 0 or gao <= 0:
            return
        mingzi = _zimu_wenjian_ming(qi_ms, zhi_ms, kuan, gao)
        tu_lu = osp.join(self.shuchu_mulu, f"{mingzi}.jpeg")
        json_lu = osp.join(self.shuchu_mulu, f"{mingzi}.json")

        # --- 图 ---
        try:
            cheng, huanchong = cv2.imencode(
                ".jpg", zhen,
                [int(cv2.IMWRITE_JPEG_QUALITY), TUPIAN_ZHILIANG],
            )
            if not cheng:
                logger.error("字幕提取：图片编码失败，这一段不存")
                return
            # 用 tofile 写：cv2.imwrite 遇到中文目录会静默失败
            huanchong.tofile(tu_lu)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：存图失败 {cuowu}")
            return

        # --- JSON ---
        # OCR 文字写在每个框自己的 description 里（软件的规矩：一个框一句）
        zi = str(wenben) if wenben else None
        px, py = int(pian_yi[0]), int(pian_yi[1])
        xingzhuang = []
        for ming, fen, (x1, y1, x2, y2) in kuang or []:
            x1 = max(0.0, min(float(x1) - px, float(kuan)))
            x2 = max(0.0, min(float(x2) - px, float(kuan)))
            y1 = max(0.0, min(float(y1) - py, float(gao)))
            y2 = max(0.0, min(float(y2) - py, float(gao)))
            xingzhuang.append(
                {
                    "kie_linking": [],
                    "label": str(ming),
                    "score": float(fen),
                    "points": [
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                    ],
                    "group_id": None,
                    "description": zi,
                    "translation": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "is_edited": False,
                    "is_manually_locked": False,
                }
            )

        neirong = {
            "version": JIAO_BEN_HAO,
            "flags": {},
            "shapes": xingzhuang,
            "imagePath": osp.basename(tu_lu),
            "imageData": None,
            "imageHeight": int(gao),
            "imageWidth": int(kuan),
            "manually_edited": RENGONG_BIANJI,
            "description": "",
        }
        try:
            import json

            with open(json_lu, "w", encoding="utf-8") as f:
                json.dump(neirong, f, ensure_ascii=False, indent=2)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：存 JSON 失败 {cuowu}")

        self.yi_cun += 1

    # ------------------------------------------------------------------
    # 调试图
    # ------------------------------------------------------------------
    def _cun_tiaoshi(self, zhen, quyu_zhen, kuang, wenben,
                     changjing_bian, zi_mu_xiao_shi):
        try:
            ms = int(self.zhen_hao * 1000 / self.fps)
            jichu = _zhengshu_biao(ms)

            if kuang:
                mulu = self.shuchu_mulu + "_调试_检测"
                os.makedirs(mulu, exist_ok=True)
                tu = zhen.copy()
                for ming, fen, (x1, y1, x2, y2) in kuang:
                    cv2.rectangle(
                        tu, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 2,
                    )
                    cv2.putText(
                        tu, f"{ming} {fen:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )
                cv2.imwrite(osp.join(mulu, f"{jichu}.jpg"), tu)

            if changjing_bian and self.yong_changjing:
                mulu = self.shuchu_mulu + "_调试_场景"
                os.makedirs(mulu, exist_ok=True)
                cv2.imwrite(osp.join(mulu, f"{jichu}.jpg"), zhen)

            if wenben:
                mulu = self.shuchu_mulu + "_调试_OCR"
                os.makedirs(mulu, exist_ok=True)
                with open(
                    osp.join(mulu, f"{jichu}.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(f"OCR: {wenben}\n")
        except Exception as cuowu:  # noqa
            logger.warning(f"字幕提取：写调试文件失败 {cuowu}")

    # ------------------------------------------------------------------
    # 收尾：写 SRT / ASS
    # ------------------------------------------------------------------
    def xie_zimu_wenjian(self, shuchu_mulu):
        """写 SRT 和 ASS（放在输出文件夹外面，与它同级，沿用旧规则）"""
        if not self.zimu_liebiao:
            return []
        from anylabeling.views.labeling.utils.video import (
            ASS_TEMPLATE,
            format_ass_time,
            format_srt_time,
        )

        ming = osp.basename(shuchu_mulu)
        xie_le = []

        srt_lu = osp.join(osp.dirname(shuchu_mulu), f"{ming}.srt")
        try:
            with open(srt_lu, "w", encoding="utf-8") as f:
                for xu, (qi, zhi, wenben) in enumerate(self.zimu_liebiao, 1):
                    f.write(f"{xu}\n")
                    f.write(
                        f"{format_srt_time(qi)} --> {format_srt_time(zhi)}\n"
                    )
                    f.write(f"{wenben}\n\n")
            xie_le.append(srt_lu)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：写 SRT 失败 {cuowu}")

        ass_lu = osp.join(osp.dirname(shuchu_mulu), f"{ming}.ass")
        try:
            with open(ass_lu, "w", encoding="utf-8-sig") as f:
                f.write(ASS_TEMPLATE)
                for _xu, (qi, zhi, wenben) in enumerate(self.zimu_liebiao, 1):
                    f.write(
                        f"Dialogue: 0,{format_ass_time(qi)},"
                        f"{format_ass_time(zhi)},Default,,0,0,0,,{wenben}\n"
                    )
            xie_le.append(ass_lu)
        except Exception as cuowu:  # noqa
            logger.error(f"字幕提取：写 ASS 失败 {cuowu}")

        return xie_le


# =====================================================================
# 跑法一：跟随播放（主线程取到帧就丢进来，只处理最新一帧）
# =====================================================================
class TuiliXiancheng(QtCore.QThread):
    """跟着播放跑推理。槽位只留最新一帧，跑不过来自然丢帧，画面不卡。"""

    jieguo = pyqtSignal(int, object, object)   # 帧号, 框列表, 字幕段或 None

    def __init__(self, chuliqi, parent=None):
        super().__init__(parent)
        self.chuli = chuliqi
        self._suo = QtCore.QMutex()
        self._zui_xin = None
        self._yunxing = True

    def fang_zhen(self, zhen, zhen_hao):
        """主线程调用：把最新一帧放进槽位（旧的直接扔掉）"""
        with QtCore.QMutexLocker(self._suo):
            self._zui_xin = (zhen, int(zhen_hao))

    def run(self):
        logger.info("视频工作台：实时推理线程启动")
        while self._yunxing:
            with QtCore.QMutexLocker(self._suo):
                zui = self._zui_xin
                self._zui_xin = None
            if zui is None:
                self.msleep(KONGXIAN_HAO_MIAO)
                continue
            try:
                kuang, xin = self.chuli.chuli(zui[0], zui[1])
            except Exception as cuowu:  # noqa
                logger.error(f"视频工作台：单帧处理失败 {cuowu}")
                continue
            self.jieguo.emit(zui[1], kuang, xin)
        logger.info("视频工作台：实时推理线程结束")

    def tingzhi(self):
        self._yunxing = False
        self.wait(3000)


# =====================================================================
# 跑法二：全片扫描（自己开 cap 全速读帧，不播放）
# =====================================================================
class SaomiaoXiancheng(QtCore.QThread):
    """整条视频扫一遍。走的是和跟随播放同一份处理逻辑。"""

    jindu = pyqtSignal(int, int, int)          # 当前帧, 总帧, 已存张数
    huamian = pyqtSignal(object, int, object)  # RGB帧, 帧号, 框列表（让画面跟着走）
    jieguo = pyqtSignal(int, object, object)   # 帧号, 框列表, 字幕段
    jieshu = pyqtSignal(bool, str, object)     # 成功?, 说明, 写出的文件列表

    def __init__(self, lujing, chuliqi, xie_srt, hua_mian_bu_dong=False,
                 parent=None):
        super().__init__(parent)
        self.lujing = lujing
        self.chuli = chuliqi
        self.xie_srt = bool(xie_srt)
        self.hua_mian_bu_dong = bool(hua_mian_bu_dong)   # True = 扫描时画面静止
        self._yunxing = True

    def tingzhi(self):
        self._yunxing = False
        self.wait(5000)

    def run(self):
        logger.info(f"视频工作台：开始全片扫描 {self.lujing}")
        cap = cv2.VideoCapture(self.lujing)
        if not cap.isOpened():
            cap.release()
            self.jieshu.emit(False, "打不开这个视频", [])
            return

        zhen_hao = 0
        shang_bao = 0.0
        cuo_wu = ""
        zui_hou = None          # 最后一帧，跑完钉在画面上
        try:
            while self._yunxing:
                cheng, zhen = cap.read()
                if not cheng or zhen is None:
                    break
                try:
                    kuang, xin = self.chuli.chuli(zhen, zhen_hao)
                except Exception as cuowu:  # noqa
                    logger.error(f"视频工作台：第 {zhen_hao} 帧处理失败 {cuowu}")
                    kuang, xin = [], None
                zui_hou = (zhen, zhen_hao, kuang)
                if xin is not None:
                    self.jieguo.emit(zhen_hao, kuang, xin)

                xian = time.perf_counter()
                if xian - shang_bao >= JINDU_JIAN_GE_HAO_MIAO / 1000.0:
                    shang_bao = xian
                    self.jindu.emit(zhen_hao, self.chuli.zong_zhen,
                                    self.chuli.yi_cun)
                    # 顺手把这一帧丢给界面显示（转成 RGB，界面直接贴）
                    # 勾了"画面静止"就不发：省掉整图转 RGB + 跨线程递给界面重画
                    if not self.hua_mian_bu_dong:
                        try:
                            self.huamian.emit(
                                cv2.cvtColor(zhen, cv2.COLOR_BGR2RGB),
                                zhen_hao, kuang,
                            )
                        except Exception:  # noqa
                            pass
                zhen_hao += 1
        except Exception as cuowu:  # noqa
            cuo_wu = str(cuowu)
            logger.exception(f"视频工作台：扫描出错 {cuowu}")
        finally:
            cap.release()

        # 跑完了把最后一帧钉在画面上（节流可能刚好错过它）
        if zui_hou is not None:
            try:
                self.huamian.emit(
                    cv2.cvtColor(zui_hou[0], cv2.COLOR_BGR2RGB),
                    zui_hou[1], zui_hou[2],
                )
            except Exception:  # noqa
                pass

        # 最后一段收尾
        try:
            xin = self.chuli.jie_shu()
            if xin is not None:
                self.jieguo.emit(max(0, zhen_hao - 1), [], xin)
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：收尾失败 {cuowu}")

        # 写 SRT / ASS
        wenjian = []
        if self.xie_srt and self.chuli.shuchu_mulu:
            try:
                wenjian = self.chuli.xie_zimu_wenjian(self.chuli.shuchu_mulu)
            except Exception as cuowu:  # noqa
                logger.error(f"视频工作台：写字幕文件失败 {cuowu}")

        if cuo_wu:
            self.jieshu.emit(False, f"扫描中断：{cuo_wu}", wenjian)
        elif not self._yunxing:
            self.jieshu.emit(True, "已停止", wenjian)
        else:
            self.jieshu.emit(True, "扫描完成", wenjian)
        logger.info(
            f"视频工作台：扫描结束，共 {zhen_hao} 帧，存了 {self.chuli.yi_cun} 张"
        )


# =====================================================================
# 改一条字幕的文字（双击硬字幕列表弹的就是它）
# =====================================================================
class ZimuBianjiDialog(QtWidgets.QDialog):
    def __init__(self, qi_ms, zhi_ms, wenben, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑字幕")
        self.setMinimumWidth(480)

        bu = QtWidgets.QVBoxLayout(self)
        bu.setContentsMargins(14, 12, 14, 12)
        bu.setSpacing(8)

        shi = QtWidgets.QLabel(
            f"{_shi_jian_wenben(qi_ms)}  →  {_shi_jian_wenben(zhi_ms)}"
        )
        shi.setObjectName("YsgHint")
        bu.addWidget(shi)

        self.bianji_kuang = QtWidgets.QPlainTextEdit()
        self.bianji_kuang.setPlainText(str(wenben or ""))
        self.bianji_kuang.setMinimumHeight(150)
        bu.addWidget(self.bianji_kuang, 1)

        an = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        an.button(QtWidgets.QDialogButtonBox.Ok).setText("确定")
        an.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        an.accepted.connect(self.accept)
        an.rejected.connect(self.reject)
        bu.addWidget(an)

        self.bianji_kuang.selectAll()
        self.bianji_kuang.setFocus()

    def wenben(self):
        return self.bianji_kuang.toPlainText()


def bianji_zimu(parent, qi_ms, zhi_ms, wenben):
    """弹框改一条字幕；改了返回新文字（空串 = 清空），没改返回 None"""
    dlg = ZimuBianjiDialog(qi_ms, zhi_ms, wenben, parent)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    xin = dlg.wenben()
    if xin == str(wenben or ""):
        return None
    return xin


# =====================================================================
# 下块：OCR 的时间轴 + 字幕列表
# =====================================================================
class ZimuMianban(QtWidgets.QWidget):
    """OCR 结果：上面一条整片时间轴（字幕块），下面字幕文字列表"""

    tiaozheng = pyqtSignal(int)     # 请求跳到某个 ms
    xuan_zhong = pyqtSignal(int)    # 选中了第几条字幕（-1 = 取消）
    zimu_xiugai = pyqtSignal(int, str)  # 第几条的文字被改了 -> 外面同步

    def __init__(self, parent=None):
        super().__init__(parent)
        bu = QtWidgets.QVBoxLayout(self)
        bu.setContentsMargins(12, 10, 12, 10)
        bu.setSpacing(6)

        ding = QtWidgets.QHBoxLayout()
        ding.setSpacing(6)
        ding.addWidget(_biaoti_wenben("硬字幕提取"))
        ding.addStretch(1)
        self.shu_wenben = QtWidgets.QLabel("0 条")
        self.shu_wenben.setObjectName("YsgHint")
        ding.addWidget(self.shu_wenben)
        bu.addLayout(ding)

        self.zhou = ZimuZhou()
        self.zhou.tiaozheng.connect(self.tiaozheng.emit)
        bu.addWidget(self.zhou, 0)

        self.liebiao = QtWidgets.QListWidget()
        self.liebiao.setStyleSheet(_ys_liebiao())
        self.liebiao.setAlternatingRowColors(False)
        self.liebiao.setToolTip("单击 = 选中这一段；双击 = 改这段的文字")
        self.liebiao.itemDoubleClicked.connect(self._shuang_ji)
        self.liebiao.currentRowChanged.connect(self._xuan_zhong_bian)
        bu.addWidget(self.liebiao, 1)

        self._zimu = []

    # ---- 外部接口 ----
    def shezhi_shichang(self, ms):
        self.zhou.shezhi_shichang(ms)

    def shezhi_bofangtou(self, ms):
        self.zhou.shezhi_bofangtou(ms)

    def qingkong(self):
        self._zimu = []
        self.liebiao.clear()
        self.zhou.qingkong()
        self.shu_wenben.setText("0 条")

    def tianjia_zimu(self, qi_ms, zhi_ms, wenben):
        self._zimu.append((int(qi_ms), int(zhi_ms), str(wenben or "")))
        self._shuaxin()

    def shezhi_zimu(self, zimu):
        self._zimu = [tuple(x) for x in (zimu or [])]
        self._shuaxin()

    def zimu_shu(self):
        return len(self._zimu)

    def zimu_liebiao(self):
        """整份字幕（外面拿去做同步用）"""
        return list(self._zimu)

    def zimu_zai_ms(self, ms):
        """这个时刻落在哪一条上 -> (起, 止, 文字)；没有就 None"""
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            return None
        for qi, zhi, wenben in self._zimu:
            if qi <= ms <= zhi:
                return (qi, zhi, wenben)
        return None

    def gai_zimu(self, xu, wenben):
        """就地改第 xu 条的文字（列表 + OCR 小时间轴一起刷）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, _jiu = self._zimu[xu]
        self._zimu[xu] = (qi, zhi, str(wenben or ""))
        self._shuaxin()

    def gai_zimu_wenben(self, xu, wenben):
        """只改文字，不重建列表

        字幕编辑区每敲一个字都走这儿，重建整个列表太费（几十上百条会卡）。
        """
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, jiu = self._zimu[xu]
        wenben = str(wenben or "")
        if wenben == jiu:
            return
        self._zimu[xu] = (qi, zhi, wenben)
        xiang = self.liebiao.item(xu)
        if xiang is not None:
            xiang.setText(f"[{_shi_jian_wenben(qi)}]  {wenben}")
            xiang.setToolTip(
                f"{_shi_jian_wenben(qi)} → {_shi_jian_wenben(zhi)}\n{wenben}"
            )
        self.zhou.update()

    def gai_zimu_shijian(self, xu, qi_ms, zhi_ms):
        """就地改第 xu 条的起止时间（列表里的时间、小时间轴的块一起刷）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        _qi, _zhi, wenben = self._zimu[xu]
        qi_ms = max(0, int(qi_ms))
        zhi_ms = max(0, int(zhi_ms))
        if zhi_ms < qi_ms:
            qi_ms, zhi_ms = zhi_ms, qi_ms
        self._zimu[xu] = (qi_ms, zhi_ms, wenben)
        self._shuaxin()

    def shezhi_xuan_zhong(self, xu):
        """外部（时间轴点了块）选中第几条：列表跟着选中并滚到可见"""
        xu = int(xu)
        if xu < 0 or xu >= len(self._zimu):
            self.liebiao.blockSignals(True)
            self.liebiao.setCurrentRow(-1)
            self.liebiao.blockSignals(False)
            self.zhou.shezhi_xuan_zhong(-1)
            return
        if self.liebiao.currentRow() != xu:
            self.liebiao.blockSignals(True)
            self.liebiao.setCurrentRow(xu)
            self.liebiao.blockSignals(False)
            xiang = self.liebiao.item(xu)
            if xiang is not None:
                self.liebiao.scrollToItem(
                    xiang, QtWidgets.QAbstractItemView.EnsureVisible
                )
        self.zhou.shezhi_xuan_zhong(xu)

    # ---- 内部 ----
    def _xuan_zhong_bian(self, hang):
        """列表里点了某一条 -> 通知外面（底部时间轴跟着定位）+ 播放头跳过去"""
        if hang < 0 or hang >= len(self._zimu):
            return
        self.zhou.shezhi_xuan_zhong(hang)
        self.xuan_zhong.emit(hang)
        self.tiaozheng.emit(int(self._zimu[hang][0]))

    def _shuaxin(self):
        self.zhou.shezhi_zimu(self._zimu)
        jiu = self.liebiao.currentRow()
        self.liebiao.blockSignals(True)
        self.liebiao.clear()
        for qi, zhi, wenben in self._zimu:
            chun = _chun_wen(wenben)
            xiang = QtWidgets.QListWidgetItem(
                f"[{_shi_jian_wenben(qi)}]  {chun}"
            )
            xiang.setData(Qt.UserRole, qi)
            xiang.setToolTip(
                f"{_shi_jian_wenben(qi)} → {_shi_jian_wenben(zhi)}\n{chun}"
            )
            self.liebiao.addItem(xiang)
        if 0 <= jiu < len(self._zimu):
            self.liebiao.setCurrentRow(jiu)
        self.liebiao.blockSignals(False)
        self.shu_wenben.setText(f"{len(self._zimu)} 条")

    def _shuang_ji(self, xiang):
        if xiang is None:
            return
        self.bianji(self.liebiao.row(xiang))

    def bianji(self, xu):
        """双击某一条 -> 弹框改文字；改完更新自己并往外发信号"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, wenben = self._zimu[xu]
        xin = bianji_zimu(self, qi, zhi, wenben)
        if xin is None:
            return
        self.gai_zimu(xu, xin)
        self.zimu_xiugai.emit(xu, xin)


# =====================================================================
# 字幕编辑：上=字幕列表，下=编辑区
# =====================================================================
class _ZimuWenbenKuang(QtWidgets.QPlainTextEdit):
    """字幕编辑框：一离开焦点就说一声「这一条改完了」（好去重写字幕文件）

    Ctrl + 滚轮 = 改框里的字号（跟 Aegisub 那排字号一个意思），不带 Ctrl 的
    滚轮还是正常上下滚。字号写在样式里，不然压不住外面套的那套样式。

    回车 = 换到下一条字幕（列表往下走一行）；Shift + 回车 = 在这条字幕里换行。
    """

    likai = pyqtSignal()
    xiayitiao = pyqtSignal()        # 回车：换到下一条字幕

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zihao = ZIMU_BIANJI_ZIHAO_MOREN
        self.setStyleSheet(_ys_bianji_kuang(self._zihao))

    def zihao(self):
        return self._zihao

    def shezhi_zihao(self, zihao):
        """改字号（超范围就夹到上下限）；返回改完之后的字号"""
        zihao = int(zihao)
        zihao = max(
            ZIMU_BIANJI_ZIHAO_ZUI_XIAO,
            min(ZIMU_BIANJI_ZIHAO_ZUIDA, zihao),
        )
        if zihao != self._zihao:
            self._zihao = zihao
            self.setStyleSheet(_ys_bianji_kuang(zihao))
        return self._zihao

    def wheelEvent(self, event):
        if not (event.modifiers() & Qt.ControlModifier):
            super().wheelEvent(event)
            return
        gun = event.angleDelta().y()
        if gun == 0:
            super().wheelEvent(event)
            return
        self.shezhi_zihao(self._zihao + (1 if gun > 0 else -1))
        event.accept()

    def keyPressEvent(self, event):
        """回车 = 换到下一条字幕；Shift + 回车 = 在字幕文本里换行

        跟 Aegisub 一个手感：改完这一条敲回车就往下走，本条里要分两行显示
        （ASS 的 \\N）就按 Shift + 回车。
        """
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.xiayitiao.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.likai.emit()


class ZimuBianjiMianban(QtWidgets.QSplitter):
    """字幕编辑：上面字幕列表（一张 ASS 字段表），下面编辑区

    列表选中 = 选中时间轴上对应的字幕块；在编辑区改字 = 直接改那一条。
    回车 = 换到下一条字幕（列表往下走一行，时间轴块和播放头一起跟过去）；
    Shift + 回车 = 在这条字幕里换行（一条能有好几行）。
    上下两块中间能拖着改占比（跟「硬字幕提取」那一页一个手感）。

    列表照着 ASS 的字段来：序号 / 开始时间 / 结束时间 / 字/秒 / 样式 / 说话人 /
    文本。样式、说话人是打开字幕文件时从那行的 Style / Name 带过来的；没有原
    文件（OCR 新提取的）就是 Default 和空。
    """

    xuan_zhong = pyqtSignal(int)            # 列表里选了第几条（-1 = 取消）
    tiaozheng = pyqtSignal(int)             # 选了某条 -> 请求跳到这条的开头 ms
    wenben_gaile = pyqtSignal(int, str)     # 正在打字：第几条的文字改成了什么
    bianji_wancheng = pyqtSignal(int, str)  # 离开编辑框：这一条改完了

    LIE = ("#", "开始时间", "结束时间", "字/秒", "样式", "说话人", "文本")
    LIE_KUAN = (40, 78, 78, 44, 68, 74)     # 前几列的宽度（文本列吃掉剩下的）
    LIE_WENBEN = 6                          # 文本列是第几列
    HANG_GAO = 24                           # 一行多高

    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setHandleWidth(9)
        self.setChildrenCollapsible(False)
        self._zimu = []
        self._fujia = []        # 每条的（样式, 说话人），跟 self._zimu 一一对应
        self._xu = -1
        self._tian = False      # 正往框里塞文字，这时候的 textChanged 不算用户改

        # 上块：字幕列表
        shang = QtWidgets.QFrame()
        shang.setObjectName("YsgPanel")
        shang_bu = QtWidgets.QVBoxLayout(shang)
        shang_bu.setContentsMargins(12, 10, 12, 10)
        shang_bu.setSpacing(6)

        ding = QtWidgets.QHBoxLayout()
        ding.setSpacing(6)
        ding.addWidget(_biaoti_wenben("字幕列表"))
        ding.addStretch(1)
        self.shu_wenben = QtWidgets.QLabel("0 条")
        self.shu_wenben.setObjectName("YsgHint")
        ding.addWidget(self.shu_wenben)
        shang_bu.addLayout(ding)

        shang_bu.addWidget(self._jian_biao(), 1)

        # 下块：编辑区
        xia = QtWidgets.QFrame()
        xia.setObjectName("YsgPanel")
        xia_bu = QtWidgets.QVBoxLayout(xia)
        xia_bu.setContentsMargins(12, 10, 12, 10)
        xia_bu.setSpacing(6)

        self.shi_wenben = QtWidgets.QLabel("（没选中字幕块）")
        self.shi_wenben.setObjectName("YsgHint")
        xia_bu.addWidget(self.shi_wenben)

        self.kuang = _ZimuWenbenKuang()
        self.kuang.setStyleSheet(_ys_bianji_kuang(self.kuang.zihao()))
        self.kuang.setPlaceholderText("选中时间轴上的一条字幕，这里就能改它的文字")
        self.kuang.setMinimumHeight(80)
        self.kuang.setEnabled(False)
        self.kuang.textChanged.connect(self._wenben_bian)
        self.kuang.likai.connect(self._bianji_wancheng)
        self.kuang.xiayitiao.connect(self._xia_yi_tiao)
        xia_bu.addWidget(self.kuang, 1)

        self.addWidget(shang)
        self.addWidget(xia)
        self.setSizes([320, 240])

    def _jian_biao(self):
        """字幕列表本体：ASS 那套字段摆成一张带网格的表"""
        biao = QtWidgets.QTableWidget(0, len(self.LIE))
        self.biao = biao
        biao.setHorizontalHeaderLabels(list(self.LIE))
        biao.verticalHeader().setVisible(False)
        biao.verticalHeader().setDefaultSectionSize(self.HANG_GAO)
        biao.setShowGrid(True)
        biao.setWordWrap(False)
        biao.setAlternatingRowColors(False)
        biao.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        biao.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        biao.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        biao.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )
        biao.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )
        biao.setStyleSheet(_ys_zimu_biao())
        tou = biao.horizontalHeader()
        tou.setHighlightSections(False)
        tou.setFixedHeight(26)
        tou.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)  # 列宽能拖着改
        for lie, kuan in enumerate(self.LIE_KUAN):
            biao.setColumnWidth(lie, kuan)
        tou.setSectionResizeMode(
            self.LIE_WENBEN, QtWidgets.QHeaderView.Stretch
        )
        biao.currentCellChanged.connect(self._xuan_zhong_bian)
        return biao

    # ---- 外部接口 ----
    def shezhi_zimu(self, zimu, xuan=None, fujia=None):
        """整份字幕换掉（打开字幕 / 删除 / 合并 / 时间改完）

        fujia = 每条的（样式, 说话人），跟 zimu 一一对应；不给就全用默认。
        """
        self._zimu = [tuple(x) for x in (zimu or [])]
        self._fujia = []
        for i in range(len(self._zimu)):
            yang, shuo = "Default", ""
            if fujia is not None and i < len(fujia) and fujia[i]:
                yang = str(fujia[i][0] or "Default")
                shuo = str(fujia[i][1] or "")
            self._fujia.append((yang, shuo))
        self.biao.blockSignals(True)
        self.biao.clearContents()
        self.biao.setRowCount(len(self._zimu))
        for i, (qi, zhi, wenben) in enumerate(self._zimu):
            self._xie_hang(i, qi, zhi, wenben)
        self.biao.blockSignals(False)
        self.shu_wenben.setText(f"{len(self._zimu)} 条")
        self._xu = -1
        self.shezhi_xuan_zhong(-1 if xuan is None else int(xuan))

    def shezhi_xuan_zhong(self, xu):
        """选中第几条：列表跟着选，编辑区换成它的文字"""
        xu = int(xu)
        if xu < 0 or xu >= len(self._zimu):
            xu = -1
        self._xu = xu
        if self.biao.currentRow() != xu:
            self.biao.blockSignals(True)
            if xu < 0:
                self.biao.setCurrentCell(-1, -1)
                self.biao.clearSelection()
            else:
                self.biao.setCurrentCell(xu, 0)
            self.biao.blockSignals(False)
            if xu >= 0:
                xiang = self.biao.item(xu, 0)
                if xiang is not None:
                    self.biao.scrollToItem(
                        xiang, QtWidgets.QAbstractItemView.EnsureVisible
                    )
        wenben = "" if xu < 0 else str(self._zimu[xu][2])
        if xu < 0:
            self.shi_wenben.setText("（没选中字幕块）")
        else:
            qi, zhi = self._zimu[xu][0], self._zimu[xu][1]
            self.shi_wenben.setText(
                f"{_shi_jian_wenben(qi)} → {_shi_jian_wenben(zhi)}"
            )
        self.kuang.setEnabled(xu >= 0)
        if self.kuang.toPlainText() != wenben:
            self._tian = True
            self.kuang.setPlainText(wenben)
            self._tian = False
        # 光标落到这条的末尾：换过来接着敲字就是往后接，不会插到最前面
        self.kuang.moveCursor(QtGui.QTextCursor.End)

    def zimu_liebiao(self):
        return list(self._zimu)

    def dangqian_xu(self):
        """现在编辑的是第几条；没选中给 -1"""
        return self._xu

    def gai_wenben_ji(self, xu, wenben):
        """外面改了这一条的字（编辑区正在打字）：本地这份和列表项跟上

        编辑框里的字就是用户自己敲的，不用回填，免得光标乱跳。
        """
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, jiu = self._zimu[xu]
        wenben = str(wenben or "")
        if wenben == jiu:
            return
        self._zimu[xu] = (qi, zhi, wenben)
        self._xie_hang(xu, qi, zhi, wenben)

    # ---- 内部 ----
    def _xie_hang(self, i, qi, zhi, wenben):
        """把第 i 行那几格填上（格子还没有就新建）"""
        yang, shuo = (
            self._fujia[i] if i < len(self._fujia) else ("Default", "")
        )
        zi = (
            str(i + 1),
            _ass_shi_jian_wenben(qi),
            _ass_shi_jian_wenben(zhi),
            str(_zishu_miao(qi, zhi, wenben)),
            yang,
            shuo,
            str(wenben or "").replace("\n", "\\N"),
        )
        for lie, t in enumerate(zi):
            xiang = self.biao.item(i, lie)
            if xiang is None:
                xiang = QtWidgets.QTableWidgetItem(t)
                if lie < 4:
                    xiang.setTextAlignment(Qt.AlignCenter)
                self.biao.setItem(i, lie, xiang)
            elif xiang.text() != t:
                xiang.setText(t)
        xiang = self.biao.item(i, self.LIE_WENBEN)
        if xiang is not None:
            xiang.setToolTip(
                f"{_shi_jian_wenben(qi)} → {_shi_jian_wenben(zhi)}"
                f"\n样式 {yang} · 说话人 {shuo}"
                f"\n{wenben}"
            )

    def _xuan_zhong_bian(self, hang, _lie=None, _qian=None, _lie_qian=None):
        """列表里点了某一行 -> 外面（时间轴）跟着选中"""
        if hang < 0 or hang >= len(self._zimu):
            return
        if int(hang) == self._xu:
            return          # 同一行里换列，不算又选了一次
        self.shezhi_xuan_zhong(hang)
        self.xuan_zhong.emit(hang)
        self.tiaozheng.emit(int(self._zimu[hang][0]))

    def _xia_yi_tiao(self):
        """回车：选列表的下一行 —— 时间轴块、播放头一起跟过去

        到头了就原地不动（不往回转，也不新建）。
        """
        if self._xu < 0:
            return
        xia = self._xu + 1
        if xia >= len(self._zimu):
            return
        self.shezhi_xuan_zhong(xia)
        self.xuan_zhong.emit(xia)
        self.tiaozheng.emit(int(self._zimu[xia][0]))

    def _wenben_bian(self):
        """编辑区里打字 -> 实时往后传（不重建列表，只改这一条）"""
        if self._tian or self._xu < 0:
            return
        self.wenben_gaile.emit(self._xu, self.kuang.toPlainText())

    def _bianji_wancheng(self):
        """点走 / 焦点离开编辑框 -> 这一条改完了"""
        if self._xu < 0:
            return
        self.bianji_wancheng.emit(self._xu, self.kuang.toPlainText())


# =====================================================================
# 上块：设置（模型 / 区域 / 场景检测 / 检测 / OCR / 输出 / 跑法）
# =====================================================================
class ShezhiMianban(QtWidgets.QWidget):
    """实时推理和硬字幕提取的所有开关都在这里；勾不勾 OCR 决定是哪一件事"""

    kaishi_qingqiu = pyqtSignal(str)    # "gensui" 跟随播放 / "saomiao" 全片扫描
    tingzhi_qingqiu = pyqtSignal()
    kuang_xuan_qingqiu = pyqtSignal()   # 请求在工作台的视频画面上框选
    quyu_biangeng = pyqtSignal(object)  # 区域变了 -> 让画面区重画
    huamian_jingzhi = pyqtSignal(bool)  # 「扫描时画面静止」被改了 -> 正在扫描的话马上生效

    def __init__(self, zhu=None, parent=None):
        super().__init__(parent)
        self.zhu = zhu                  # 视频工作台，用来当弹框父窗口、取视频信息
        self.lujing = ""
        self.fps = 0.0
        self.kuan = 0
        self.gao = 0
        self.zong_zhen = 0
        self.quyu = None                # (x,y,w,h) 或 None
        self.moxing_lu = ""             # 检测模型 YAML
        self.ocr_lu = ""                # OCR 模型 YAML
        self.ocr_guolv = []
        self.zhengzai = False
        self._kaishi_shike = 0.0        # 这次跑起来的时刻（算已用 / 预计完成）
        self._jian_ui()
        self.zhuang_lishi()             # 载入上次用过的检测模型
        self.zhuang_ocr_lishi()         # 载入上次用过的 OCR 模型
        self.zhuang_quyu_lishi()        # 载入框选过的区域
        self._shuaxin_lian_dong()

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _jian_ui(self):
        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(0, 0, 0, 0)
        wai.setSpacing(0)

        gun = QtWidgets.QScrollArea()
        gun.setWidgetResizable(True)
        gun.setFrameShape(QtWidgets.QFrame.NoFrame)
        gun.setStyleSheet(_ys_huadongqu())
        gun.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wai.addWidget(gun, 1)

        nei = QtWidgets.QWidget()
        gun.setWidget(nei)
        bu = QtWidgets.QVBoxLayout(nei)
        bu.setContentsMargins(12, 10, 12, 10)
        bu.setSpacing(9)

        # ---- 标题 ----
        t = _biaoti_wenben("实时推理 / 硬字幕提取")
        bu.addWidget(t)
        bu.addWidget(_zici_wenben(
            "勾上 OCR 就是硬字幕提取（按字幕段存图 + SRT）；"
            "不勾就是实时推理（只画框 + 存图）。"
        ))

        # ---- 检测模型 ----
        bu.addWidget(_biaoti_wenben("① 检测模型（YAML）"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.a_xuan_moxing = QtWidgets.QPushButton("选择 YAML…")
        self.a_xuan_moxing.setStyleSheet(_ys_ci_anniu())
        self.a_xuan_moxing.clicked.connect(self._xuan_moxing)
        hang.addWidget(self.a_xuan_moxing)
        self.moxing_xiala = QtWidgets.QComboBox()
        self.moxing_xiala.addItem("-- 历史模型 --", "")
        self.moxing_xiala.currentIndexChanged.connect(self._lishi_moxing)
        hang.addWidget(self.moxing_xiala, 1)
        bu.addLayout(hang)
        self.moxing_wenben = _zici_wenben("未选择")
        bu.addWidget(self.moxing_wenben)

        hang2 = QtWidgets.QHBoxLayout()
        hang2.setSpacing(6)
        self.gou_yolo = QtWidgets.QCheckBox("启用检测")
        self.gou_yolo.setChecked(True)
        self.gou_yolo.setStyleSheet(_ys_xuanxiang())
        self.gou_yolo.stateChanged.connect(self._shuaxin_lian_dong)
        hang2.addWidget(self.gou_yolo)
        hang2.addStretch(1)
        hang2.addWidget(QtWidgets.QLabel("置信度"))
        self.zhixin_du = QtWidgets.QSpinBox()
        self.zhixin_du.setRange(1, 100)
        self.zhixin_du.setValue(50)
        self.zhixin_du.setSuffix("%")
        self.zhixin_du.setFixedWidth(64)
        self.zhixin_du.setStyleSheet(_ys_shuzi())
        hang2.addWidget(self.zhixin_du)
        bu.addLayout(hang2)

        hang3 = QtWidgets.QHBoxLayout()
        hang3.setSpacing(6)
        hang3.addWidget(QtWidgets.QLabel("只检测类别"))
        self.leibie_shuru = QtWidgets.QLineEdit("changfangtiao")
        self.leibie_shuru.setPlaceholderText("留空 = 全部；多个用逗号隔开")
        self.leibie_shuru.setStyleSheet(_ys_shuru())
        hang3.addWidget(self.leibie_shuru, 1)
        bu.addLayout(hang3)

        # ---- 区域 ----
        bu.addWidget(_biaoti_wenben("② 检测区域"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.a_kuang_xuan = QtWidgets.QPushButton("框选区域")
        self.a_kuang_xuan.setStyleSheet(_ys_ci_anniu())
        self.a_kuang_xuan.clicked.connect(self._kuang_xuan)
        hang.addWidget(self.a_kuang_xuan)
        self.a_quan_hua = QtWidgets.QPushButton("整图")
        self.a_quan_hua.setStyleSheet(_ys_ci_anniu())
        self.a_quan_hua.clicked.connect(self._quan_hua)
        hang.addWidget(self.a_quan_hua)
        hang.addStretch(1)
        bu.addLayout(hang)
        self.quyu_wenben = _zici_wenben("区域：全画面")
        bu.addWidget(self.quyu_wenben)
        self.quyu_xiala = QtWidgets.QComboBox()
        self.quyu_xiala.addItem("-- 历史区域 --", None)
        self.quyu_xiala.setToolTip("选一条用过的区域，直接套到画面上")
        self.quyu_xiala.currentIndexChanged.connect(self._lishi_quyu)
        bu.addWidget(self.quyu_xiala)

        # ---- 场景检测 ----
        bu.addWidget(_biaoti_wenben("③ 场景检测（跳过静止画面，快）"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.gou_changjing = QtWidgets.QCheckBox("启用")
        self.gou_changjing.setChecked(True)
        self.gou_changjing.setStyleSheet(_ys_xuanxiang())
        self.gou_changjing.stateChanged.connect(self._shuaxin_lian_dong)
        hang.addWidget(self.gou_changjing)
        hang.addStretch(1)
        hang.addWidget(QtWidgets.QLabel("灵敏度"))
        self.changjing_yuzhi = QtWidgets.QSpinBox()
        self.changjing_yuzhi.setRange(1, 100)
        self.changjing_yuzhi.setValue(70)
        self.changjing_yuzhi.setToolTip("越大越敏感：1 = 要差 10% 才算变化；100 = 差 0.1% 就算")
        self.changjing_yuzhi.setFixedWidth(58)
        self.changjing_yuzhi.setStyleSheet(_ys_shuzi())
        hang.addWidget(self.changjing_yuzhi)
        hang.addWidget(QtWidgets.QLabel("每"))
        self.jiance_jiange = QtWidgets.QSpinBox()
        self.jiance_jiange.setRange(1, 30)
        self.jiance_jiange.setValue(1)
        self.jiance_jiange.setToolTip("每 N 帧比一次；调大更快，但可能漏")
        self.jiance_jiange.setFixedWidth(48)
        self.jiance_jiange.setStyleSheet(_ys_shuzi())
        hang.addWidget(self.jiance_jiange)
        hang.addWidget(QtWidgets.QLabel("帧"))
        bu.addLayout(hang)

        # ---- OCR ----
        bu.addWidget(_biaoti_wenben("④ OCR（勾上 = 硬字幕提取）"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.gou_ocr = QtWidgets.QCheckBox("启用 OCR 去重")
        self.gou_ocr.setStyleSheet(_ys_xuanxiang())
        self.gou_ocr.stateChanged.connect(self._shuaxin_lian_dong)
        hang.addWidget(self.gou_ocr)
        self.a_guolv = QtWidgets.QPushButton("过滤规则…")
        self.a_guolv.setStyleSheet(_ys_ci_anniu())
        self.a_guolv.clicked.connect(self._bian_guolv)
        hang.addWidget(self.a_guolv)
        self.gou_youxian_chang = QtWidgets.QCheckBox("优先长文本")
        self.gou_youxian_chang.setChecked(True)
        self.gou_youxian_chang.setStyleSheet(_ys_xuanxiang())
        self.gou_youxian_chang.setToolTip(
            "新文字包含旧文字时用新的：OCR 不会凭空多出字，长的那条更可靠"
        )
        hang.addWidget(self.gou_youxian_chang)
        hang.addStretch(1)
        hang.addWidget(QtWidgets.QLabel("相似度"))
        self.ocr_xiangsi = QtWidgets.QSpinBox()
        self.ocr_xiangsi.setRange(50, 100)
        self.ocr_xiangsi.setValue(70)
        self.ocr_xiangsi.setSuffix("%")
        self.ocr_xiangsi.setFixedWidth(64)
        self.ocr_xiangsi.setStyleSheet(_ys_shuzi())
        hang.addWidget(self.ocr_xiangsi)
        bu.addLayout(hang)

        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.a_xuan_ocr = QtWidgets.QPushButton("选择 OCR YAML…")
        self.a_xuan_ocr.setStyleSheet(_ys_ci_anniu())
        self.a_xuan_ocr.clicked.connect(self._xuan_ocr)
        hang.addWidget(self.a_xuan_ocr)
        self.ocr_xiala = QtWidgets.QComboBox()
        self.ocr_xiala.addItem("-- 历史路径 --", "")
        self.ocr_xiala.currentIndexChanged.connect(self._lishi_ocr)
        hang.addWidget(self.ocr_xiala, 1)
        bu.addLayout(hang)
        self.ocr_wenben = _zici_wenben("未选择")
        bu.addWidget(self.ocr_wenben)

        # ---- 其它 ----
        bu.addWidget(_biaoti_wenben("⑤ 其它"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(10)
        self.gou_kuaisu = QtWidgets.QCheckBox("快速模式")
        self.gou_kuaisu.setChecked(True)
        self.gou_kuaisu.setStyleSheet(_ys_xuanxiang())
        self.gou_kuaisu.setToolTip("只在场景变化时才跑检测，快很多")
        hang.addWidget(self.gou_kuaisu)
        self.gou_tiaoxiangsi = QtWidgets.QCheckBox("跳过相似帧")
        self.gou_tiaoxiangsi.setChecked(True)
        self.gou_tiaoxiangsi.setStyleSheet(_ys_xuanxiang())
        self.gou_tiaoxiangsi.setToolTip(
            "只在「不勾 OCR」时有意义：相邻帧几乎一样就不重复存图"
        )
        hang.addWidget(self.gou_tiaoxiangsi)
        self.gou_tiaoshi = QtWidgets.QCheckBox("调试模式")
        self.gou_tiaoshi.setStyleSheet(_ys_xuanxiang())
        self.gou_tiaoshi.setToolTip("额外输出检测图 / 场景图 / OCR 文字，排查问题时用")
        hang.addWidget(self.gou_tiaoshi)
        hang.addStretch(1)
        bu.addLayout(hang)

        # ---- 输出 ----
        bu.addWidget(_biaoti_wenben("⑥ 输出目录"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.shuchu_shuru = QtWidgets.QLineEdit()
        self.shuchu_shuru.setStyleSheet(_ys_shuru())
        hang.addWidget(self.shuchu_shuru, 1)
        self.a_shuchu = QtWidgets.QPushButton("浏览…")
        self.a_shuchu.setStyleSheet(_ys_ci_anniu())
        self.a_shuchu.clicked.connect(self._xuan_shuchu)
        hang.addWidget(self.a_shuchu)
        bu.addLayout(hang)
        bu.addWidget(_zici_wenben("每张图都会配一个同名 JSON（框 + OCR 文字）"))

        # ---- 跑法 ----
        bu.addWidget(_biaoti_wenben("⑦ 怎么跑"))
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(10)
        self.xuan_gensui = QtWidgets.QRadioButton("跟随播放")
        self.xuan_gensui.setChecked(True)
        self.xuan_gensui.setStyleSheet(_ys_xuanxiang())
        self.xuan_gensui.setToolTip("边放边推理，看框的效果；跑不过来会自动丢帧")
        hang.addWidget(self.xuan_gensui)
        self.xuan_saomiao = QtWidgets.QRadioButton("全片扫描")
        self.xuan_saomiao.setStyleSheet(_ys_xuanxiang())
        self.xuan_saomiao.setToolTip("不播放，整条视频跑一遍，跑完出图 + SRT")
        self.xuan_saomiao.toggled.connect(self._shuaxin_lian_dong)
        hang.addWidget(self.xuan_saomiao)
        hang.addStretch(1)
        bu.addLayout(hang)

        # 扫描时画面静止：省掉"整图转 RGB + 递给界面重画"的开销
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.gou_huamian_budong = QtWidgets.QCheckBox("扫描时画面静止")
        self.gou_huamian_budong.setStyleSheet(_ys_xuanxiang())
        self.gou_huamian_budong.setToolTip(
            "全片扫描时画面不再一帧一帧跟着刷，把刷新的开销省下来换速度。\n"
            "正在扫描时也能随时改，改完马上生效，不用停下重来。\n"
            "进度条和已用/预计时间照常走，跑完把最后一帧钉在画面上。"
        )
        self.gou_huamian_budong.toggled.connect(self.huamian_jingzhi)
        hang.addWidget(self.gou_huamian_budong)
        hang.addStretch(1)
        bu.addLayout(hang)

        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(6)
        self.a_kaishi = QtWidgets.QPushButton("开始")
        self.a_kaishi.setStyleSheet(_ys_zhu_anniu())
        self.a_kaishi.clicked.connect(self._dian_kaishi)
        hang.addWidget(self.a_kaishi)
        self.jindu = QtWidgets.QProgressBar()
        self.jindu.setRange(0, 1000)
        self.jindu.setValue(0)
        self.jindu.setTextVisible(False)
        self.jindu.setStyleSheet(_ys_jindu())
        hang.addWidget(self.jindu, 1)
        bu.addLayout(hang)

        self.zhuangtai = _zici_wenben("未开始")
        self.zhuangtai.setWordWrap(True)
        bu.addWidget(self.zhuangtai)
        bu.addStretch(1)

        self._shuaxin_lian_dong()

    # ------------------------------------------------------------------
    # 联动
    # ------------------------------------------------------------------
    def _zhuang_lian_dong(self):
        """锁定 / 解锁（锁的是"清空视频后不能点开始"这类，不是运行中锁定）"""
        you = bool(self.lujing)
        # 开始/停止是同一个按钮，跑起来之后它就是停止按钮，任何时候都得能点
        self.a_kaishi.setEnabled(you)
        for w in (self.a_xuan_moxing, self.a_kuang_xuan, self.quyu_xiala,
                  self.a_quan_hua, self.a_xuan_ocr):
            w.setEnabled(you and not self.zhengzai)

    def _shuaxin_lian_dong(self):
        huo = not self.zhengzai
        chang = self.gou_changjing.isChecked()
        self.changjing_yuzhi.setEnabled(chang and huo)
        self.jiance_jiange.setEnabled(chang and huo)

        yolo = self.gou_yolo.isChecked()
        self.zhixin_du.setEnabled(yolo and huo)
        self.leibie_shuru.setEnabled(yolo and huo)
        self.a_xuan_moxing.setEnabled(huo)
        self.moxing_xiala.setEnabled(huo)

        ocr = self.gou_ocr.isChecked()
        for w in (self.ocr_xiangsi, self.a_guolv, self.gou_youxian_chang,
                  self.a_xuan_ocr, self.ocr_xiala):
            w.setEnabled(ocr and huo)
        # 跳相似帧只在"不勾 OCR"（实时推理）时有意义
        self.gou_tiaoxiangsi.setEnabled((not ocr) and huo)

        for w in (self.gou_kuaisu, self.gou_tiaoshi, self.a_shuchu,
                  self.shuchu_shuru, self.quyu_xiala,
                  self.xuan_gensui, self.xuan_saomiao):
            w.setEnabled(huo)
        # 画面动不动只有全片扫描用得上，跟随播放时它是灰的。
        # 扫描过程中也一直能点：改完马上作用到正在跑的扫描，不用停下重来。
        self.gou_huamian_budong.setEnabled(self.xuan_saomiao.isChecked())
        self._zhuang_lian_dong()

    def _suoding(self, suoding):
        self.zhengzai = bool(suoding)
        self.a_kaishi.setText("停止" if suoding else "开始")
        self._shuaxin_lian_dong()

    # ------------------------------------------------------------------
    # 视频
    # ------------------------------------------------------------------
    def shezhi_video(self, lujing, fps, kuan, gao, zong_zhen):
        if not lujing or lujing == self.lujing:
            return
        self.lujing = lujing
        self.fps = float(fps or 0.0)
        self.kuan = int(kuan or 0)
        self.gao = int(gao or 0)
        self.zong_zhen = int(zong_zhen or 0)
        self.quyu = None
        self.quyu_wenben.setText("区域：全画面")
        self.quyu_biangeng.emit(None)
        # 换片子了：下拉回到"-- 历史区域 --"，但条目留着
        self.quyu_xiala.blockSignals(True)
        self.quyu_xiala.setCurrentIndex(0)
        self.quyu_xiala.blockSignals(False)
        self.shuchu_shuru.setText(shuchu_mulu(lujing))
        self.qingkong_jindu()
        self.zhuangtai.setText("未开始")
        self._shuaxin_lian_dong()

    def qingkong_jindu(self):
        self.jindu.setValue(0)
        self._kaishi_shike = 0.0

    def jindu_gengxin(self, zhen_hao, zong_zhen, yi_cun):
        if zong_zhen > 0:
            self.jindu.setValue(
                max(0, min(1000, int(zhen_hao / float(zong_zhen) * 1000)))
            )
        wen = f"已处理 {zhen_hao}/{zong_zhen} 帧 · 已存 {yi_cun} 张"

        # 已用时间 + 预计完成时刻
        if self._kaishi_shike > 0:
            yong = max(0.0, time.time() - self._kaishi_shike)
            wen += f" · 已用 {_haoshi_wenben(yong)}"
            sheng = max(0, int(zong_zhen) - int(zhen_hao))
            if sheng <= 0:
                wen += " · 已完成"
            elif zhen_hao >= YUJI_ZUI_SHAO_ZHEN:
                yuji = yong / float(zhen_hao) * sheng
                wancheng = time.strftime(
                    "%H:%M:%S", time.localtime(time.time() + yuji)
                )
                wen += f" · 预计 {wancheng} 完成（还需 {_haoshi_wenben(yuji)}）"
        self.zhuangtai.setText(wen)

    def zhuangtai_shezhi(self, wenben):
        self.zhuangtai.setText(wenben)

    # ------------------------------------------------------------------
    # 参数快照
    # ------------------------------------------------------------------
    def canyu(self):
        return {
            "quyu": self.quyu,
            "changjing": self.gou_changjing.isChecked(),
            "changjing_yuzhi": self.changjing_yuzhi.value(),
            "jiance_jiange": self.jiance_jiange.value(),
            "yong_yolo": self.gou_yolo.isChecked(),
            "zhixin_du": self.zhixin_du.value() / 100.0,
            "leibie": self.leibie_shuru.text().strip(),
            "kuaisu": self.gou_kuaisu.isChecked(),
            "ocr": self.gou_ocr.isChecked(),
            "ocr_xiangsi": self.ocr_xiangsi.value() / 100.0,
            "ocr_guolv": list(self.ocr_guolv),
            "ocr_youxian_chang": self.gou_youxian_chang.isChecked(),
            "cun_tu": True,
            "tiao_xiangsi": self.gou_tiaoxiangsi.isChecked(),
            "huamian_budong": self.gou_huamian_budong.isChecked(),
            "tiaoshi": self.gou_tiaoshi.isChecked(),
            "shuchu": self.shuchu_shuru.text().strip(),
            "fps": self.fps,
            "kuan": self.kuan,
            "gao": self.gao,
            "zong_zhen": self.zong_zhen,
            "moxing_lu": self.moxing_lu,
            "ocr_lu": self.ocr_lu,
        }

    # ------------------------------------------------------------------
    # 选择模型 / 区域 / 目录
    # ------------------------------------------------------------------
    def _xuan_moxing(self):
        lujing, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择检测模型 YAML", "",
            "YAML 配置 (*.yaml *.yml);;所有文件 (*)",
        )
        if not lujing:
            return
        self._she_moxing(lujing, ji_lishi=True)

    def _she_moxing(self, lujing, ji_lishi=False):
        self.moxing_lu = lujing
        self.moxing_wenben.setText(f"已选：{osp.basename(lujing)}")
        self.moxing_wenben.setToolTip(lujing)
        if ji_lishi:
            self._ji_lishi_moxing(lujing)

    def _ji_lishi_moxing(self, lujing):
        lishi = self._du_lishi_moxing()
        if lujing in lishi:
            lishi.remove(lujing)
        lishi.insert(0, lujing)
        lishi = lishi[:10]
        try:
            import json

            with open(self._lishi_wenjian(), "w", encoding="utf-8") as f:
                json.dump(lishi, f, ensure_ascii=False, indent=2)
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：存模型历史失败 {cuowu}")
        self._zhuang_lishi_moxing(lishi)

    @staticmethod
    def _lishi_wenjian():
        return osp.join(osp.expanduser("~"), ".ysg_video_jiance_moxing.json")

    def _du_lishi_moxing(self):
        try:
            import json

            lu = self._lishi_wenjian()
            if not osp.exists(lu):
                return []
            with open(lu, "r", encoding="utf-8") as f:
                return [p for p in json.load(f) if osp.exists(p)]
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：读模型历史失败 {cuowu}")
            return []

    def _zhuang_lishi_moxing(self, lishi):
        self.moxing_xiala.blockSignals(True)
        self.moxing_xiala.clear()
        self.moxing_xiala.addItem("-- 历史模型 --", "")
        for p in lishi:
            self.moxing_xiala.addItem(osp.basename(p), p)
        self.moxing_xiala.setCurrentIndex(0)
        self.moxing_xiala.blockSignals(False)

    def _lishi_moxing(self, _xu):
        p = self.moxing_xiala.currentData()
        if p:
            self._she_moxing(p)

    def zhuang_lishi(self):
        """填历史下拉，并默认选中最近用过的那个"""
        lishi = self._du_lishi_moxing()
        self._zhuang_lishi_moxing(lishi)
        if lishi:
            self.moxing_xiala.setCurrentIndex(1)
            self._she_moxing(lishi[0])

    def _xuan_ocr(self):
        lujing, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 OCR 模型 YAML", "",
            "YAML 配置 (*.yaml *.yml);;所有文件 (*)",
        )
        if not lujing:
            return
        self._she_ocr(lujing, ji_lishi=True)

    def _she_ocr(self, lujing, ji_lishi=False):
        self.ocr_lu = lujing
        self.ocr_wenben.setText(f"已选：{osp.basename(lujing)}")
        self.ocr_wenben.setToolTip(lujing)
        if ji_lishi:
            try:
                from anylabeling.views.labeling.utils.video import (
                    save_ocr_path_history,
                )

                save_ocr_path_history(lujing)
            except Exception as cuowu:  # noqa
                logger.warning(f"视频工作台：存 OCR 历史失败 {cuowu}")

    def _lishi_ocr(self, _xu):
        p = self.ocr_xiala.currentData()
        if p:
            self._she_ocr(p)

    def zhuang_ocr_lishi(self):
        try:
            from anylabeling.views.labeling.utils.video import (
                load_ocr_path_history,
            )

            lishi = load_ocr_path_history()
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：读 OCR 历史失败 {cuowu}")
            lishi = []
        self.ocr_xiala.blockSignals(True)
        self.ocr_xiala.clear()
        self.ocr_xiala.addItem("-- 历史路径 --", "")
        for p in lishi:
            self.ocr_xiala.addItem(osp.basename(p), p)
        self.ocr_xiala.setCurrentIndex(0)
        self.ocr_xiala.blockSignals(False)
        if lishi:
            self.ocr_xiala.setCurrentIndex(1)
            self._she_ocr(lishi[0])

    def _kuang_xuan(self):
        """在工作台的视频画面上框选检测区域（不弹窗）"""
        if not self.lujing or self.kuan <= 0 or self.gao <= 0:
            QtWidgets.QMessageBox.warning(self, "提示", "先打开一个视频")
            return
        self.kuang_xuan_qingqiu.emit()

    def kuang_xuan_zhuangtai(self, kaishi):
        """框选开始了 / 结束了 -> 换按钮文字"""
        self.a_kuang_xuan.setText("退出框选" if kaishi else "框选区域")

    def shezhi_quyu(self, quyu, ji_lishi=True):
        """画面区框好了 -> 更新这里记的区域和文字"""
        if quyu:
            self.quyu = tuple(int(v) for v in quyu)
            x, y, w, h = self.quyu
            self.quyu_wenben.setText(f"区域：({x},{y}) {w}x{h}")
            if ji_lishi:
                self._ji_lishi_quyu(self.quyu)
        else:
            self.quyu = None
            self.quyu_wenben.setText("区域：全画面")

    def _quan_hua(self):
        self.quyu = None
        self.quyu_wenben.setText("区域：全画面")
        self.quyu_biangeng.emit(None)

    # ------------------------------------------------------------------
    # 框选区域的历史（跟模型历史一个路子，存在用户主目录）
    # ------------------------------------------------------------------
    @staticmethod
    def _quyu_lishi_wenjian():
        return osp.join(osp.expanduser("~"), ".ysg_video_quyu_lishi.json")

    def _du_lishi_quyu(self):
        try:
            import json

            lu = self._quyu_lishi_wenjian()
            if not osp.exists(lu):
                return []
            with open(lu, "r", encoding="utf-8") as f:
                shu = json.load(f)
            gui = []
            for x in shu or []:
                if isinstance(x, (list, tuple)) and len(x) == 4:
                    gui.append(tuple(int(v) for v in x))
            return gui[:QUYU_LISHI_TIAO]
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：读区域历史失败 {cuowu}")
            return []

    def _ji_lishi_quyu(self, quyu):
        qu = tuple(int(v) for v in quyu)
        lishi = [x for x in self._du_lishi_quyu() if x != qu]
        lishi.insert(0, qu)
        lishi = lishi[:QUYU_LISHI_TIAO]
        try:
            import json

            with open(
                self._quyu_lishi_wenjian(), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    [list(x) for x in lishi], f,
                    ensure_ascii=False, indent=2,
                )
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：存区域历史失败 {cuowu}")
        self._zhuang_lishi_quyu(lishi)

    def _zhuang_lishi_quyu(self, lishi):
        """把历史填进下拉；不碰当前选中的区域"""
        self.quyu_xiala.blockSignals(True)
        self.quyu_xiala.clear()
        self.quyu_xiala.addItem("-- 历史区域 --", None)
        for x, y, w, h in lishi:
            self.quyu_xiala.addItem(f"{x},{y}  {w}x{h}", (x, y, w, h))
        self.quyu_xiala.setCurrentIndex(0)
        self.quyu_xiala.blockSignals(False)

    def zhuang_quyu_lishi(self):
        self._zhuang_lishi_quyu(self._du_lishi_quyu())

    def _lishi_quyu(self, xu):
        """从历史里选了一条 -> 直接套到画面上"""
        if xu <= 0:
            return
        qu = self.quyu_xiala.itemData(xu)
        if not qu:
            return
        self.shezhi_quyu(qu, ji_lishi=False)
        self.quyu_biangeng.emit(self.quyu)

    def _bian_guolv(self):
        try:
            from anylabeling.views.labeling.utils.video import (
                OcrFilterDialog,
                load_ocr_filter_rules,
            )

            if not self.ocr_guolv:
                self.ocr_guolv = list(load_ocr_filter_rules() or [])
            dlg = OcrFilterDialog(self, self.ocr_guolv)
            if dlg.exec_():
                self.ocr_guolv = list(dlg.get_rules() or [])
                logger.info(f"视频工作台：OCR 过滤规则 {len(self.ocr_guolv)} 条")
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：过滤规则打不开 {cuowu}")
            QtWidgets.QMessageBox.warning(self, "提示", f"打不开：{cuowu}")

    def _xuan_shuchu(self):
        qi = self.shuchu_shuru.text().strip() or osp.expanduser("~")
        mulu = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择输出目录", qi
        )
        if mulu:
            self.shuchu_shuru.setText(mulu)

    # ------------------------------------------------------------------
    # 开始 / 停止
    # ------------------------------------------------------------------
    def _dian_kaishi(self):
        if self.zhengzai:
            self.tingzhi_qingqiu.emit()
            return

        if not self.lujing:
            QtWidgets.QMessageBox.warning(self, "提示", "先打开一个视频")
            return
        if not self.gou_yolo.isChecked() and not self.gou_changjing.isChecked():
            QtWidgets.QMessageBox.warning(
                self, "提示", "「启用检测」和「场景检测」至少勾一个"
            )
            return
        if self.gou_yolo.isChecked() and not self.moxing_lu:
            QtWidgets.QMessageBox.warning(
                self, "提示", "先选检测模型的 YAML"
            )
            return
        if self.gou_ocr.isChecked() and not self.ocr_lu:
            QtWidgets.QMessageBox.warning(
                self, "提示", "勾了 OCR，就得先选 OCR 模型的 YAML"
            )
            return
        if not self.shuchu_shuru.text().strip():
            QtWidgets.QMessageBox.warning(self, "提示", "先选输出目录")
            return

        paofa = "gensui" if self.xuan_gensui.isChecked() else "saomiao"
        self._kaishi_shike = time.time()
        self.kaishi_qingqiu.emit(paofa)
