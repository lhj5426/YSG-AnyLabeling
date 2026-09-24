# -*- coding: utf-8 -*-
"""视频工作台（打开视频后进入的界面）

界面结构：
    顶部    文件名 + 分辨率/帧率信息 + 右上角功能按钮
    中间    视频画面（左） + 参数区（右）      两块之间可拖拽
    底部    时间轴（刻度尺 + 音频波形 + 播放头）  与上面之间可拖拽

播放内核：
    画面 = OpenCV 逐帧读取（能画检测框、能直接喂模型、能逐帧步进）
        取帧线程只把"最新一帧"放在一个格子里，主线程定时来取走。
        不逐帧往主线程扔大图 —— 那样事件队列会越堆越长，帧率掉一半
        且画面越放越延迟。现在主线程慢了就自然丢帧，时间轴不出错。
        缩放和 BGR->RGB 都在取帧线程里做完，主线程只负责贴图。
        播放时临时把系统定时器精度提到 1ms（不然 msleep 会多睡十几毫秒，
        画面会一帧帧变慢，看着像慢动作）。
    声音 = Qt 播放器只出声不出画（mp4 内嵌音轨 / 外挂 wav 都能播）
    波形 = ffmpeg 把音轨解成 8kHz 单声道，算每毫秒的振幅包络
        解析时每块之间会歇一下（JIE_LIU_HAO_MIAO），不然十几秒内
        一个核被吃满，打开视频的头几秒画面会卡。

入口：文件 -> 打开视频文件
"""

import ctypes
import json
import math
import os
import os.path as osp
import re
import shutil
import struct
import subprocess
import time
from collections import OrderedDict

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets.video_infer_panel import (
    CAOWEI_SHU,
    YANSE_KUANG,
    YANSE_KUANG_ZI,
    YANSE_ZIMU,
    YANSE_ZIMU_BIAN_A,
    YANSE_ZIMU_BIAN_B,
    YANSE_ZIMU_QIU,
    YANSE_ZIMU_XUAN,
    YANSE_ZIMU_ZAI,
    YANSE_ZIMU_ZI,
    ZIMU_GAO_MOREN,
    ZIMU_GAO_ZUI_XIAO,
    ZIMU_GAO_ZUIDA,
    ZIMU_TOUMING_DU,
    SaomiaoXiancheng,
    ShezhiMianban,
    TuiliXiancheng,
    YangshiBianjiDialog,
    ZhenChuli,
    ZimuMianban,
    ZimuBianjiMianban,
    buju_qsettings,
    du_zidonghua_peizhi,
    zimu_charu_weizhi,
)
from anylabeling.views.labeling.widgets.video_work_search import (
    SuoSuoDialog,
    XuanZeDialog,
)

try:
    from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

    _YOU_YINPIN = True
except Exception:  # noqa
    _YOU_YINPIN = False


# =====================================================================
# 用户可调参数（只改这一段）
# =====================================================================
ZHUTI = "light"                   # 界面主题：dark 深色 / light 浅色
SHIJIANZHOU_GAO = 130             # 时间轴（波形）初始高度，之后可以拖着改
SUOFANG_ZUIDA = 500.0             # 时间轴最大放大倍数（1 = 整条铺满）
SUOFANG_ZUI_XIAO = 1.0            # 时间轴最小放大倍数（1 = 整条铺满）
SUOFANG_MOREN = 15.0              # 打开视频时时间轴的默认放大倍数（1 = 整条铺满，字幕块会缩成一小条看不清）
BO_CAN_YANG_LV = 8000             # 波形解码采样率（单声道）
BO_HAO_MIAO = 1                   # 波形每格 = 多少毫秒（越细越清晰、越费内存）
JIE_LIU_HAO_MIAO = 12             # 解析音频时每块之间歇多少毫秒（0 = 不歇）
BO_WANCHENG_MIAO = 1200           # 波形解完后"波形就绪"在带上停留多久（毫秒）
BO_ZHONG_GAO = 1                  # 波形带中线的粗细（像素）
BEISU_LIEBIAO = [                     # 倍速下拉里可选的倍速（跟 Arctime 一样，从快到慢排）
    4.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25,
]
MO_REN_BEISU = 1.0                # 打开视频时的默认倍速
BEISU_KUANG_KUAN = 56             # 倍速框宽度：嫌框大就往小改（比如 48），字被裁就往大改（比如 62）
MO_REN_YINLIANG = 70              # 默认音量 0-100
JINGYIN = False                   # 是否静音启动
FANGXIANG_JIAN_GE_MIAO = 0.5      # 方向键：隔这么久没按就当新的一串，重新从当前位置起步
DINGWEI_SHOUWEI_MIAO = 1.0        # 定位守卫：等这么久还没等到要的那一帧，就当它不会来了，别再拦着
GUNDONG_BU_BI = 0.1               # 时间轴上滚一格滚轮，视野往左右挪多少（占一屏的比例）
BO_FANGDA_MOREN = 1.0             # 波形振幅缩放：1 = 原样，越大波形越饱满
BO_FANGDA_ZUI_XIAO = 0.2          # 波形振幅缩放最小倍数
BO_FANGDA_ZUIDA = 50.0            # 波形振幅缩放最大倍数
YANSE_QUYU = "#00E5FF"            # 检测区域框颜色（在视频画面上框选时）
YANSE_CHONGDIE = "#FFC107"        # 字幕块摞在一起时，时间轴顶端那个 L 形标记的颜色
CHONGDIE_L_GAO = 26               # 那个 L 形标记的竖线往上竖多高（竖到时间轴的刻度条里）
CHONGDIE_SHEN_ALPHA = 75          # 摞在一起的那一段压深多少（0-255，越大越深）
ZIMU_LIEBIAO_GAO = 280            # 「字幕列表」被换到下方时，那一栏的初始高度
WEIZHI_PEIZHI_JIAN = "jiemian/liebiao_zai_xia"   # 记住"谁在下面"用的配置项名
JIEMIAN_ZHENGGE_JIAN = "jiemian/chuangkou_weizhi"    # 记住窗口位置 + 大小
JIEMIAN_TAB_JIAN = "jiemian/tab_ye"                  # 记住右侧停在哪个标签页
JIEMIAN_BILI_JIAN = "jiemian/fenlan_bili"            # 记住各处拖的分栏位置
JIEMIAN_SHANGXIA_DUO = "jiemian/shangxia_tuoguo"     # 上下大分栏你自己拖过没有
JIEMIAN_SUOFANG_JIAN = "jiemian/shijianzhou_suofang"  # 记住时间轴的放大倍数
JIEMIAN_MEIHANG_GAO = "jiemian/meihang_gao"          # 记住字幕块每行多高
JIEMIAN_BO_FANGDA = "jiemian/bo_fangda"              # 记住波形振幅放大倍数
JIEMIAN_ZIMU_TIAO_JIAN = "jiemian/zimu_tiao"         # 记住画面下方那条独立字幕条开不开
JIEMIAN_GEN_TIAO = "jiemian/gen_tiao"                # 记住「选中字幕时画面跟着跳」开关
JIEMIAN_SHISHI_GUN = "jiemian/shishi_gun"            # 记住「时间轴实时滚动」开关
CHEXIAO_ZUIDA = 100               # 撤销栈最多留几版（照 AEG 的 Limits/Undo Levels）
CHEXIAO_HEBING_MIAO = 1.5         # 同一类动作连着做，隔这么近就算一笔（长按挪块不刷爆栈）
ZIMU_HOUZHUI = (".srt", ".ass", ".ssa")   # 认得的字幕文件后缀（拖进窗口就载入）
ZIMU_TIAO_GAO = 34                # 画面下方那条字幕对照条的高度
ZIMU_TIAO_ZIHAO = 15              # 字幕对照条的字号
XINJIAN_ZIMU_MS = 1000            # 没选中字幕块时按回车：以播放头为起点新建的字幕块多长（毫秒）
ZIDONG_BAOCUN_MIAO = 60           # 自动保存间隔（秒）：隔这么久、字幕又变过，就存一份带时间戳的进「自动保存」
ZIDONG_BAOCUN_JIA = "自动保存"     # 自动保存文件夹名（跟字幕文件建在同一层）
ZIDONG_BEIFEN_JIA = "自动备份"     # 自动备份文件夹名（跟字幕文件建在同一层）
HUA_ZIMU_JIZHUN = (1280, 720)     # 没打开 ASS（比如 SRT）时的基准分辨率：字号、位置都按它的比例缩放
HUA_ZIMU_ZIHAO = 46.0             # 没打开 ASS 时画面里字幕的字号（按上面的基准分辨率算）
HUA_ZIMU_BEI_JIAO = 30.0          # 没打开 ASS 时字幕离画面底部 / 左右的距离（按基准分辨率算）
HUA_ZIMU_FANGDA = 16              # 画字幕时先把字放大这么多倍再缩回来（字号只能取整数，放大再缩才够准）
SHIPIN_FILTER = (
    "视频文件 (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.flv *.wmv "
    "*.ts *.asf *.mpg *.mpeg);;所有文件 (*)"
)
# =====================================================================


B = QtWidgets.QStyle     # 按钮图标用 Qt 自带的标准图标编号


# ---------------------------------------------------------------------
# 配色
# ---------------------------------------------------------------------
_YANSE_AN = {
    "zhuse": "#0A84FF",
    "zhuse_hover": "#409CFF",
    "beijing": "#1c1c1e",
    "beijing2": "#2c2c2e",
    "mian": "#2c2c2e",
    "mian_hover": "#3a3a3c",
    "mian_anxia": "#48484a",
    "biankuang": "#3a3a3c",
    "biankuang_liang": "#48484a",
    "wenzi": "#f5f5f7",
    "wenzi_ci": "#aeaeb2",
    "gundong": "#48484a",
    "gundong_hover": "#636366",
    "hong": "#FF453A",
    "juzi": "#FF9F0A",
    "qing": "#30D158",
    "bo": "#c7c7cc",
    "bo_zhong": "#373839",
    "bo_di": "#1c1c1e",
}

_YANSE_LIANG = {
    "zhuse": "#0071e3",
    "zhuse_hover": "#0077ED",
    "beijing": "#ffffff",
    "beijing2": "#F9F9F9",
    "mian": "#f5f5f7",
    "mian_hover": "#e5e5e5",
    "mian_anxia": "#d5d5d5",
    "biankuang": "#E5E5E5",
    "biankuang_liang": "#d2d2d7",
    "wenzi": "#1d1d1f",
    "wenzi_ci": "#86868b",
    "gundong": "#c1c1c1",
    "gundong_hover": "#a8a8a8",
    "hong": "#FF453A",
    "juzi": "#FF9F0A",
    "qing": "#30D158",
    "bo": "#9aa0a6",
    "bo_zhong": "#F0F0F1",
    "bo_di": "#ffffff",
}


def _ys():
    return _YANSE_AN if ZHUTI == "dark" else _YANSE_LIANG


def _shijian_wenben(ms, fps=0.0):
    """毫秒 -> 时:分:秒:帧"""
    ms = max(0, int(ms or 0))
    zong_miao = ms // 1000
    zhen = int(round(ms / 1000.0 * fps)) % max(1, int(round(fps or 0) or 1))
    if not fps or fps <= 0:
        zhen = 0
    h, r = divmod(zong_miao, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{zhen:02d}"


def _shijian_zhen_wenben(ms, zhen):
    """底栏那个时间显示，照 Aegisub 的视频窗时间框：

        `0:40:08.666 - 72260`

    前面是 时:分:秒.毫秒，后面跟当前帧号，中间空格 - 空格。
    时间码和帧号都是 Aegisub 原样的写法。
    """
    ms = max(0, int(ms or 0))
    h, r = divmod(ms // 1000, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms % 1000:03d} - {max(0, int(zhen or 0))}"


def _zhen_tu_huan(zhen_rgb):
    """cv2 RGB ndarray -> QImage（缩放和转色已在取帧线程做完，这里只包装）"""
    if zhen_rgb is None:
        return QtGui.QImage()
    h, w = zhen_rgb.shape[:2]
    if w <= 0 or h <= 0:
        return QtGui.QImage()
    return QtGui.QImage(
        zhen_rgb.data, w, h, zhen_rgb.strides[0], QtGui.QImage.Format_RGB888
    ).copy()


# ---------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------
class _JunfenTabBar(QtWidgets.QTabBar):
    """标签均分整条标签栏的宽度

    Qt 只把标签栏摆到"标签刚好放得下"那么宽（宽度取 sizeHint），右栏窄的时候
    右边会空出一大块，标签也各自按文字长短缩着排。这里直接报"跟右栏一样宽"，
    Qt 给的宽度就是整条右栏，标签再按数量均分。
    """

    def sizeHint(self):
        da = super().sizeHint()
        zhu = self.parentWidget()
        if zhu is None:
            return da
        return QtCore.QSize(zhu.width(), da.height())

    def tabSizeHint(self, i):
        da = super().tabSizeHint(i)
        jun = self.width() // max(1, self.count())
        return QtCore.QSize(max(da.width(), jun), da.height())


# 编辑区里所有输入框（单行框 / 下拉框 / 数字框）统一的内容高度：照 AEG 那样
# 一排框高低对齐。改这一个数，整片界面的输入框一起变。
SHURU_GAO = 22


def _yangshi_quanju():
    c = _ys()
    return f"""
    QDialog#YsgVideoWork {{
        background-color: {c['beijing']};
    }}
    QDialog#YsgVideoWork QLabel {{
        color: {c['wenzi']};
        background: transparent;
    }}
    QFrame#YsgPreview, QFrame#YsgTimeline, QFrame#YsgPanel {{
        background-color: {c['beijing']};
        border: 1px solid {c['biankuang']};
        border-radius: 8px;
    }}
    QFrame#YsgPreviewFooter, QFrame#YsgTimelineBar {{
        background-color: {c['beijing2']};
        border: none;
    }}
    QLabel#YsgMeta {{
        color: {c['wenzi_ci']};
        font-size: 12px;
    }}
    QDialog#YsgVideoWork QCheckBox {{
        color: {c['wenzi']};
        font-size: 12px;
        background: transparent;
    }}
    QLabel#YsgPanelTitle {{
        color: {c['wenzi']};
        font-size: 13px;
        font-weight: 600;
    }}
    QLabel#YsgTimeNow {{
        color: {c['zhuse']};
        font-family: Consolas, monospace;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#YsgHint {{
        color: {c['wenzi_ci']};
        font-size: 12px;
    }}
    QLabel#YsgZimuTiao {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang']};
        border-radius: 6px;
        font-size: {ZIMU_TIAO_ZIHAO}px;
        padding: 0 12px;
    }}
    QComboBox {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 3px 22px 3px 8px;
        font-size: 12px;
        min-height: {SHURU_GAO}px;
        max-height: {SHURU_GAO}px;
    }}
    QLineEdit {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 12px;
        min-height: {SHURU_GAO}px;
        max-height: {SHURU_GAO}px;
    }}
    QLineEdit:disabled {{ color: {c['wenzi_ci']}; }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 3px 4px;
        font-size: 12px;
        min-height: {SHURU_GAO}px;
        max-height: {SHURU_GAO}px;
    }}
    QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {c['wenzi_ci']}; }}
    QComboBox QLineEdit {{
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        min-height: 0px;
        max-height: 16777215px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 18px;
    }}
    QComboBox::down-arrow {{
        image: url(:/images/images/caret-down.svg);
        width: 9px;
        height: 14px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang']};
        selection-background-color: {c['zhuse']};
        selection-color: #ffffff;
        outline: none;
    }}
    QScrollBar#YsgShijianZhouTiao:horizontal {{
        border: none;
        background-color: {c['beijing2']};
        height: 8px;
        margin: 0;
    }}
    QScrollBar#YsgShijianZhouTiao::handle:horizontal {{
        background-color: {c['gundong']};
        min-width: 20px;
        border-radius: 4px;
    }}
    QScrollBar#YsgShijianZhouTiao::handle:horizontal:hover {{
        background-color: {c['gundong_hover']};
    }}
    QScrollBar#YsgShijianZhouTiao::add-line:horizontal,
    QScrollBar#YsgShijianZhouTiao::sub-line:horizontal {{
        width: 0;
        border: none;
        background: transparent;
    }}
    QScrollBar#YsgShijianZhouTiao::add-page:horizontal,
    QScrollBar#YsgShijianZhouTiao::sub-page:horizontal {{
        background: transparent;
    }}
    QTabWidget#YsgRightTabs::pane {{
        border: none;
        background-color: {c['beijing']};
    }}
    QTabWidget#YsgRightTabs QTabBar::tab {{
        background-color: {c['beijing2']};
        color: {c['wenzi_ci']};
        border: 1px solid {c['biankuang']};
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 7px 10px;
        font-size: 12px;
        margin-right: 2px;
    }}
    QTabWidget#YsgRightTabs QTabBar::tab:hover:!selected {{
        background-color: {c['mian_hover']};
        color: {c['wenzi']};
    }}
    QTabWidget#YsgRightTabs QTabBar::tab:selected {{
        background-color: {c['beijing']};
        color: {c['zhuse']};
        font-weight: 600;
        border-color: {c['biankuang_liang']};
    }}
    QTabWidget#YsgRightTabs QTabBar::tab:selected:hover {{
        background-color: {c['beijing']};
        color: {c['zhuse']};
    }}
    QTabWidget#YsgRightTabs QTabBar::tab:disabled {{
        color: {c['wenzi_ci']};
    }}
    QSplitter::handle {{ background-color: transparent; border-radius: 3px; }}
    QSplitter::handle:horizontal {{ width: 9px; }}
    QSplitter::handle:vertical {{ height: 9px; }}
    QSplitter::handle:hover {{ background-color: {c['zhuse']}; }}
    """


def _ys_tubiao():
    c = _ys()
    # 图标固定压在浅色小方块上 —— 系统标准图标是深色的，
    # 给个浅底才不会在深色主题里糊成一片
    return f"""
    QPushButton {{
        background-color: #ffffff;
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 4px;
        font-size: 12px;
        min-width: 22px;
        max-width: 22px;
        min-height: 22px;
        max-height: 22px;
        padding: 0px;
    }}
    QPushButton:hover {{
        background-color: {c['mian_hover']};
        border-color: {c['zhuse']};
    }}
    QPushButton:checked {{
        background-color: #ffffff;
        border: 2px solid {c['zhuse']};
    }}
    QPushButton:pressed {{ background-color: {c['mian_anxia']}; }}
    QPushButton:disabled {{
        background-color: {c['beijing2']};
        color: {c['wenzi_ci']};
        border-color: {c['biankuang']};
    }}
    """


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
        padding: 0 14px;
        min-height: 32px;
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
        font-size: 13px;
        padding: 0 14px;
        min-height: 32px;
    }}
    QPushButton:hover {{ background-color: {c['mian_hover']}; }}
    QPushButton:disabled {{
        color: {c['wenzi_ci']};
        border-color: {c['biankuang']};
    }}
    """


def _ys_ci_anniu_xiao():
    """视频底栏用的小号文字按钮（打开字幕 / 保存字幕）

    底栏那行控件都只有 22px 高，用普通按钮（32px）会整条撑高。
    """
    c = _ys()
    return f"""
    QPushButton {{
        background-color: {c['mian']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 4px;
        font-size: 11px;
        padding: 0 8px;
        min-height: {SHURU_GAO}px;
        max-height: {SHURU_GAO}px;
    }}
    QPushButton:hover {{ background-color: {c['mian_hover']}; }}
    QPushButton:disabled {{
        color: {c['wenzi_ci']};
        border-color: {c['biankuang']};
    }}
    """


def _ys_huadong_tiao():
    c = _ys()
    return f"""
    QSlider::groove:horizontal {{
        height: 4px;
        border-radius: 2px;
        background: {c['biankuang_liang']};
    }}
    QSlider::sub-page:horizontal {{
        height: 4px;
        border-radius: 2px;
        background: {c['zhuse']};
    }}
    QSlider::handle:horizontal {{
        width: 12px;
        height: 12px;
        margin: -5px 0;
        border-radius: 6px;
        background: {c['beijing']};
        border: 1px solid {c['biankuang_liang']};
    }}
    QSlider::handle:horizontal:hover {{ border-color: {c['zhuse']}; }}
    """


def _ys_xiao_xiang():
    """小号下拉框 / 数值框：倍速、时间轴缩放这种只装两三个字的框

    通用那套的内边距是照普通输入框定的，套在这种小框上会白占一片宽度，
    这里单独来一套，箭头也收窄。
    """
    c = _ys()
    return f"""
    QComboBox, QSpinBox {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 4px;
        font-size: 12px;
        min-height: {SHURU_GAO}px;
        max-height: {SHURU_GAO}px;
    }}
    QComboBox {{ padding: 3px 12px 3px 4px; }}
    QSpinBox {{ padding: 3px 0px 3px 6px; }}
    QComboBox::drop-down {{ width: 12px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang']};
        selection-background-color: {c['zhuse']};
        selection-color: #ffffff;
        outline: none;
    }}
    """


def _tupian_lu(ming):
    """软件自带的按钮图片（放 anylabeling/resources/images 下）的完整路径"""
    gen = osp.dirname(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    )
    return osp.join(gen, "resources", "images", ming)


def _shezhi_tubiao(anniu, tubiao):
    """给按钮换图标：Qt 系统标准图标（传枚举）或自带的 png（传文件名 / 路径）"""
    if tubiao is None:
        anniu.setIcon(QtGui.QIcon())
        return
    if isinstance(tubiao, str):
        anniu.setIcon(QtGui.QIcon(_tupian_lu(tubiao)))
        anniu.setIconSize(QtCore.QSize(16, 16))
        return
    anniu.setIcon(QtWidgets.QApplication.style().standardIcon(tubiao))
    anniu.setIconSize(QtCore.QSize(12, 12))


def _tubiao_anniu(tubiao, tishi, cao, wenben=""):
    """统一的小方形按钮（图标用 Qt 系统标准图标，稳）"""
    a = QtWidgets.QPushButton()
    a.setStyleSheet(_ys_tubiao())
    a.setToolTip(tishi)
    a.setCursor(Qt.PointingHandCursor)
    _shezhi_tubiao(a, tubiao)
    if wenben:
        a.setText(wenben)
    if cao is not None:
        a.clicked.connect(cao)
    return a


# ---------------------------------------------------------------------
# 播放时把系统定时器精度提到 1ms
#   Windows 默认睡眠粒度约 15ms，msleep(17) 实际会睡到 30ms 上下，
#   画面就会一帧帧变慢。播放期间临时提高精度，播完还原。
# ---------------------------------------------------------------------
WINMM = None
try:
    import ctypes

    WINMM = ctypes.WinDLL("winmm")
except Exception:  # noqa
    WINMM = None

_JINGDU_CISHU = 0


def _tigao_jingdu():
    global _JINGDU_CISHU
    if WINMM is None:
        return
    try:
        if _JINGDU_CISHU == 0:
            WINMM.timeBeginPeriod(1)
        _JINGDU_CISHU += 1
    except Exception:  # noqa
        pass


def _huifu_jingdu():
    global _JINGDU_CISHU
    if WINMM is None:
        return
    try:
        _JINGDU_CISHU = max(0, _JINGDU_CISHU - 1)
        if _JINGDU_CISHU == 0:
            WINMM.timeEndPeriod(1)
    except Exception:  # noqa
        pass


# 取帧定位：拿这个片子实测过（720p、30fps、关键帧 10 秒一个）
#   · 顺着往下 read 一帧         7 ms
#   · 重新 set 定位再 read 一帧  0.1 ~ 2.1 秒
# 差这么多是因为 OpenCV 一 set 就跳回目标前面那个关键帧，再一帧一帧解到目标；
# 关键帧 300 帧一个，落得离关键帧最远时就等于要解 299 帧 ≈ 2 秒 —— 前后挪帧
# 时"字幕切过去了画面还在后头"就是这么来的。所以：
#   · 往前挪（含"下一帧"）-> 顺着 read 往下追，7ms
#   · 往回挪             -> 翻缓存里刚显示过的帧，瞬间
#   · 跳得远             -> 才准 set
ZHEN_SHUNDU_ZUI_DUO = 30            # 往前挪不超过这么多帧就顺着读，不 set
ZHEN_HOU_TUI_BEIHUAN = 30           # 暂停时定位，顺手把目标前面这么多帧也解出来存着
ZHEN_HUANCUN_ZUI_DUO = 96           # 往回退最多能退多少帧（更远就得重新定位）
ZHEN_HUANCUN_ZUI_DA = 192 * 1024 * 1024   # 这份帧缓存最多占多少内存


# ---------------------------------------------------------------------
# mpv 播放内核（libmpv）
#   解码 / 声音 / 定位 / 变速 / 时钟全交给 mpv —— 跟 ArcTime 用的是同一个
#   引擎，所以拖播放头跟手，不会像 OpenCV 那样一次定位几百毫秒到两秒。
#   画面：mpv 渲染进 Qt 的 OpenGL 画布（不是另开一个窗口），所以画面上叠的
#         字幕、检测框、框选还是我们自己用 QPainter 画，一行都不用改。
#   dll：软件根目录下 mpv\libmpv-2.dll（跟 gongzuotai.ini 平级）
# ---------------------------------------------------------------------
MPV_JIA = "mpv"                       # 播放内核放软件根目录下这个文件夹里
MPV_DLL_MING = "libmpv-2.dll"

# 拖着播放头走时，每隔这么多毫秒看一眼鼠标在哪（毫秒）。
# 指针一秒能挪上百个位置，每个位置都发一发 seek 会把 mpv 淹了 —— 反而一
# 张新图都出不来。所以按这个节拍看一眼，每次只把"当前最新位置"交出去，中
# 间滑过的位置一律扔掉。
# 交出去的定位是精确到帧的：画面永远落在鼠标指着的那一帧上，不将就。真正
# 发给 mpv 的节奏由解码速度决定（上一发解完了才发下一个，见
# DINGWEI_CHAO_SHI_MS），这个节拍只是"看鼠标"的频率。
# 只改这一行：数越小越不容易漏掉鼠标位置、mpv 越吃力；数越大越省。
TUO_SEEK_JIAN_GE_MS = 40

# 一次定位最多认它"还没落地"多久（毫秒）。定位是一发一发排队的：上一发没
# 落地就不发下一发（互相打断的话每一发都解不完，画面一直是残的）。落地靠
# mpv 的 PLAYBACK_RESTART 事件通知；正常情况这个值用不到，只有那事件没来
# 时才靠它把队列放行，免得卡死。
# 只改这一行：一般不用动。
DINGWEI_CHAO_SHI_MS = 800

# 给 mpv 的参数（想调画质 / 性能就改这里）
MPV_CANSHU = [
    ("vo", "libmpv"),              # 画面由我们自己取去渲染（不另开窗口）
    ("gpu-api", "opengl"),
    ("hwdec", "auto-safe"),        # 硬解优先，4K / 高码率省 CPU
    ("keep-open", "yes"),          # 播到结尾留住最后一帧
    ("keep-open-pause", "yes"),
    ("idle", "yes"),
    ("pause", "yes"),              # 载入先停着，等按播放
    # 定位默认不精确到帧：一按播放头就拖的时候，每一次都精确到帧得从关键帧
    # 重解几百帧，画面根本追不上鼠标。拖的过程用不精确的（落最近的关键帧，
    # 二三十毫秒），松手那一下再补一发精确的落到位。
    ("hr-seek", "no"),
    ("osc", "no"),                 # 不要 mpv 自己的那套界面
    ("osd-level", "0"),
    # 不许 mpv 显示任何字幕：我们是字幕编辑工具，字幕是自己画在画面上的。
    # sub-auto 是"自动加载同目录同名字幕"（mpv 默认 fuzzy），sid 管内封软字幕。
    ("sub-auto", "no"),
    ("sid", "no"),
    ("input-default-bindings", "no"),
    ("input-vo-keyboard", "no"),
    ("terminal", "no"),
    ("msg-level", "all=no"),
]

MPV_RENDER_PARAM_INVALID = 0
MPV_RENDER_PARAM_API_TYPE = 1
MPV_RENDER_PARAM_OPENGL_INIT_PARAMS = 2
MPV_RENDER_PARAM_OPENGL_FBO = 3
MPV_RENDER_PARAM_FLIP_Y = 4

MPV_FORMAT_FLAG = 3
MPV_FORMAT_DOUBLE = 5

MPV_EVENT_NONE = 0
MPV_EVENT_END_FILE = 7
MPV_EVENT_FILE_LOADED = 8
# 一次定位（seek）真正落地了：mpv 解到目标位置、画面已就位。定位排队就靠
# 这个事件判"上一发完事了没有"（不能看 time-pos —— 命令一发出去它就报目标
# 位置了，比画面早）。
MPV_EVENT_PLAYBACK_RESTART = 21

# mpv_render_context_update() 返回的标记：该渲染了。
# 注意：重画（画面没变）也带这个标记，它区分不了"真出了新视频帧"。
MPV_RENDER_UPDATE_FRAME = 1

# 问"接下来要画的这一帧到底是什么帧"用的（mpv_render_context_get_info）。
# 只有真解出来的新视频帧才算数：重画（画面没变）和重复上一帧都不算 ——
# 那两种时候画面上还是老那一帧，字幕跟着换就跑到画面前头去了。
MPV_RENDER_PARAM_NEXT_FRAME_INFO = 11
ZHEN_PRESENT = 1 << 0   # 有一帧要画（重画也算）
ZHEN_REDRAW = 1 << 1    # 不是真新解出来的视频帧，只是重画
ZHEN_REPEAT = 1 << 2    # 要求把上一帧一模一样再画一遍

GL_COLOR_BUFFER_BIT = 0x00004000


class _MpvRenderParam(ctypes.Structure):
    _fields_ = [("leixing", ctypes.c_int), ("shuju", ctypes.c_void_p)]


class _MpvOpenglInitParams(ctypes.Structure):
    _fields_ = [
        ("qu_hanshu", ctypes.c_void_p),      # get_proc_address 回调
        ("shangxiawen", ctypes.c_void_p),    # 上面那个回调的上下文
    ]


class _MpvOpenglFbo(ctypes.Structure):
    _fields_ = [
        ("fbo", ctypes.c_int),
        ("kuan", ctypes.c_int),
        ("gao", ctypes.c_int),
        ("nei_geshi", ctypes.c_int),
    ]


class _MpvZhenXinxi(ctypes.Structure):
    """mpv_render_frame_info：接下来要画的那一帧的说明（标记 + 该显示的时刻）"""

    _fields_ = [
        ("biaozhi", ctypes.c_uint64),
        ("mubiao_shike", ctypes.c_int64),
    ]


class _MpvEvent(ctypes.Structure):
    _fields_ = [
        ("shi_jian", ctypes.c_int),
        ("cuowu", ctypes.c_int),
        ("huifu_id", ctypes.c_uint64),
        ("shuju", ctypes.c_void_p),
    ]


_OPENGL32 = None
_KERNEL32 = None


def _qu_gl_hanshu(_shangxiawen, mingzi):
    """mpv 来问 OpenGL 函数地址时给它（wglGetProcAddress 先，再退到导出的）"""
    global _OPENGL32, _KERNEL32
    if _OPENGL32 is None:
        try:
            _OPENGL32 = ctypes.WinDLL("opengl32")
            _OPENGL32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
            _OPENGL32.wglGetProcAddress.restype = ctypes.c_void_p
            _OPENGL32.glClear.argtypes = [ctypes.c_uint]
            _OPENGL32.glClearColor.argtypes = [ctypes.c_float] * 4
        except Exception:  # noqa
            _OPENGL32 = False
            return None
    if not _OPENGL32:
        return None
    try:
        dizhi = _OPENGL32.wglGetProcAddress(mingzi)
    except Exception:  # noqa
        dizhi = None
    if dizhi:
        return dizhi
    try:
        if _KERNEL32 is None:
            _KERNEL32 = ctypes.WinDLL("kernel32")
            _KERNEL32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _KERNEL32.GetProcAddress.restype = ctypes.c_void_p
        return _KERNEL32.GetProcAddress(_OPENGL32._handle, mingzi)
    except Exception:  # noqa
        return None


_MPV_DLL = None                # 载进来的 libmpv（整个进程一份）
_MPV_JIAZAI_CUO = ""           # 载入失败的原因
_API_TYPE_OPENGL = ctypes.c_char_p(b"opengl")


def mpv_dll_lu():
    """libmpv-2.dll 的绝对路径：软件根目录下 mpv\（不写死盘符 / 目录名）"""
    gen = osp.dirname(
        osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    )
    return osp.join(gen, MPV_JIA, MPV_DLL_MING)


def jiazai_mpv():
    """载入 libmpv，把要用到的函数签名标好（只做一次）"""
    global _MPV_DLL, _MPV_JIAZAI_CUO
    if _MPV_DLL is not None:
        return _MPV_DLL
    lu = mpv_dll_lu()
    if not osp.isfile(lu):
        _MPV_JIAZAI_CUO = f"没找到播放内核：{lu}"
        return None
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(osp.dirname(lu))
        except Exception:  # noqa
            pass
    try:
        d = ctypes.CDLL(lu)
    except Exception as e:  # noqa
        _MPV_JIAZAI_CUO = f"载入播放内核失败：{e}"
        return None
    d.mpv_create.restype = ctypes.c_void_p
    d.mpv_initialize.argtypes = [ctypes.c_void_p]
    d.mpv_initialize.restype = ctypes.c_int
    d.mpv_set_option_string.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    d.mpv_set_option_string.restype = ctypes.c_int
    d.mpv_set_property_string.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    d.mpv_set_property_string.restype = ctypes.c_int
    d.mpv_get_property.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
    ]
    d.mpv_get_property.restype = ctypes.c_int
    d.mpv_command.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    d.mpv_command.restype = ctypes.c_int
    d.mpv_wait_event.argtypes = [ctypes.c_void_p, ctypes.c_double]
    d.mpv_wait_event.restype = ctypes.POINTER(_MpvEvent)
    d.mpv_error_string.argtypes = [ctypes.c_int]
    d.mpv_error_string.restype = ctypes.c_char_p
    d.mpv_terminate_destroy.argtypes = [ctypes.c_void_p]
    d.mpv_render_context_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(_MpvRenderParam),
    ]
    d.mpv_render_context_create.restype = ctypes.c_int
    d.mpv_render_context_render.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_MpvRenderParam),
    ]
    d.mpv_render_context_render.restype = ctypes.c_int
    d.mpv_render_context_update.argtypes = [ctypes.c_void_p]
    d.mpv_render_context_update.restype = ctypes.c_uint64
    d.mpv_render_context_set_update_callback.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    d.mpv_render_context_report_swap.argtypes = [ctypes.c_void_p]
    d.mpv_render_context_free.argtypes = [ctypes.c_void_p]
    # 问"接下来这一帧是不是真新视频帧"用的。老版本 mpv 没这个函数 —— 那就
    # 不标签名，用的时候自己会发现（见 HuamianQu._zhe_yi_zhen_shi_xin_zhen）。
    try:
        d.mpv_render_context_get_info.argtypes = [
            ctypes.c_void_p, _MpvRenderParam,
        ]
        d.mpv_render_context_get_info.restype = ctypes.c_int
    except AttributeError:
        pass
    _MPV_DLL = d
    return d


class MpvBofang(QtCore.QObject):
    """libmpv 封装：播放这一件事全归它

    对外的名字跟老的取帧线程对齐（zanting / jixu / tiaozheng / zhen_hao /
    shezhi_beisu / tingzhi / daowei），外面那一堆调用点基本不用动。
    """

    daowei = pyqtSignal()          # 播到结尾了（跟老的一样）
    xin_zhen = pyqtSignal()        # 有新画面了，画布收到就重画

    def __init__(self, parent=None):
        super().__init__(parent)
        self._d = jiazai_mpv()
        self.h = None
        self.fps = 30.0
        self.lujing = ""
        self.suo_dao = None        # 老接口留的占位（缩放交给 mpv）
        self.zhen_hook = None      # 老接口留的占位（跟随播放推理另开一路解码）
        self._yinliang = int(MO_REN_YINLIANG)
        self._beisu = float(MO_REN_BEISU)
        self._daowei_bao = False
        self._zai_ru_wan = False     # 片子真载入完了没有
        self._dai_dingwei = None     # 载入完成前先攒着的那次定位
        # 定位排队：一次只让一发 seek 在飞。
        #   后一发会把前一发的解码打断 —— 每一发都解不完，画面就一直是残的，
        #   直到你松手那一发才真解出来（那 1 秒延迟就是这么来的）。所以有在
        #   飞的就先不发，只记下最新位置（_dai_fa），等它落地（mpv 发
        #   PLAYBACK_RESTART）立刻把最新的那发补出去。
        self._seek_fa_hao = 0        # 已发出的定位编号
        self._seek_luo_di = 0        # 已确认落地的定位编号
        self._seek_fa_shi = 0.0      # 上一发是什么时候发出去的（兜底用）
        self._dai_fa = None          # 有在飞时攒着的最新目标 (秒, 精确?)
        if self._d is None:
            logger.error(f"视频工作台：{_MPV_JIAZAI_CUO}")
            return
        self._kai()

    # ---- 内核 ----
    def _cuo(self, hao):
        try:
            return self._d.mpv_error_string(hao).decode("utf-8", "ignore")
        except Exception:  # noqa
            return str(hao)

    def _mingling(self, *can):
        """给 mpv 下命令（参数一律 UTF-8，中文路径也认）"""
        if self.h is None:
            return False
        bu = (ctypes.c_char_p * (len(can) + 1))()
        for i, can_shu in enumerate(can):
            bu[i] = str(can_shu).encode("utf-8")
        bu[len(can)] = None
        return self._d.mpv_command(self.h, bu) >= 0

    def _she(self, ming, zhi):
        if self.h is None:
            return
        self._d.mpv_set_property_string(
            self.h, ming.encode("utf-8"), str(zhi).encode("utf-8")
        )

    def _shi_ma(self, ming):
        """读一个"是 / 不是"的属性；读不到当 False"""
        if self.h is None:
            return False
        zhi = ctypes.c_int(0)
        hao = self._d.mpv_get_property(
            self.h, ming.encode("utf-8"), MPV_FORMAT_FLAG, ctypes.byref(zhi)
        )
        return bool(zhi.value) if hao >= 0 else False

    def _shuzi(self, ming, moren=0.0):
        """读一个数字属性（time-pos / duration 这种）"""
        if self.h is None:
            return moren
        zhi = ctypes.c_double(0.0)
        hao = self._d.mpv_get_property(
            self.h, ming.encode("utf-8"), MPV_FORMAT_DOUBLE, ctypes.byref(zhi)
        )
        return float(zhi.value) if hao >= 0 else moren

    def _kai(self):
        """建内核 + 设参数（只建一次，换视频是 loadfile）"""
        if self.h is not None:
            return True
        self.h = self._d.mpv_create()
        if not self.h:
            logger.error("视频工作台：播放内核创建失败")
            return False
        for ming, zhi in MPV_CANSHU + [("volume", str(self._yinliang))]:
            hao = self._d.mpv_set_option_string(
                self.h, ming.encode("utf-8"), zhi.encode("utf-8")
            )
            if hao < 0:
                logger.warning(
                    f"视频工作台：mpv 参数 {ming}={zhi} 没设上（{self._cuo(hao)}）"
                )
        hao = self._d.mpv_initialize(self.h)
        if hao < 0:
            logger.error(f"视频工作台：播放内核初始化失败 {self._cuo(hao)}")
            self.h = None
            return False
        return True

    def dakai(self, lu):
        """换片子"""
        self.lujing = lu or ""
        self._daowei_bao = False
        self._zai_ru_wan = False
        self._dai_dingwei = None
        self._dai_fa = None
        self._seek_fa_hao = self._seek_luo_di = 0
        if not self._kai():
            return False
        return self._mingling("loadfile", lu, "replace")

    def zanting(self):
        self._she("pause", "yes")

    def jixu(self):
        self._she("pause", "no")

    def zai_bofang(self):
        return not self._shi_ma("pause")

    def daowei_ma(self):
        """播到结尾没有"""
        return self._shi_ma("eof-reached")

    def shijian_ms(self):
        return int(round(max(0.0, self._shuzi("time-pos", 0.0)) * 1000))

    def shichang_ms(self):
        return int(round(max(0.0, self._shuzi("duration", 0.0)) * 1000))

    def zhen_hao(self):
        """现在在第几帧（时间换算出来的，跟画面显示的那一帧对得上）"""
        return int(round(self._shuzi("time-pos", 0.0) * max(1e-6, self.fps)))

    def tiaozheng(self, zhen_hao, dan_bu=False, jingque=True):
        """定位到某一帧（排队：一次只让一发在飞）

        定位是精确到帧的：mpv 得从前面那个关键帧一路解到目标帧，几百毫秒。
        这期间要是又发一发（拖动时鼠标一直在动），mpv 会把前一发扔掉重来 ——
        每一发都解不完，画面就一直是残的，直到你松手那一发才真解出来。
        所以这里排队：有在飞的就只记下最新位置，等它落地（PLAYBACK_RESTART）
        立刻把最新的补出去。画面的节奏跟着解码速度走，但每一发都真解出来。
        （dan_bu 是老接口留下的，mpv 不需要，收下不用）
        """
        self._pai_tiaozheng(
            max(0, int(zhen_hao)) / max(1e-6, self.fps), bool(jingque)
        )

    def _zai_fei(self):
        """有一发定位还没落地？

        落地靠 mpv 的 PLAYBACK_RESTART 事件（不能看 time-pos：命令一发出去
        它就报目标位置了，比画面早）。万一那事件没来，超过兜底时长就当落地，
        免得整条队列卡死。
        """
        if self._seek_fa_hao <= self._seek_luo_di:
            return False
        if time.perf_counter() - self._seek_fa_shi > DINGWEI_CHAO_SHI_MS / 1000.0:
            self._seek_luo_di = self._seek_fa_hao
            return False
        return True

    def _pai_tiaozheng(self, miao, jingque):
        """把一次定位交给队列：能发就发，不能发就先攒着（只留最新那份）"""
        miao = max(0.0, float(miao))
        if self.h is None:
            return
        if not self._zai_ru_wan:
            # 片子还没载入完：先攒着，等载入完再定位。
            # （载入期间发出去的 seek 会被丢掉 —— 那一帧就永远等不到了）
            self._dai_dingwei = (miao, jingque)
            return
        if self._zai_fei():
            self._dai_fa = (miao, jingque)
            return
        self._fa_tiaozheng(miao, jingque)

    def _fa_tiaozheng(self, miao, jingque):
        self._seek_fa_hao += 1
        self._seek_fa_shi = time.perf_counter()
        self._mingling(
            "seek", "%.6f" % max(0.0, float(miao)),
            "absolute+exact" if jingque else "absolute",
        )

    def tiaozheng_miao(self, miao, jingque=True):
        self._pai_tiaozheng(miao, bool(jingque))

    def tiaozheng_ms(self, ms, jingque=True):
        self.tiaozheng_miao(max(0, int(ms)) / 1000.0, jingque=jingque)

    def zhen_bu(self, fangxiang):
        """走一帧（正数前进、负数后退），下成了返回真

        用 mpv 自己的单帧步进：前进是接着往下解一帧，后退是翻它自己的回退
        缓存，都是十几毫秒。走 seek 那条路每一下都得回关键帧重解，几百毫秒，
        而且往前和往后一样慢 —— 这就是挪一帧画面跟不上的原因。
        """
        if self.h is None:
            return False
        return self._mingling(
            "frame-step" if fangxiang >= 0 else "frame-back-step"
        )

    def shezhi_beisu(self, beisu):
        self._beisu = max(0.05, float(beisu or 1.0))
        self._she("speed", "%g" % self._beisu)

    def shezhi_yinliang(self, zhi):
        self._yinliang = max(0, min(100, int(zhi)))
        self._she("volume", str(self._yinliang))

    def yinliang(self):
        return int(self._yinliang)

    def pai_shijian(self):
        """顺手排一下 mpv 的事件队列（没人取的话它会一直攒着）

        顺带认一下"载入完了没有"：载入期间发出去的定位先攒着，等它载入完
        再补上，不然那一下会被丢掉。
        """
        if self.h is None:
            return
        for _ in range(64):
            e = self._d.mpv_wait_event(self.h, 0.0)
            if not e or e.contents.shi_jian == MPV_EVENT_NONE:
                break
            if e.contents.shi_jian == MPV_EVENT_FILE_LOADED:
                self._zai_ru_wan = True
                if self._dai_dingwei is not None:
                    miao, jingque = self._dai_dingwei
                    self._dai_dingwei = None
                    self._fa_tiaozheng(miao, jingque)
            elif e.contents.shi_jian == MPV_EVENT_END_FILE:
                self._zai_ru_wan = False
            elif e.contents.shi_jian == MPV_EVENT_PLAYBACK_RESTART:
                # 上一发定位落地了（画面已经到那一帧）
                self._seek_luo_di = self._seek_fa_hao
        # 前一发落地了、还攒着新位置没发 -> 立刻补上。
        # （拖动中手停住不再产生新位置时，就靠这一条把攒的那份消化掉）
        if self._dai_fa is not None and not self._zai_fei():
            miao, jingque = self._dai_fa
            self._dai_fa = None
            self._fa_tiaozheng(miao, jingque)

    def tingzhi(self):
        """停播并卸掉当前文件（内核留着，下次 loadfile 直接用）"""
        if self.h is None:
            return
        self._mingling("stop")
        self.lujing = ""
        self._daowei_bao = False
        self._zai_ru_wan = False
        self._dai_dingwei = None
        self._dai_fa = None
        self._seek_fa_hao = self._seek_luo_di = 0

    def guan(self):
        """整个关掉（关窗口时）"""
        if self.h is None:
            return
        try:
            self._d.mpv_terminate_destroy(self.h)
        except Exception:  # noqa
            pass
        self.h = None


class _YinpinDaiLi:
    """把原来那个 Qt 音频播放器换成 mpv —— 声音也归播放内核

    方法名照旧，外面写 setVolume / volume / play / pause / stop /
    setPlaybackRate 的地方一行都不用改。定位不在这儿做（统一走 mpv 的
    seek，画面和声音一起走）。
    """

    def __init__(self, mpv):
        self.mpv = mpv

    def setMedia(self, *_a):
        pass

    def setPosition(self, _ms):
        pass

    def play(self):
        self.mpv.jixu()

    def pause(self):
        self.mpv.zanting()

    def stop(self):
        self.mpv.zanting()

    def setVolume(self, zhi):
        self.mpv.shezhi_yinliang(zhi)

    def volume(self):
        return self.mpv.yinliang()

    def setPlaybackRate(self, beisu):
        self.mpv.shezhi_beisu(beisu)


# ---------------------------------------------------------------------
# 视频取帧线程
#   OpenCV 逐帧读取。以前画面靠它，现在画面归 mpv 了，这一路只留给
#   「跟随播放做推理」用（只喂模型，不显示、不转色、不留缓存）。
# ---------------------------------------------------------------------
class _ZhenDuQu(QtCore.QThread):
    """OpenCV 逐帧读取。支持暂停、定位、变速、单帧步进。"""

    daowei = pyqtSignal()                   # 播到结尾

    def __init__(self, lujing, fps, parent=None):
        super().__init__(parent)
        self.lujing = lujing
        self.fps = float(fps or 30.0)
        self._yunxing = True
        self._zanting = True
        self._beisu = 1.0
        self._tiaozheng = None              # 请求定位到的 (帧号, 是不是一下一下挪)
        self._dangqian = 0
        self._chong_ji_zhun = False         # 改倍速时要求重设时间基准
        self.suo_dao = None                 # (宽,高) 显示尺寸；None = 原尺寸
        self._zui_xin = None                # 最新一帧 (RGB, 帧号)，主线程来取
        self._xiaci = 0                     # 读取器内部"下一帧"是第几号（顺着读记的）
        self._huan_cun = OrderedDict()      # 刚显示过的帧 {帧号: RGB}，往回退帧直接取
        self._huan_cun_zi = 0               # 上面这份缓存一共占了多少字节
        # 推理用：每读一帧原图（BGR，不缩放）就回调一次；
        # 为 None 表示没开推理，一帧都不往外送，不白耗内存
        self.zhen_hook = None
        # 只喂模型（跟随播放做推理）：不转色、不留缓存、不往主线程送图
        self.zhi_tuili = False
        # 跟着 mpv 的播放时间走：返回"现在播到第几秒"，None = 不跟
        self.pace_hook = None

    # ---- 外部控制 ----
    def zanting(self):
        self._zanting = True

    def jixu(self):
        self._zanting = False

    def qu_zui_xin(self):
        """给主线程取最新帧；取走就清空，永远不会堆积、不会延迟累积"""
        zhen = self._zui_xin
        self._zui_xin = None
        return zhen

    def shezhi_beisu(self, beisu):
        self._beisu = max(0.05, float(beisu or 1.0))
        self._chong_ji_zhun = True

    def tiaozheng(self, zhen_hao, dan_bu=False):
        """请求定位到第 zhen_hao 帧

        dan_bu=True = 一下一下挪（方向键 / 点一下定位），不是鼠标拖着扫。
        这种就把目标前面一段也顺手解出来存着，往回退帧同样是瞬间。
        """
        self._tiaozheng = (max(0, int(zhen_hao)), bool(dan_bu))

    def zhen_hao(self):
        return self._dangqian

    def tingzhi(self):
        self._yunxing = False
        self.wait(3000)
        self._huan_cun.clear()
        self._huan_cun_zi = 0

    # ---- 帧转换（缩放和转色都放本线程，主线程只管画）----
    def _zhen_zhuan(self, zhen):
        kao = self.suo_dao
        if kao:
            bw, bh = int(kao[0]), int(kao[1])
            if bw > 0 and bh > 0 and (
                zhen.shape[1] != bw or zhen.shape[0] != bh
            ):
                zhen = cv2.resize(zhen, (bw, bh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(zhen, cv2.COLOR_BGR2RGB)

    def _cun_zhen(self, hao, rgb):
        """把已经显示过的那一帧留在缓存里

        留着是为了往回退帧：往回退只能靠 set 重新定位，一次几百毫秒到两秒，
        卡就卡在这。缓存里有就直接取，往前、往回一样快。
        """
        if hao in self._huan_cun:
            return
        self._huan_cun[hao] = rgb
        self._huan_cun_zi += int(rgb.nbytes)
        while (
            len(self._huan_cun) > ZHEN_HUANCUN_ZUI_DUO
            or self._huan_cun_zi > ZHEN_HUANCUN_ZUI_DA
        ):
            _, jiu = self._huan_cun.popitem(last=False)   # 丢最老的那帧
            self._huan_cun_zi -= int(jiu.nbytes)

    def _song_zhen(self, zhen, hao):
        """把原始帧交给推理（钩子里只是丢进槽位，很快，不拖慢读取）"""
        if self.zhen_hook is None:
            return
        try:
            self.zhen_hook(zhen, hao)
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：送推理失败 {cuowu}")

    def _deng_bofang(self):
        """只喂模型的那一路：读到画面前头去了就等一等（跟着 mpv 播放走）"""
        hs = self.pace_hook
        if hs is None:
            return
        while self._yunxing and not self._zanting:
            dq = hs()
            if dq is None:
                break
            if self._dangqian / max(1e-6, self.fps) <= dq + 0.25:
                break
            self.msleep(4)

    def _dao_mubiao(self, cap, mubiao, dan_bu=False):
        """走到第 mubiao 帧，返回它的 RGB（拿不到返回 None）

        三条路，按开销从便宜到贵：
          1. 缓存里有（刚显示过 / 刚顺手解过）-> 直接拿，瞬间；
          2. 正好在前面不远 -> 顺着 read 往下追，一帧 7ms；
          3. 别的（往回退得远、大跨度跳）-> 只能 set 重新定位。
        """
        # 0) 只喂模型的那一路：定位一下、读一帧丢给钩子就完事，
        #    不转色、不留缓存、不往主线程送（省内存也省时间）
        if self.zhi_tuili:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(mubiao)))
            chenggong, zhen = cap.read()
            if not chenggong or zhen is None:
                return None
            self._dangqian = int(mubiao)
            self._xiaci = self._dangqian + 1
            self._song_zhen(zhen, self._dangqian)
            return None

        # 1) 刚显示过：直接翻缓存，往前往回都是瞬间
        jiu = self._huan_cun.get(mubiao)
        if jiu is not None:
            return jiu

        # 2) 在前面不远（含"下一帧"）：顺着读过去，别 set。一下一下挪的时候
        #    中间跳过的帧也一并收进缓存（往前跳 20 帧再往回退一帧也不用定位）。
        tiao = mubiao - self._xiaci
        if 0 <= tiao <= ZHEN_SHUNDU_ZUI_DUO:
            hao = self._xiaci
            xian = None
            while hao <= mubiao:
                chenggong, zhen = cap.read()
                if not chenggong or zhen is None:
                    return None
                if dan_bu or hao == mubiao:
                    xian = self._zhen_zhuan(zhen)
                    self._cun_zhen(hao, xian)
                hao += 1
            self._xiaci = hao
            return xian

        # 3) 剩下的只能重新定位。定位这一下不管目标是哪一帧都要跳回它前面那个
        #    关键帧再解过来，所以一下一下挪的时候干脆从目标前面一点起步，把
        #    中间这几帧顺手收进缓存 —— 往回退帧就不用再定位了，前后一样快。
        hou = ZHEN_HOU_TUI_BEIHUAN if dan_bu else 0
        qi = max(0, mubiao - hou)
        cap.set(cv2.CAP_PROP_POS_FRAMES, qi)
        xian = None
        hao = qi
        while hao <= mubiao:
            chenggong, zhen = cap.read()
            if not chenggong or zhen is None:
                return None
            xian = self._zhen_zhuan(zhen)
            self._cun_zhen(hao, xian)
            hao += 1
        self._xiaci = hao
        return xian

    # ---- 线程体 ----
    def run(self):
        cap = cv2.VideoCapture(self.lujing)
        if not cap.isOpened():
            logger.error(f"视频工作台：打不开视频 {self.lujing}")
            self.daowei.emit()
            return

        self._xiaci = 0                         # 刚打开，读到的下一帧是第 0 帧
        _tigao_jingdu()
        try:
            ji_zhun = time.perf_counter()       # 基准时刻
            ji_zhun_zhen = self._dangqian       # 基准时刻对应的帧号
            while self._yunxing:
                if self._chong_ji_zhun:
                    self._chong_ji_zhun = False
                    ji_zhun = time.perf_counter()
                    ji_zhun_zhen = self._dangqian

                if self._tiaozheng is not None:
                    mubiao, dan_bu = self._tiaozheng
                    self._tiaozheng = None
                    xian = self._dao_mubiao(cap, mubiao, dan_bu)
                    if xian is not None:
                        self._dangqian = mubiao
                        self._cun_zhen(mubiao, xian)
                        self._zui_xin = (xian, mubiao)
                    ji_zhun = time.perf_counter()
                    ji_zhun_zhen = self._dangqian
                    if self._zanting:
                        continue

                if self._zanting:
                    self.msleep(10)
                    ji_zhun = time.perf_counter()
                    ji_zhun_zhen = self._dangqian
                    continue

                chenggong, zhen = cap.read()
                if not chenggong or zhen is None:
                    self.daowei.emit()
                    break
                self._dangqian += 1
                self._xiaci = self._dangqian + 1
                if self.zhi_tuili:
                    # 只喂模型：读到画面前头去了就等 MPV 一下，别跑飞
                    self._deng_bofang()
                    self._song_zhen(zhen, self._dangqian)
                    continue
                # 原始帧先交给推理（钩子里只是丢进槽位，很快，不拖慢播放）
                self._song_zhen(zhen, self._dangqian)
                xian = self._zhen_zhuan(zhen)
                self._cun_zhen(self._dangqian, xian)
                self._zui_xin = (xian, self._dangqian)

                # 关键：按"这一帧本该出现的绝对时刻"校准，而不是按上一帧花了多久
                # 来睡。msleep 在 Windows 上总会多睡几毫秒，用相对睡眠会一帧帧累积
                # 误差，越播越慢（慢动作）；用绝对时刻则下一帧自动把欠账补回来。
                su_du = max(1e-6, self.fps * self._beisu)
                mubiao_shi = ji_zhun + (
                    self._dangqian - ji_zhun_zhen
                ) / su_du
                shengyu = mubiao_shi - time.perf_counter()
                if shengyu > 0.0015:
                    self.msleep(max(1, int(shengyu * 1000)))
        finally:
            cap.release()
            _huifu_jingdu()


# ---------------------------------------------------------------------
# 音频波形：解码线程
# ---------------------------------------------------------------------
_FFMPEG_LUJING = None


def zhao_ffmpeg():
    """找 ffmpeg：优先 imageio-ffmpeg 自带的，其次系统 PATH 里的"""
    global _FFMPEG_LUJING
    if _FFMPEG_LUJING is None:
        lujing = ""
        try:
            import imageio_ffmpeg  # noqa

            lujing = imageio_ffmpeg.get_ffmpeg_exe() or ""
        except Exception:  # noqa
            lujing = ""
        if not lujing or not osp.isfile(lujing):
            lujing = shutil.which("ffmpeg") or ""
        _FFMPEG_LUJING = lujing
    return _FFMPEG_LUJING


class _BofangXiancheng(QtCore.QThread):
    """后台用 ffmpeg 解音轨，边解边算每格的最大/最小振幅（画波形用）"""

    kuaijie = pyqtSignal(int, object, object)   # 起始格号, 最大数组, 最小数组
    jieshu = pyqtSignal(str)                    # 空串 = 解完；否则是出错原因

    def __init__(self, lujing, parent=None):
        super().__init__(parent)
        self.lujing = lujing
        self._yunxing = True
        self._jincheng = None
        self._mei_ge = max(
            1, int(round(BO_CAN_YANG_LV * BO_HAO_MIAO / 1000.0))
        )

    def tingzhi(self):
        self._yunxing = False
        if self._jincheng is not None:
            try:
                self._jincheng.kill()
            except Exception:  # noqa
                pass
            self._jincheng = None
        self.wait(3000)

    def run(self):
        ff = zhao_ffmpeg()
        if not ff:
            self.jieshu.emit("没找到 ffmpeg（可 pip install imageio-ffmpeg）")
            return
        mingling = [
            ff,
            "-nostdin",
            "-v",
            "error",
            "-i",
            self.lujing,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(BO_CAN_YANG_LV),
            "-f",
            "s16le",
            "-",
        ]
        biaozhi = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            biaozhi = subprocess.CREATE_NO_WINDOW
        try:
            self._jincheng = subprocess.Popen(
                mingling,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=biaozhi,
            )
        except Exception as e:  # noqa
            self.jieshu.emit(f"启动 ffmpeg 失败：{e}")
            return

        mei = self._mei_ge
        buf = b""
        ge_hao = 0
        try:
            while self._yunxing:
                shuju = self._jincheng.stdout.read(65536)
                if not shuju:
                    break
                buf += shuju
                zong = len(buf) // 2
                if zong < mei:
                    continue
                ge_shu = zong // mei
                a = np.frombuffer(buf[: ge_shu * mei * 2], dtype="<i2")
                a = a.reshape(ge_shu, mei)
                self.kuaijie.emit(
                    ge_hao,
                    a.max(axis=1).astype(np.int16),
                    a.min(axis=1).astype(np.int16),
                )
                ge_hao += ge_shu
                buf = buf[ge_shu * mei * 2:]
                # 每解一块歇一下：不然 ffmpeg 会把一个核吃满，把画面解码挤掉，
                # 打开视频的头十几秒画面就一直卡。
                if JIE_LIU_HAO_MIAO > 0:
                    self.msleep(JIE_LIU_HAO_MIAO)
        except Exception as e:  # noqa
            self._yunxing = False
            self.jieshu.emit(f"解析音频出错：{e}")
            return
        finally:
            if self._jincheng is not None:
                try:
                    self._jincheng.kill()
                except Exception:  # noqa
                    pass
                self._jincheng = None
        if not self._yunxing:
            return
        if ge_hao <= 0:
            self.jieshu.emit("这段视频没有音轨")
            return
        self.jieshu.emit("")


# ---------------------------------------------------------------------
# 视频下方那条刻度进度条
#   刻度：照 Aegisub 的音频刻度尺（src/audio_display.cpp 的
#         AudioDisplayTimeline::Paint）—— 按"每秒占多少像素"决定一格代表
#         多少毫秒、每几格算一个主刻度；刻度贴着底边往上画，长的 6px、
#         短的 3px；只有主刻度写时间，写不下（要压到上一个了）就跳过。
#   指针：照 Aegisub 视频窗口的滑块（src/video_slider.cpp 的
#         VideoSlider::OnPaint）—— 一个朝上的小三角加一条竖线。
#   点 / 拖 = 定位。
# ---------------------------------------------------------------------
JINDU_TIAO_GAO = 14                      # 进度条高度（只有刻度 + 指针，不写时间）


class ShipinJinduTiao(QtWidgets.QWidget):
    """视频下方的刻度进度条：整条片子的进度（不跟时间轴缩放走）"""

    dingwei = pyqtSignal(int)     # 点 / 拖 -> 要定位到的毫秒
    tuo_kaishi = pyqtSignal()     # 鼠标按下去、准备拖播放头了
    tuo_jieshu = pyqtSignal()     # 手松了（拖播放头结束）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(JINDU_TIAO_GAO)
        self.setMinimumWidth(120)
        self.setCursor(Qt.PointingHandCursor)
        self._shichang = 0        # 整条多长（毫秒）
        self._dangqian = 0        # 播放头在哪（毫秒）
        self._tuozhe = False

    # ---- 外面喂数据 ----
    def shezhi_shichang(self, ms):
        self._shichang = max(0, int(ms))
        self.update()

    def shezhi_bofangtou(self, ms):
        ms = max(0, int(ms))
        if ms == self._dangqian:
            return
        self._dangqian = ms
        self.update()

    # ---- 鼠标：点 / 拖定位 ----
    def _x_dui_hao_miao(self, x):
        kuan = self.width() - 10
        if kuan <= 0 or self._shichang <= 0:
            return 0
        bi = (x - 5) / float(kuan)
        return int(round(max(0.0, min(1.0, bi)) * self._shichang))

    def mousePressEvent(self, shi_jian):
        if shi_jian.button() == Qt.LeftButton:
            self._tuozhe = True
            self.tuo_kaishi.emit()
            self.dingwei.emit(self._x_dui_hao_miao(shi_jian.x()))

    def mouseMoveEvent(self, shi_jian):
        if self._tuozhe:
            self.dingwei.emit(self._x_dui_hao_miao(shi_jian.x()))

    def mouseReleaseEvent(self, shi_jian):
        if shi_jian.button() == Qt.LeftButton:
            if self._tuozhe:
                self._tuozhe = False
                self.tuo_jieshu.emit()
                # 松手这一下再定一次位：拖动过程里音视频都没跟着走，
                # 这回才真正落到帧上（画面 + 音频一次到位）
                self.dingwei.emit(self._x_dui_hao_miao(shi_jian.x()))

    # ---- 一格代表多少毫秒（照 AEG 按每秒像素数选）----
    def _dan_wei(self):
        kuan = max(1, self.width() - 10)
        miao = self._shichang / 1000.0
        if miao <= 0:
            return 1000, 10
        px_sec = kuan / miao
        for yu, dan, mo in (
            (3000.0, 1, 10),           # 毫秒
            (300.0, 10, 10),           # 厘秒
            (30.0, 100, 10),           # 分秒
            (3.0, 1000, 10),           # 秒
            (1.0 / 3.0, 10000, 6),     # 十秒
            (1.0 / 9.0, 60000, 10),    # 分钟
            (1.0 / 90.0, 600000, 6),   # 十分钟
        ):
            if px_sec > yu:
                return dan, mo
        return 3600000, 10             # 小时

    def paintEvent(self, _shi_jian):
        c = _ys()
        hua = QtGui.QPainter(self)
        kuan, gao = self.width(), self.height()
        di = gao - 1
        # 底边线
        hua.setPen(QtGui.QPen(QtGui.QColor(c["biankuang_liang"])))
        hua.drawLine(0, di, kuan, di)
        if self._shichang <= 0 or kuan <= 12:
            return

        dan, mo = self._dan_wei()
        zong = float(self._shichang)
        kuai = float(kuan - 10)

        # 刻度线：跟 Aegisub 的刻度条一样只有长短刻度，不写时间
        hao = 0
        hua.setPen(QtGui.QPen(QtGui.QColor(c["wenzi_ci"])))
        while True:
            x = 5 + int(round(hao / zong * kuai))
            if x > kuan - 5:
                break
            if int(round(hao / float(dan))) % mo == 0:
                hua.drawLine(x, di - 6, x, di - 1)
            else:
                hua.drawLine(x, di - 4, x, di - 1)
            hao += dan

        # 指针：一条蓝柱子，压着刻度（不要箭头）
        x = 5 + int(round(self._dangqian / zong * kuai))
        yan = QtGui.QColor(c["zhuse"])
        hua.setPen(QtGui.QPen(yan))
        hua.setBrush(QtGui.QBrush(yan))
        hua.drawRect(x - 1, 1, 3, di - 1)


# ---------------------------------------------------------------------
# 视频画面
# ---------------------------------------------------------------------
class HuamianQu(QtWidgets.QOpenGLWidget):
    """视频画面：mpv 把视频直接渲染到这块 OpenGL 画布上

    画面上叠的东西（检测区域虚线、检测框、字幕）还是我们用 QPainter 画，
    跟以前一模一样 —— 只是底下那张视频不再是我们自己贴的图了。

    另外留了一条老路：外面把整张图塞进来（全片扫描时那样）就直接画那张图，
    这时候不让 mpv 插手，免得两边打架。
    """

    chicun_bian = pyqtSignal()      # 控件尺寸变了 -> 外面重算该把帧缩到多大
    quyu_bian = pyqtSignal(object)  # 框选完了 -> (x,y,w,h) 原始像素；没框成 -> None
    xin_zhen_dao = pyqtSignal(bool)  # mpv 那边说该重画了（真出新帧 -> True）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._tupian = None
        self._shipin_kuan = 0        # 视频原始宽高（不依赖当前帧就能算显示尺寸）
        self._shipin_gao = 0
        self._kuang = []             # 检测框 [(名字, 分数, (x1,y1,x2,y2))] 原始像素
        self._quyu = None            # 检测区域 (x,y,w,h) 原始像素；None = 整图
        self._zimu = []              # 现在要叠在画面上的字幕（外面算好塞进来）
        self._zimu_jizhun = HUA_ZIMU_JIZHUN   # 字幕坐标的基准分辨率
        self._zai_kuang_xuan = False  # 是否正处于框选状态
        self._tuo_qi = None          # 拖拽起点（原始像素）
        self._tuo_dao = None         # 拖拽终点（原始像素）
        # mpv 渲染上下文（建一次，关窗口时释放）
        self._mpv = None             # MpvBofang
        self._render = None
        self._hui_c = None           # mpv 的回调，得留着引用，不然会被回收
        self._gl_qu = None
        self._guan_le = False        # 已经松开了，mpv 再回调就别理它
        # 字幕层缓存：(指纹, QImage)。画面上叠的那层字幕逐字量宽度、折行、
        # 描边加填充，比贴一张现成的图贵几十倍，而它只在"换条 / 淡入淡出
        # 进度变了 / 画面尺寸变了"时才真不一样 —— 播放时一秒 30 帧里绝大
        # 多数帧画出来的东西一个像素都不差，白画。
        self._zimu_tu = None
        # mpv 每出来一帧、重画之前先叫外面一声：外面按 mpv 这一帧的时刻把
        # 画面上那层字幕换好（拖动时字幕得跟画面一起出，不能跑在画面前头）
        self._huan_zhen_hui = None
        self.xin_zhen_dao.connect(self._shuo_hua, Qt.QueuedConnection)
        self.setMouseTracking(True)

    # ---- 检测区域 ----
    def quyu(self):
        return self._quyu

    def shezhi_quyu(self, quyu):
        """从外面同步区域（面板清空/换视频时用）"""
        self._quyu = tuple(int(v) for v in quyu) if quyu else None
        self.update()

    def zai_kuang_xuan(self):
        return self._zai_kuang_xuan

    def kaishi_kuang_xuan(self):
        self._zai_kuang_xuan = True
        self._tuo_qi = None
        self._tuo_dao = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def tingzhi_kuang_xuan(self):
        self._zai_kuang_xuan = False
        self._tuo_qi = None
        self._tuo_dao = None
        self.unsetCursor()
        self.update()

    # ---- 坐标换算：控件像素 <-> 视频原始像素 ----
    def _suofang_bi(self):
        """原始像素 -> 控件像素的比例；算不出来给 0"""
        mubiao = self.mubiao_qu()
        iw = self._shipin_kuan
        if iw <= 0 and self._tupian is not None and not self._tupian.isNull():
            iw = self._tupian.width()
        if iw <= 0 or mubiao.width() <= 0:
            return 0.0, mubiao
        return mubiao.width() / float(iw), mubiao

    def _dao_yuanshi(self, dian):
        """控件里的点 -> 视频原始像素（夹在画面内）"""
        bi, mubiao = self._suofang_bi()
        if bi <= 0:
            return None
        iw = self._shipin_kuan or self._tupian.width()
        ih = self._shipin_gao or self._tupian.height()
        x = (dian.x() - mubiao.left()) / bi
        y = (dian.y() - mubiao.top()) / bi
        return (
            max(0.0, min(float(iw), x)),
            max(0.0, min(float(ih), y)),
        )

    def _tuo_chu_de_qu(self):
        """拖出来的矩形 -> (x,y,w,h) 原始像素；太小当没拖"""
        if self._tuo_qi is None or self._tuo_dao is None:
            return None
        x1, y1 = self._tuo_qi
        x2, y2 = self._tuo_dao
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        if w < 8 or h < 8:
            return None
        return (int(round(x1)), int(round(y1)), w, h)

    # ---- 鼠标：框选 ----
    def mousePressEvent(self, event):
        if self._zai_kuang_xuan and event.button() == Qt.LeftButton:
            dian = self._dao_yuanshi(event.pos())
            if dian is not None:
                self._tuo_qi = dian
                self._tuo_dao = dian
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._zai_kuang_xuan and self._tuo_qi is not None:
            dian = self._dao_yuanshi(event.pos())
            if dian is not None:
                self._tuo_dao = dian
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._zai_kuang_xuan
            and event.button() == Qt.LeftButton
            and self._tuo_qi is not None
        ):
            dian = self._dao_yuanshi(event.pos())
            if dian is not None:
                self._tuo_dao = dian
            qu = self._tuo_chu_de_qu()
            self.tingzhi_kuang_xuan()
            if qu:
                self._quyu = qu
                self.quyu_bian.emit(qu)
            else:
                self.quyu_bian.emit(None)   # 只是点了一下，不改已有区域
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def shezhi_kuang(self, kuang):
        """更新检测框（视频原始像素坐标），空列表 = 清掉"""
        self._kuang = list(kuang or [])
        self.update()

    def qingkong_kuang(self):
        if self._kuang:
            self._kuang = []
            self.update()

    def shezhi_shipin_chicun(self, kuan, gao):
        self._shipin_kuan = max(0, int(kuan or 0))
        self._shipin_gao = max(0, int(gao or 0))
        self.chicun_bian.emit()

    def yuan_chicun(self):
        """视频原始尺寸 -> (宽, 高)；没视频给 None"""
        if self._shipin_kuan <= 0 or self._shipin_gao <= 0:
            return None
        return (self._shipin_kuan, self._shipin_gao)

    def xianshi_chicun(self):
        """该把帧缩到多大 -> (宽, 高)；算不出来给 None"""
        qu = self.mubiao_qu()
        if qu.width() <= 8 or qu.height() <= 8:
            return None
        return (qu.width(), qu.height())

    def shezhi_zhen(self, qimage):
        self._tupian = qimage
        self.update()

    def shezhi_zimu(self, tiao, jizhun=None):
        """画面里要叠的字幕：[{"hang": [文字行...], "gs": 格式}]，空列表 = 不画

        外面（主窗口）按播放头算好再塞进来，这儿只管画。一样就不重画。
        """
        if jizhun:
            kuan = max(1, int(jizhun[0]))
            gao = max(1, int(jizhun[1]))
            self._zimu_jizhun = (kuan, gao)
        tiao = list(tiao or [])
        if tiao == self._zimu:
            return
        self._zimu = tiao
        self.update()

    def qingkong(self):
        self._tupian = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.chicun_bian.emit()

    def mubiao_qu(self):
        """画面在控件里的实际显示矩形"""
        w = max(1, self.width())
        h = max(1, self.height())
        iw = self._shipin_kuan
        ih = self._shipin_gao
        if iw <= 0 or ih <= 0:
            if self._tupian is None or self._tupian.isNull():
                return QtCore.QRect()
            iw = max(1, self._tupian.width())
            ih = max(1, self._tupian.height())
        bi = min(w / float(iw), h / float(ih))
        tw = max(1, int(round(iw * bi)))
        th = max(1, int(round(ih * bi)))
        return QtCore.QRect((w - tw) // 2, (h - th) // 2, tw, th)

    # ---- mpv：把视频渲染进这块画布 ----
    def shezhi_mpv(self, mpv):
        """接上播放内核（真正的渲染上下文等 GL 起来了再建，见 zhunbei_mpv）"""
        self._mpv = mpv

    def zhunbei_mpv(self):
        """保证渲染上下文已建好

        必须赶在载入视频之前建好 —— mpv 那边要是载入时还没看到渲染上下文，
        它就自己去开一个窗口（那画面就跑到我们控件外面去了）。
        """
        if self._render is not None:
            return True
        try:
            # 先把 GL 上下文拉起来（Qt 会在这一步把 initializeGL 叫起来）
            self.makeCurrent()
            self.doneCurrent()
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：OpenGL 还没准备好（{cuowu}）")
        self._jian_render()
        return self._render is not None

    def initializeGL(self):
        self._jian_render()

    def _jian_render(self):
        """建 mpv 的渲染上下文（得 GL 已就绪 + 已有内核句柄，两个都齐了才建）"""
        if self._render is not None or self._mpv is None:
            return
        handle = getattr(self._mpv, "h", None)
        if handle is None:
            return
        d = jiazai_mpv()
        if d is None:
            return
        try:
            self.makeCurrent()
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：拿不到 OpenGL 上下文（{cuowu}）")
            return
        try:
            self._gl_qu = ctypes.CFUNCTYPE(
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
            )(_qu_gl_hanshu)
            can = _MpvOpenglInitParams(
                ctypes.cast(self._gl_qu, ctypes.c_void_p), None
            )
            can_shu = (_MpvRenderParam * 3)(
                _MpvRenderParam(
                    MPV_RENDER_PARAM_API_TYPE,
                    ctypes.cast(_API_TYPE_OPENGL, ctypes.c_void_p),
                ),
                _MpvRenderParam(
                    MPV_RENDER_PARAM_OPENGL_INIT_PARAMS,
                    ctypes.cast(ctypes.byref(can), ctypes.c_void_p),
                ),
                _MpvRenderParam(MPV_RENDER_PARAM_INVALID, None),
            )
            zhi = ctypes.c_void_p()
            hao = d.mpv_render_context_create(
                ctypes.byref(zhi), handle, can_shu
            )
            if hao < 0:
                logger.error(
                    f"视频工作台：mpv 渲染上下文建不起来（错误码 {hao}）"
                )
                return
            self._render = zhi
            # mpv 那边说有新画面就回主线程重画（回调是它自己的线程调的）
            self._hui_c = ctypes.CFUNCTYPE(
                None, ctypes.c_void_p
            )(self._mpv_shuo_hua)
            d.mpv_render_context_set_update_callback(
                self._render, ctypes.cast(self._hui_c, ctypes.c_void_p), None
            )
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：建 mpv 渲染上下文出错（{cuowu}）")
        finally:
            self.doneCurrent()

    def _mpv_shuo_hua(self, *_a):
        """mpv 说"该重画了"（这是它自己的线程）

        顺手问一句"是不是真新出了一帧视频"：定位（seek）还没走完的时候 mpv
        也会叫我们重画，可画面上还是老那一帧。那种不算新帧 —— 字幕得等真出
        新帧才换，这样画面和字幕永远对在同一个时刻上。
        """
        if self._guan_le:
            return
        xin_zhen = False
        try:
            d = jiazai_mpv()
            if d is not None and self._render:
                xin_zhen = bool(
                    d.mpv_render_context_update(self._render)
                    & MPV_RENDER_UPDATE_FRAME
                )
        except Exception:  # noqa
            pass
        try:
            self.xin_zhen_dao.emit(xin_zhen)
        except Exception:  # noqa
            pass

    def shezhi_huan_zhen_hui(self, hui):
        """外面接这个回调：mpv 每出一帧、重画之前先叫一声（在主线程里叫）"""
        self._huan_zhen_hui = hui

    def _shuo_hua(self):
        """mpv 那边该重画了（已经回到主线程）

        这儿只管叫重画。画面上那层字幕该不该换，挪到 paintGL 里判：那儿能
        先问清楚"这一帧到底是不是真新解出来的视频帧"（见 _zhe_yi_zhen_shi_xin_zhen），
        问准了才重算 —— 而且和画面是同一笔画出去的，一起出来。
        """
        self.update()

    def _rang_mpv_hua(self):
        """让 mpv 把当前这一帧渲染到这块画布上（没有视频就什么都不做）"""
        if self._render is None:
            return
        d = jiazai_mpv()
        if d is None:
            return
        bei = self.devicePixelRatioF()
        kuan = max(1, int(round(self.width() * bei)))
        gao = max(1, int(round(self.height() * bei)))
        fbo = _MpvOpenglFbo(
            int(self.defaultFramebufferObject()), kuan, gao, 0
        )
        fan = ctypes.c_int(1)
        can_shu = (_MpvRenderParam * 3)(
            _MpvRenderParam(
                MPV_RENDER_PARAM_OPENGL_FBO,
                ctypes.cast(ctypes.byref(fbo), ctypes.c_void_p),
            ),
            _MpvRenderParam(
                MPV_RENDER_PARAM_FLIP_Y,
                ctypes.cast(ctypes.byref(fan), ctypes.c_void_p),
            ),
            _MpvRenderParam(MPV_RENDER_PARAM_INVALID, None),
        )
        d.mpv_render_context_render(self._render, can_shu)
        try:
            d.mpv_render_context_report_swap(self._render)
        except Exception:  # noqa
            pass

    def shifang(self):
        """松开 mpv 的渲染上下文（关窗口时调，得先让 GL 上下文当前）"""
        if self._render is None:
            return
        self._guan_le = True
        d = jiazai_mpv()
        self.makeCurrent()
        try:
            if d is not None:
                d.mpv_render_context_free(self._render)
        except Exception:  # noqa
            pass
        finally:
            self._render = None
            self._hui_c = None
            self.doneCurrent()

    def _hei_di(self):
        """先把画布刷黑（letterbox 那两条黑边）"""
        global _OPENGL32
        if _OPENGL32 is None:
            _qu_gl_hanshu(None, b"glClearColor")     # 顺手把它载进来
        if not _OPENGL32:
            return
        try:
            _OPENGL32.glClearColor(0.0, 0.0, 0.0, 1.0)
            _OPENGL32.glClear(GL_COLOR_BUFFER_BIT)
        except Exception:  # noqa
            pass

    def _zhe_yi_zhen_shi_xin_zhen(self):
        """接下来要画的这一帧，是不是真解出来的新视频帧

        问 mpv 的渲染接口（get_info + 帧信息里的标记位）。它要是说"重画"或
        "重复上一帧"，那就不是新帧 —— 那会儿画面上还是老那一帧，字幕跟着换
        就跑到画面前头去了。问不出来（老版本 mpv）就一律当新帧，退回老样子，
        宁可字幕跟着时间走，也别卡着不换。
        """
        if self._render is None:
            return False
        d = jiazai_mpv()
        if d is None or not hasattr(d, "mpv_render_context_get_info"):
            return True
        xinxi = _MpvZhenXinxi()
        can = _MpvRenderParam(
            MPV_RENDER_PARAM_NEXT_FRAME_INFO,
            ctypes.cast(ctypes.byref(xinxi), ctypes.c_void_p),
        )
        try:
            hao = d.mpv_render_context_get_info(self._render, can)
        except Exception:  # noqa
            return True
        if hao < 0:
            return True
        if not (xinxi.biaozhi & ZHEN_PRESENT):
            return False
        return not (xinxi.biaozhi & (ZHEN_REDRAW | ZHEN_REPEAT))

    def paintGL(self):
        self._jian_render()          # 头一回画的时候把渲染上下文建上
        tupian = self._tupian
        if tupian is not None and tupian.isNull():
            tupian = None
        if tupian is None:
            # 视频走 mpv，我们只把底刷黑 + 叠自己的东西。
            # 先问清楚这一帧是不是真新帧（得赶在画之前问：它说的是"接下来要
            # 画的这一帧"），画完再按 mpv 现在的位置重算画面上那层字幕 ——
            # 同一笔画出去，字幕和画面就是同一刻出来的，不会字幕先跑。
            shi_xin = self._zhe_yi_zhen_shi_xin_zhen()
            self._hei_di()
            self._rang_mpv_hua()
        else:
            # 整张图塞进来的老路（全片扫描）：每来一张都当新的一帧
            shi_xin = True
        if shi_xin:
            hui = self._huan_zhen_hui
            if hui is not None:
                try:
                    hui()
                except Exception:  # noqa
                    pass
        huabi = QtGui.QPainter(self)
        mubiao = self.mubiao_qu()
        if tupian is not None:
            # 整张图塞进来的老路（全片扫描）：自己贴图
            if (
                tupian.width() == mubiao.width()
                and tupian.height() == mubiao.height()
            ):
                huabi.drawImage(mubiao.topLeft(), tupian)
            else:
                huabi.setRenderHint(
                    QtGui.QPainter.SmoothPixmapTransform, True
                )
                huabi.drawImage(mubiao, tupian)
        # 先画检测区域（框外压暗），再把检测框压在上面，最后叠字幕
        self._hua_quyu(huabi, mubiao)
        self._hua_kuang(huabi, mubiao)
        tu = self._zimu_ceng(mubiao)
        if tu is not None:
            huabi.drawImage(mubiao.topLeft(), tu)
        huabi.end()

    def _zimu_ceng(self, mubiao):
        """画面上那层字幕：整层合成一张图缓存着，每帧只贴这张图

        指纹里用的是算好的透明度倍数，不是毫秒 —— 这样 \\fad 渐入渐出的
        时候该重画还得重画，不给它加 fad 的时候一动不动。
        """
        if not self._zimu or mubiao.width() <= 8 or mubiao.height() <= 8:
            return None
        qian = _zimu_qian(self._zimu, mubiao, self._zimu_jizhun)
        jiu = self._zimu_tu
        if jiu is not None and jiu[0] == qian:
            return jiu[1]
        bi = self.devicePixelRatioF() or 1.0
        tu = QtGui.QImage(
            max(1, int(round(mubiao.width() * bi))),
            max(1, int(round(mubiao.height() * bi))),
            QtGui.QImage.Format_ARGB32_Premultiplied,
        )
        tu.setDevicePixelRatio(bi)
        tu.fill(Qt.transparent)
        huabi = QtGui.QPainter(tu)
        qu = QtCore.QRect(0, 0, mubiao.width(), mubiao.height())
        for tiao in self._zimu:
            _hua_ass_tiao(huabi, qu, tiao, self._zimu_jizhun)
        huabi.end()
        self._zimu_tu = (qian, tu)
        return tu

    def _hua_quyu(self, huabi, mubiao):
        """画检测区域：只有一圈虚线框，不动画面；正在拖的时候显示拖出来的框"""
        qu = self._quyu
        if self._zai_kuang_xuan and self._tuo_qi is not None:
            qu = self._tuo_chu_de_qu()
        if not qu or mubiao.width() <= 0:
            return
        iw = self._shipin_kuan
        if iw <= 0 and self._tupian is not None and not self._tupian.isNull():
            iw = self._tupian.width()
        if iw <= 0:
            return
        bi = mubiao.width() / float(iw)
        x1 = mubiao.left() + qu[0] * bi
        y1 = mubiao.top() + qu[1] * bi
        w = qu[2] * bi
        h = qu[3] * bi
        bi_hu = QtGui.QPen(QtGui.QColor(YANSE_QUYU), 2)
        bi_hu.setStyle(Qt.DashLine)
        huabi.setPen(bi_hu)
        huabi.setBrush(Qt.NoBrush)
        huabi.drawRect(QtCore.QRectF(x1, y1, w, h))

    def _hua_kuang(self, huabi, mubiao):
        """按视频原始像素 -> 屏幕像素的比例，把检测框画上去"""
        if not self._kuang:
            return
        iw = self._shipin_kuan or (self._tupian.width() if self._tupian else 0)
        if iw <= 0 or mubiao.width() <= 0:
            return
        bi = mubiao.width() / float(iw)
        huabi.setRenderHint(QtGui.QPainter.Antialiasing, True)
        ziti = QtGui.QFont("Microsoft YaHei", 9)
        huabi.setFont(ziti)
        zhun = huabi.fontMetrics()
        for mingzi, fenshu, kuang in self._kuang:
            x1 = mubiao.left() + kuang[0] * bi
            y1 = mubiao.top() + kuang[1] * bi
            x2 = mubiao.left() + kuang[2] * bi
            y2 = mubiao.top() + kuang[3] * bi
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            huabi.setPen(QtGui.QPen(QtGui.QColor(YANSE_KUANG), 2))
            huabi.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))

            if not mingzi:
                continue
            wenben = (
                f"{mingzi} {fenshu:.2f}" if fenshu is not None else str(mingzi)
            )
            kuan_t = zhun.horizontalAdvance(wenben) + 8
            gao_t = zhun.height()
            zuo = min(max(x1, mubiao.left()), mubiao.right() - kuan_t)
            shang = max(mubiao.top(), y1 - gao_t)
            bei = QtCore.QRectF(zuo, shang, kuan_t, gao_t)
            huabi.fillRect(bei, QtGui.QColor(0, 0, 0, 165))
            huabi.setPen(QtGui.QColor(YANSE_KUANG_ZI))
            huabi.drawText(bei, Qt.AlignCenter, wenben)


# ---------------------------------------------------------------------
# 读 / 写字幕文件（SRT、ASS）
# ---------------------------------------------------------------------
_SRT_SHIJIAN = re.compile(
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_ASS_SHIJIAN = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[.:](\d{1,2})")


def _du_wenben(lu):
    """读文本文件，尽量猜对编码：带 BOM 的 UTF-16 -> UTF-8 -> GBK"""
    with open(lu, "rb") as f:
        shuju = f.read()
    if shuju[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return shuju.decode("utf-16")
    try:
        return shuju.decode("utf-8-sig")
    except UnicodeDecodeError:
        return shuju.decode("gb18030", errors="replace")


def _shi_fen_miao_hao_ms(sh, fen, miao, hao):
    """时:分:秒,毫秒 -> 毫秒（毫秒不足 3 位按 3 位补）"""
    hao = (str(hao) + "000")[:3]
    return (int(sh) * 3600 + int(fen) * 60 + int(miao)) * 1000 + int(hao)


def _jiexi_srt(wenben):
    """SRT 文本 -> [(起ms, 止ms, 文字), ...]"""
    zimu = []
    for kuai in re.split(r"\r?\n\s*\r?\n", (wenben or "").strip()):
        hang = [h for h in kuai.splitlines() if h.strip()]
        if not hang:
            continue
        if hang[0].strip().isdigit():       # 序号行，丢掉
            hang = hang[1:]
        if not hang:
            continue
        pi = _SRT_SHIJIAN.search(hang[0])
        if not pi:
            continue
        qi = _shi_fen_miao_hao_ms(pi.group(1), pi.group(2),
                                  pi.group(3), pi.group(4))
        zhi = _shi_fen_miao_hao_ms(pi.group(5), pi.group(6),
                                   pi.group(7), pi.group(8))
        zimu.append((qi, zhi, "\n".join(hang[1:]).strip()))
    return zimu


def _jiexi_ass(wenben):
    """ASS / SSA 文本 -> [(起ms, 止ms, 文字), ...]（注释行也算进来）"""
    zimu = []
    for hang in (wenben or "").splitlines():
        hang = hang.strip()
        if not hang.lower().startswith(("dialogue:", "comment:")):
            continue
        # Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        bu = hang.split(":", 1)[1].split(",", 9)
        if len(bu) < 10:
            continue
        m1 = _ASS_SHIJIAN.search(bu[1])
        m2 = _ASS_SHIJIAN.search(bu[2])
        if not (m1 and m2):
            continue
        qi = (int(m1.group(1)) * 3600 + int(m1.group(2)) * 60
              + int(m1.group(3))) * 1000 + int(m1.group(4)) * 10
        zhi = (int(m2.group(1)) * 3600 + int(m2.group(2)) * 60
               + int(m2.group(3))) * 1000 + int(m2.group(4)) * 10
        wen = bu[9].replace("\\N", "\n").replace("\\n", "\n")
        # 原样留着 {\pos(75,705)} 这些行内标签：编辑区要照着 ASS 显示全代码，
        # 画面里叠字幕也直接拿这一串去认（要去干净文字用 _ass_chun_wen）
        zimu.append((qi, zhi, wen.strip()))
    return zimu


def _du_gao_ji():
    """读「字幕块每行高 / 波形放大倍数」（跟界面记忆一起存在 gongzuotai.ini 里）"""
    try:
        pei = buju_qsettings()
        ji = {}
        for ming, jian_ming in (
            ("meihang_gao", JIEMIAN_MEIHANG_GAO),
            ("bo_fangda", JIEMIAN_BO_FANGDA),
        ):
            zhi = pei.value(jian_ming, None)
            if zhi is not None:
                ji[ming] = zhi
        return ji
    except Exception:  # noqa
        return {}


def _cun_gao_ji(**xiang):
    """存「字幕块每行高 / 波形放大倍数」（某项传 None = 恢复默认，直接抹掉）"""
    try:
        pei = buju_qsettings()
        for ming, zhi in xiang.items():
            jian_ming = {
                "meihang_gao": JIEMIAN_MEIHANG_GAO,
                "bo_fangda": JIEMIAN_BO_FANGDA,
            }.get(ming)
            if jian_ming is None:
                continue
            if zhi is None:
                pei.remove(jian_ming)
            else:
                pei.setValue(jian_ming, zhi)
        pei.sync()
    except Exception as cuowu:  # noqa
        logger.error(f"视频工作台：记录高度失败 {cuowu}")


def _du_zimu_gao():
    """读上次调好的每行字幕块高度；没记录就返回 None（用默认高度）"""
    try:
        gao = int(_du_gao_ji().get("meihang_gao") or 0)
    except Exception:  # noqa
        return None
    return gao if gao > 0 else None


def _cun_zimu_gao(gao):
    """记下每行字幕块高度（传 None = 恢复默认，直接抹掉记录）"""
    _cun_gao_ji(meihang_gao=None if gao is None else int(gao))


def _du_bo_fangda():
    """读上次调好的波形放大倍数；没记录就返回 None（用默认倍数）"""
    try:
        fang = float(_du_gao_ji().get("bo_fangda") or 0)
    except Exception:  # noqa
        return None
    if fang <= 0:
        return None
    return max(BO_FANGDA_ZUI_XIAO, min(fang, BO_FANGDA_ZUIDA))


def _du_zimu_wenjian(lu):
    """读一份字幕文件 -> [(起ms, 止ms, 文字), ...]，按时间排好序"""
    wenben = _du_wenben(lu)
    if osp.splitext(lu)[1].lower() in (".ass", ".ssa"):
        zimu = _jiexi_ass(wenben) or _jiexi_srt(wenben)
    else:
        zimu = _jiexi_srt(wenben) or _jiexi_ass(wenben)
    zimu.sort(key=lambda x: (x[0], x[1]))
    return zimu


def _xie_zimu_wenjian(lu, zimu):
    """把字幕写到指定文件：.ass 按 ASS 格式，其余按 SRT"""
    from anylabeling.views.labeling.utils.video import (
        ASS_TEMPLATE,
        format_ass_time,
        format_srt_time,
    )

    if osp.splitext(lu)[1].lower() in (".ass", ".ssa"):
        with open(lu, "w", encoding="utf-8-sig") as f:
            f.write(ASS_TEMPLATE)
            for qi, zhi, wenben in zimu:
                # ASS 的换行必须写成 \N，直接写换行会把一条拆成两条
                wenben = str(wenben or "").replace("\n", "\\N")
                f.write(
                    f"Dialogue: 0,{format_ass_time(qi)},"
                    f"{format_ass_time(zhi)},Default,,0,0,0,,{wenben}\n"
                )
    else:
        with open(lu, "w", encoding="utf-8") as f:
            for xu, (qi, zhi, wenben) in enumerate(zimu, 1):
                f.write(f"{xu}\n")
                f.write(f"{format_srt_time(qi)} --> {format_srt_time(zhi)}\n")
                # SRT 放不下 ASS 的行内标签，写出去之前先摘干净
                f.write(f"{_ass_chun_wen(wenben)}\n\n")


# ASS 事件行的标准字段（文件里写了 Format 行就按它自己的来）
_ASS_ZIDUAN = [
    "Layer", "Start", "End", "Style", "Name",
    "MarginL", "MarginR", "MarginV", "Effect", "Text",
]


def _ass_hao_ms(wenben):
    """ASS 的 0:00:00.00 -> 毫秒；认不出来给 None"""
    pi = _ASS_SHIJIAN.search(str(wenben or ""))
    if not pi:
        return None
    return (
        (int(pi.group(1)) * 3600 + int(pi.group(2)) * 60 + int(pi.group(3)))
        * 1000
        + int(pi.group(4)) * 10
    )


def _ass_chun_wen(wenben):
    """ASS 的 Text 归一（\\N 还原成换行、去掉 {\\...} 样式标签），跟读进来时一个规矩"""
    wen = str(wenben or "").replace("\\N", "\n").replace("\\n", "\n")
    return re.sub(r"\{[^}]*\}", "", wen).strip()


def _ass_hang_fu(hang):
    """一行 Dialogue / Comment 的字段 -> 编辑区上方那排要显示的东西

    层 / 左边距 / 右边距 / 垂直边距 / 特效 / 注释（True = 这行是 Comment，
    存回 ASS 还是 Comment，也不画在画面上）。
    """
    def _zheng(jian):
        pi = re.search(r"(-?\d+)", str((hang or {}).get(jian) or ""))
        return int(pi.group(1)) if pi else 0

    return {
        "ceng": _zheng("Layer"),
        "zuo": _zheng("MarginL"),
        "you": _zheng("MarginR"),
        "shu": _zheng("MarginV"),
        "texiao": str((hang or {}).get("Effect") or "").strip(),
        "zhushi": str((hang or {}).get("_lei") or "") == "Comment",
    }


# 行内标签的"抠掉"用正则在开头那对花括号里找：
#   b / i / u / s：后面不能跟字母，免得把 \bord \shad \fsp 这些认成 B / S
_BIAOQIAN_ZHENG = {
    "b": r"\\b(?![a-zA-Z])[^\\]*",
    "i": r"\\i(?![a-zA-Z])[^\\]*",
    "u": r"\\u(?![a-zA-Z])[^\\]*",
    "s": r"\\s(?![a-zA-Z])[^\\]*",
    "fn": r"\\fn[^\\]*",
    "c1": r"\\1?c(?![a-zA-Z0-9])[^\\]*",
    "c2": r"\\2c[^\\]*",
    "c3": r"\\3c[^\\]*",
    "c4": r"\\4c[^\\]*",
}


def _jia_biaoqian(wen, biao):
    """往开头那对花括号里加一个行内标签（没有就新建一对）"""
    wen = str(wen or "")
    pi = re.match(r"^\{([^{}]*)\}", wen)
    if pi:
        return "{" + pi.group(1) + biao + "}" + wen[pi.end():]
    return "{" + biao + "}" + wen


def _wei_yiduan(wen, xuan, qian, hou):
    """在选中的那一段两头插标签：头上插 qian，尾巴插 hou

    AEG 就是这么干的：编辑框里选中「到底是哪一步」，点一下颜色，就只在那几个
    字前后加标签，不动整条。
    """
    wen = str(wen or "")
    n = len(wen)
    qi = max(0, min(n, int(xuan[0])))
    zhi = max(0, min(n, int(xuan[1])))
    if zhi < qi:
        qi, zhi = zhi, qi
    return wen[:qi] + str(qian) + wen[qi:zhi] + str(hou) + wen[zhi:]


def _qu_biaoqian(wen, jian):
    """把开头那对花括号里的某一类标签抠掉（\\b1、\\3c&H..& 这些）"""
    wen = str(wen or "")
    pi = re.match(r"^\{([^{}]*)\}", wen)
    if not pi:
        return wen
    zheng = _BIAOQIAN_ZHENG.get(str(jian or "").lower())
    if not zheng:
        return wen
    nei = re.sub(zheng, "", pi.group(1))
    hou = wen[pi.end():]
    if not nei.strip():
        return hou
    return "{" + nei + "}" + hou


def _ass_yuan_wen(wenben):
    """ASS 的 Text 归一，但 {\\pos(75,705)} 这些标签原样留着（比"改没改过"用）"""
    return str(wenben or "").replace("\\N", "\n").replace("\\n", "\n").strip()


# ---------------------------------------------------------------------
# 画面里叠字幕（照 ASS 的样式画）
# ---------------------------------------------------------------------
# 认不出来的标签直接扔掉，不影响文字：
#   \t 渐变动画 / \clip \iclip 裁剪 / \p 图形绘制 / \k \K \kf \ko 卡拉OK
_ASS_HUA_KUO = re.compile(r"\{[^}]*\}")

_ASS_MOREN_YANGSHI = {
    "font": "",                                     # 空 = 用界面默认字体
    "fs": HUA_ZIMU_ZIHAO,
    "c1": "#FFFFFF", "c2": "#FF0000", "c3": "#000000", "c4": "#000000",
    "a1": 255, "a2": 255, "a3": 255, "a4": 140,
    "b": False, "i": False, "u": False, "s": False,
    "fscx": 100.0, "fscy": 100.0, "fsp": 0.0, "frz": 0.0,
    "bord": 2.0, "shad": 1.0, "bs": 1, "an": 2,
    "fad": None,                                    # \fad(入ms, 出ms)：渐入渐出
    "ml": HUA_ZIMU_BEI_JIAO, "mr": HUA_ZIMU_BEI_JIAO,
    "mv": HUA_ZIMU_BEI_JIAO,
}


def _ass_yanse(wen, moren=None):
    """ASS 颜色 &HAABBGGRR / &HBBGGRR& -> ("#RRGGBB", 不透明度 0-255)

    ASS 是 BGR 倒着写、而且写的是"透明多少"：&H80FFFFFF 的 80 是半透明。
    认不出来给 moren。
    """
    s = str(wen or "").strip().strip("&").lower()
    if s.startswith("h"):
        s = s[1:]
    if not s or any(ch not in "0123456789abcdef" for ch in s):
        return moren
    try:
        v = int(s, 16)
    except ValueError:
        return moren
    r = v & 0xFF
    g = (v >> 8) & 0xFF
    b = (v >> 16) & 0xFF
    a = 255 if len(s) <= 6 else 255 - ((v >> 24) & 0xFF)
    return ("#%02X%02X%02X" % (r, g, b), max(0, min(255, a)))


def _ass_shuzi(wen, moren=0.0):
    try:
        return float(str(wen).strip())
    except (TypeError, ValueError):
        return moren


def _ass_touming(zhi, moren=None):
    """\\alpha / \\1a 里的值 -> 不透明度 0-255（它写的是"透明多少"，反着来）"""
    s = str(zhi or "").strip().strip("&").lower()
    if s.startswith("h"):
        s = s[1:]
    if not s or any(ch not in "0123456789abcdef" for ch in s):
        return moren
    try:
        return max(0, min(255, 255 - int(s, 16)))
    except ValueError:
        return moren


def _ass_geshi(hang):
    """ASS 样式表里的一行 -> 画字幕用的格式字典"""
    g = dict(_ASS_MOREN_YANGSHI)
    zi = str(hang.get("Fontname") or "").strip()
    if zi:
        g["font"] = zi
    g["fs"] = max(1.0, _ass_shuzi(hang.get("Fontsize"), g["fs"]))
    for jian, lie in (("c1", "PrimaryColour"), ("c2", "SecondaryColour"),
                      ("c3", "OutlineColour"), ("c4", "BackColour")):
        y = _ass_yanse(hang.get(lie))
        if y:
            g[jian], g["a" + jian[1]] = y
    g["b"] = _ass_shuzi(hang.get("Bold"), 0) != 0
    g["i"] = _ass_shuzi(hang.get("Italic"), 0) != 0
    g["u"] = _ass_shuzi(hang.get("Underline"), 0) != 0
    g["s"] = _ass_shuzi(hang.get("StrikeOut"), 0) != 0
    g["fscx"] = _ass_shuzi(hang.get("ScaleX"), 100) or 100
    g["fscy"] = _ass_shuzi(hang.get("ScaleY"), 100) or 100
    g["fsp"] = _ass_shuzi(hang.get("Spacing"), 0)
    g["frz"] = _ass_shuzi(hang.get("Angle"), 0)
    g["bs"] = int(_ass_shuzi(hang.get("BorderStyle"), 1) or 1)
    g["bord"] = max(0.0, _ass_shuzi(hang.get("Outline"), g["bord"]))
    g["shad"] = max(0.0, _ass_shuzi(hang.get("Shadow"), g["shad"]))
    an = int(_ass_shuzi(hang.get("Alignment"), g["an"]) or g["an"])
    g["an"] = an if 1 <= an <= 9 else g["an"]
    for jian, lie in (("ml", "MarginL"), ("mr", "MarginR"),
                      ("mv", "MarginV")):
        g[jian] = _ass_shuzi(hang.get(lie), g[jian])
    return g


def _du_ass_yangshi(tou_hang):
    """从 ASS 骨架里读基准分辨率和样式表

    返回 ((PlayResX, PlayResY), {样式名小写: 格式字典})；
    没写基准分辨率就给 None（用默认的）。
    """
    play = None
    biao = {}
    zai_yangshi = False
    ming = None
    for yuan in tou_hang or []:
        tiao = str(yuan).strip()
        if not tiao:
            continue
        if tiao.startswith("[") and tiao.endswith("]"):
            zai_yangshi = tiao.lower() in ("[v4+ styles]", "[v4 styles]")
            ming = None
            continue
        if tiao.lower().startswith("playresx"):
            play = (_ass_zheng(tiao), play[1] if play else None)
            continue
        if tiao.lower().startswith("playresy"):
            play = (play[0] if play else None, _ass_zheng(tiao))
            continue
        if not zai_yangshi:
            continue
        if tiao.lower().startswith("format:"):
            ming = [x.strip() for x in tiao.split(":", 1)[1].split(",")]
            continue
        if tiao.lower().startswith("style:") and ming:
            bu = tiao.split(":", 1)[1].split(",", len(ming) - 1)
            bu += [""] * (len(ming) - len(bu))
            hang = dict(zip(ming, bu))
            biao[str(hang.get("Name") or "").strip().lower()] = _ass_geshi(hang)
    if play and (not play[0] or not play[1]):
        play = None
    return play, biao


# 样式表那一行的列名 -> 格式字典里的名字（样式编辑器写回时按这个对）
_YANGSHI_LIE_MING = {
    "Fontname": "font",
    "Fontsize": "fs",
    "Bold": "b",
    "Italic": "i",
    "Underline": "u",
    "StrikeOut": "s",
    "ScaleX": "fscx",
    "ScaleY": "fscy",
    "Spacing": "fsp",
    "Angle": "frz",
    "BorderStyle": "bs",
    "Outline": "bord",
    "Shadow": "shad",
    "Alignment": "an",
    "MarginL": "ml",
    "MarginR": "mr",
    "MarginV": "mv",
}

# 四个颜色列 -> (格式字典里的颜色名, 不透明度名)
_YANGSHI_LIE_YANSE = {
    "PrimaryColour": ("c1", "a1"),
    "SecondaryColour": ("c2", "a2"),
    "OutlineColour": ("c3", "a3"),
    "BackColour": ("c4", "a4"),
}


def _shu_wen(v):
    """数字写回 ASS 什么样：整数就不带小数点（40.0 -> 40，40.5 -> 40.5）"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "0"
    if abs(v - round(v)) < 0.001:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _hui_yangshi_tou(tou, yang_ming, zi):
    """把样式编辑器改出来的字段写回骨架里 [V4+ Styles] 那一行

    只动这一个样式那一行：按 Format 那行的列名一个个对，其余列原样留着。
    找不到这一段 / 没这个样式就返回 False（一个字都不改）。
    """
    yang_ming = str(yang_ming or "").strip()
    if not yang_ming:
        return False
    zai = False
    ming = None
    zhao = -1
    for i, yuan in enumerate(tou):
        tiao = str(yuan).strip()
        if tiao.startswith("[") and tiao.endswith("]"):
            zai = tiao.lower() in ("[v4+ styles]", "[v4 styles]")
            ming = None
            continue
        if not zai:
            continue
        if tiao.lower().startswith("format:"):
            ming = [x.strip() for x in tiao.split(":", 1)[1].split(",")]
            continue
        if tiao.lower().startswith("style:") and ming:
            bu = tiao.split(":", 1)[1].split(",", len(ming) - 1)
            if str(bu[0] if bu else "").strip().lower() == yang_ming.lower():
                zhao = i
                break
    if zhao < 0 or not ming:
        return False

    bu = str(tou[zhao]).split(":", 1)[1].split(",", len(ming) - 1)
    bu += [""] * (len(ming) - len(bu))
    for k, lie in enumerate(ming):
        if lie in _YANGSHI_LIE_YANSE:
            wei, tou_jian = _YANGSHI_LIE_YANSE[lie]
            se = str(zi.get(wei) or "").strip().lstrip("#")
            if len(se) != 6:
                continue
            try:
                r = int(se[0:2], 16)
                g = int(se[2:4], 16)
                b = int(se[4:6], 16)
            except ValueError:
                continue
            try:
                a = max(0, min(255, int(zi.get(tou_jian) or 255)))
            except (TypeError, ValueError):
                a = 255
            bu[k] = f"&H{255 - a:02X}{b:02X}{g:02X}{r:02X}"
            continue
        jian = _YANGSHI_LIE_MING.get(lie)
        if not jian or jian not in zi:
            continue
        if jian == "font":
            bu[k] = str(zi.get("font") or "").strip() or "Arial"
        elif jian in ("b", "i", "u", "s"):
            bu[k] = "-1" if zi.get(jian) else "0"
        elif jian in ("ml", "mr", "mv"):
            bu[k] = f"{max(0, int(_ass_shuzi(zi.get(jian), 0))):04d}"
        elif jian in ("bs", "an"):
            bu[k] = str(int(_ass_shuzi(zi.get(jian), 1)))
        else:
            bu[k] = _shu_wen(zi.get(jian))
    if bu:
        bu[0] = str(bu[0]).strip()      # 样式名两边的空格不带出去
    tou[zhao] = "Style: " + ",".join(str(x) for x in bu)
    return True


def _ass_yangshi_ming(tou_hang):
    """骨架里 [V4+ Styles] 那些样式名，按原样（大小写别动，要写回去的）"""
    ming = []
    zai_yangshi = False
    for yuan in tou_hang or []:
        tiao = str(yuan).strip()
        if tiao.startswith("[") and tiao.endswith("]"):
            zai_yangshi = tiao.lower() in ("[v4+ styles]", "[v4 styles]")
            continue
        if not zai_yangshi or not tiao.lower().startswith("style:"):
            continue
        ge = tiao.split(":", 1)[1].split(",", 1)
        name = str(ge[0]).strip() if ge else ""
        if name and name not in ming:
            ming.append(name)
    return ming


def _ass_zheng(tiao):
    """从 "PlayResX: 1280" 里抠出那个整数；抠不出来给 None"""
    pi = re.search(r"(-?\d+)", str(tiao))
    return int(pi.group(1)) if pi else None


def _qie_biaoqian(biao):
    """把 {\\pos(1,2)\\fs30} 里那串切成一个个标签

    括号里还有反斜杠的（\\t(0,500,\\fs30)）不当成分隔，整块留着。
    """
    jieguo = []
    dangqian = ""
    shen = 0
    for ch in biao:
        if ch == "(":
            shen += 1
        elif ch == ")":
            shen = max(0, shen - 1)
        if ch == "\\" and shen == 0:
            if dangqian.strip():
                jieguo.append(dangqian)
            dangqian = ""
            continue
        dangqian += ch
    if dangqian.strip():
        jieguo.append(dangqian)
    return jieguo


def _jia_wenben(hang, wen, geshi):
    """普通文字接到当前行上；\\N \\n 换行，\\h 当空格

    每段文字都记下"这一刻的格式"（复制一份），这样 {\n1c&H..&}某几个字{\n r}
    这种只改一段的写法，那几个字和后面的字各画各的颜色（照 libass）。
    """
    if not wen:
        return
    gs = dict(geshi or {})
    bu = re.split(r"\\[Nn]", wen)
    zhi = bu[0].replace("\\h", " ")
    if zhi:
        hang[-1].append((zhi, gs))
    for x in bu[1:]:
        x = x.replace("\\h", " ")
        hang.append([(x, dict(gs))] if x else [])


# 老 SSA 的 \a 对齐写法 -> 现在的 \an
_ASS_JIU_DUIQI = {1: 1, 2: 2, 3: 3, 5: 7, 6: 8, 7: 9, 9: 4, 10: 5, 11: 6}


def _yi_ge_biaoqian(pian, g, jichu, yangshi_biao):
    """一个行内标签（\\pos(75,705) / \\an5 / \\fs30 …）改到格式上"""
    # \r 后面紧跟样式名、中间没有空格（\rSign），单独认，不能被当成长标签名
    if pian[:1] in ("r", "R"):
        jiu = dict(jichu)
        zhao = (yangshi_biao or {}).get(pian[1:].strip().lower())
        if zhao:
            jiu = dict(zhao)
        # \pos \an \fad 管的是"这一行摆哪儿 / 什么时候出现"，\r 只该把字体颜色
        # 这些还原成样式，不该把定位也一起抹了（libass 就是这么处理的）。不然
        # 在行中间写 {\r} 截断颜色，后面的字会跑到默认位置上去。
        for k in ("pos", "an", "fad"):
            if g.get(k) is not None:
                jiu[k] = g[k]
        g.clear()
        g.update(jiu)
        return
    pi = re.match(r"^\s*(\d?[a-zA-Z]+)\s*(.*)$", pian)
    if not pi:
        return
    m = pi.group(1).lower()
    zhi = pi.group(2).strip()
    if m == "pos" or m == "move":
        shu = re.findall(r"-?\d+(?:\.\d+)?", zhi)
        if len(shu) >= 2:
            g["pos"] = (float(shu[0]), float(shu[1]))
    elif m == "an":
        shu = re.search(r"\d+", zhi)
        if shu:
            an = int(shu.group(0))
            if 1 <= an <= 9:
                g["an"] = an
    elif m == "a":
        # 老 SSA 的对齐写法
        shu = re.search(r"\d+", zhi)
        if shu:
            an = _ASS_JIU_DUIQI.get(int(shu.group(0)))
            if an:
                g["an"] = an
    elif m == "fs":
        if zhi:
            g["fs"] = max(1.0, _ass_shuzi(zhi, g["fs"]))
    elif m in ("frz", "fr"):
        if zhi:
            g["frz"] = _ass_shuzi(zhi, 0.0)
    elif m == "fscx":
        if zhi:
            g["fscx"] = _ass_shuzi(zhi, 100) or 100
    elif m == "fscy":
        if zhi:
            g["fscy"] = _ass_shuzi(zhi, 100) or 100
    elif m == "fsp":
        g["fsp"] = _ass_shuzi(zhi, 0.0)
    elif m in ("fn", "fontname"):
        if zhi:
            g["font"] = zhi
    elif m in ("c", "1c"):
        y = _ass_yanse(zhi)
        if y:
            g["c1"], g["a1"] = y
    elif m == "2c":
        y = _ass_yanse(zhi)
        if y:
            g["c2"], g["a2"] = y
    elif m == "3c":
        y = _ass_yanse(zhi)
        if y:
            g["c3"], g["a3"] = y
    elif m == "4c":
        y = _ass_yanse(zhi)
        if y:
            g["c4"], g["a4"] = y
    elif m == "alpha":
        t = _ass_touming(zhi)
        if t is not None:
            for jian in ("a1", "a3", "a4"):
                g[jian] = t
    elif m in ("1a", "3a", "4a"):
        t = _ass_touming(zhi)
        if t is not None:
            g["a" + m[0]] = t
    elif m == "b":
        g["b"] = _ass_shuzi(zhi, 0) != 0 if zhi else True
    elif m == "i":
        g["i"] = _ass_shuzi(zhi, 0) != 0 if zhi else True
    elif m == "u":
        g["u"] = _ass_shuzi(zhi, 0) != 0 if zhi else True
    elif m == "s":
        g["s"] = _ass_shuzi(zhi, 0) != 0 if zhi else True
    elif m in ("bord", "xbord", "ybord"):
        if zhi:
            g["bord"] = max(0.0, _ass_shuzi(zhi, g["bord"]))
    elif m in ("shad", "xshad", "yshad"):
        if zhi:
            g["shad"] = max(0.0, _ass_shuzi(zhi, g["shad"]))
    elif m == "fad":
        # \fad(渐入ms, 渐出ms)：两个数就是起 / 收各花多少毫秒
        shu = re.findall(r"-?\d+(?:\.\d+)?", zhi)
        if len(shu) >= 2:
            g["fad"] = (
                max(0, int(float(shu[0]))), max(0, int(float(shu[1])))
            )
    elif m == "fade":
        # \fade(a1,a2,a3,t1,t2,t3,t4)：起 = t1，收 = t4 - t3
        shu = re.findall(r"-?\d+(?:\.\d+)?", zhi)
        if len(shu) >= 7:
            g["fad"] = (
                max(0, int(float(shu[3]))),
                max(0, int(float(shu[6]) - float(shu[5]))),
            )


def _jie_ass_tiao(yuan_wen, yangshi, yangshi_biao=None):
    """一行的 ASS 原文 -> (文字行列表, 格式字典)

    文字行列表里每一行是若干段：[(文字, 格式), ...]——一段一段各带各的格式，
    {\n1c&H..&}某几个字{\n r}后面的字 这种只改一段的写法才能画出来。
    返回的格式字典是这一行最后生效的那份（\\an \\pos \\fs 这些行级的用它）。
    """
    g = dict(yangshi)
    jichu = dict(yangshi)
    hang = [[]]
    wen = str(yuan_wen or "")
    wei = 0
    for pi in _ASS_HUA_KUO.finditer(wen):
        _jia_wenben(hang, wen[wei:pi.start()], g)
        gs_kuo = pi.group(0)[1:-1]
        for pian in _qie_biaoqian(gs_kuo):
            _yi_ge_biaoqian(pian, g, jichu, yangshi_biao)
        wei = pi.end()
    _jia_wenben(hang, wen[wei:], g)
    return [x for x in hang if x], g


_ZITI_CHIDU = {}        # (字体名, 走不走粗档) -> 字身倍率（算一次存着，见 _ziti_chidu）
_ZITI_ZI_ZHONG = {}     # 字体名 -> 系统里的字重（问一次存着，见 _ziti_xi_tong_zi_zhong）


class _ZITI_LOGFONTW(ctypes.Structure):
    """Windows 的 LOGFONTW：问系统字体重的时候填名字用"""

    _fields_ = [
        ("lfHeight", ctypes.c_long),
        ("lfWidth", ctypes.c_long),
        ("lfEscapement", ctypes.c_long),
        ("lfOrientation", ctypes.c_long),
        ("lfWeight", ctypes.c_long),
        ("lfItalic", ctypes.c_ubyte),
        ("lfUnderline", ctypes.c_ubyte),
        ("lfStrikeOut", ctypes.c_ubyte),
        ("lfCharSet", ctypes.c_ubyte),
        ("lfOutPrecision", ctypes.c_ubyte),
        ("lfClipPrecision", ctypes.c_ubyte),
        ("lfQuality", ctypes.c_ubyte),
        ("lfPitchAndFamily", ctypes.c_ubyte),
        ("lfFaceName", ctypes.c_wchar * 32),
    ]


class _ZITI_MEIJU(ctypes.Structure):
    """Windows 的 ENUMLOGFONTEXW：枚举字体时回调收到的东西"""

    _fields_ = [
        ("elfLogFont", _ZITI_LOGFONTW),
        ("elfFullName", ctypes.c_wchar * 64),
        ("elfStyle", ctypes.c_wchar * 32),
        ("elfScript", ctypes.c_wchar * 32),
    ]


def _ziti_xi_tong_zi_zhong(ming):
    """问 Windows：这个名字的字体本身是多重（400 常规、700 粗）

    Aegisub 是按"字体名"问系统的，拿到哪支就是哪支。Qt 走的是 DirectWrite，
    Windows 会把 新兰圆-B / -M / -R 这种按内部家族名并成一个家族，常规那一档
    反而是细的字形——所以字看着细、描边比字粗。这里直接问系统要真实字重，
    字重 >= 600 就让 Qt 走粗那一档，跟 Aegisub 一样。
    问不出来（不是 Windows 之类）就按 400 走。
    """
    ming = str(ming or "").strip()
    if not ming:
        return 400
    if ming in _ZITI_ZI_ZHONG:
        return _ZITI_ZI_ZHONG[ming]
    zhong = 400
    try:
        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32
        huifu = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(_ZITI_MEIJU),
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_long,
        )
        gdi32.EnumFontFamiliesExW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ZITI_LOGFONTW),
            huifu,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        gdi32.EnumFontFamiliesExW.restype = ctypes.c_int
        user32.GetDC.argtypes = [ctypes.c_void_p]
        user32.GetDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        zhong_men = []

        def _kan(zhizhen, zi_mei, lei, can):
            zhong_men.append(int(zhizhen.contents.elfLogFont.lfWeight))
            return 1

        huiti = user32.GetDC(None)
        try:
            wen = _ZITI_LOGFONTW()
            wen.lfCharSet = 1                      # DEFAULT_CHARSET
            wen.lfFaceName = ming
            gdi32.EnumFontFamiliesExW(
                huiti, ctypes.byref(wen), huifu(_kan), None, 0
            )
        finally:
            user32.ReleaseDC(None, huiti)
        if zhong_men:
            zhong = min(zhong_men)
    except Exception as cuowu:  # noqa
        logger.error(f"视频工作台：读系统字体字重失败 {cuowu}")
    _ZITI_ZI_ZHONG[ming] = zhong
    return zhong


def _ziti_ming_cu(ming):
    """这个名字的字体本身就粗（字重 >= 600）-> True"""
    return _ziti_xi_tong_zi_zhong(ming) >= 600


_ZITI_MING_MEN = None   # 系统字体名清单（枚举一次存着，见 _ziti_ming_men）


def _ziti_ming_men():
    """系统里所有的字体名（AEG 那个字体下拉列的就是这些）

    Aegisub 走 Windows 的字体枚举，拿到的是"字体名"——整支字体算一个名字
    （新兰圆-B / 新兰圆-M / 新兰圆-R 各算一个），名字前面带 @ 的是竖排那一支。
    Qt 走 DirectWrite，会把它们按内部家族名并成一个，也看不到 @ 的竖排支，
    所以这里直接问 Windows 要；问不出来（不是 Windows）才退回 Qt 的家族表。

    排序：不带 @ 的排前面，各自按名字排（@ 的那些要是排前面就一大堆，不好挑）。
    """
    global _ZITI_MING_MEN
    if _ZITI_MING_MEN is not None:
        return _ZITI_MING_MEN
    ming_men = []
    try:
        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32
        huifu = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(_ZITI_MEIJU),
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_long,
        )
        gdi32.EnumFontFamiliesExW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ZITI_LOGFONTW),
            huifu,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        gdi32.EnumFontFamiliesExW.restype = ctypes.c_int
        user32.GetDC.argtypes = [ctypes.c_void_p]
        user32.GetDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        def _kan(zhizhen, zi_mei, lei, can):
            ming = str(zhizhen.contents.elfLogFont.lfFaceName or "").strip()
            if ming and ming not in ming_men:
                ming_men.append(ming)
            return 1

        huiti = user32.GetDC(None)
        try:
            wen = _ZITI_LOGFONTW()
            wen.lfCharSet = 1                      # DEFAULT_CHARSET
            wen.lfFaceName = ""                    # 空 = 全家都列出来
            gdi32.EnumFontFamiliesExW(
                huiti, ctypes.byref(wen), huifu(_kan), None, 0
            )
        finally:
            user32.ReleaseDC(None, huiti)
    except Exception as cuowu:  # noqa
        logger.error(f"视频工作台：枚举系统字体失败 {cuowu}")
    if not ming_men:
        try:
            ming_men = [str(x) for x in QtGui.QFontDatabase().families()]
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：读 Qt 字体家族失败 {cuowu}")
            ming_men = []
    ming_men.sort(key=lambda m: (m.startswith("@"), m.casefold()))
    _ZITI_MING_MEN = ming_men
    return ming_men


# ---- 竖排字体（字体名带 @ 的那一支，如 @新兰圆-B） --------------------------
#
# Windows 的规矩：字体名前面加 @ 就是那一支"躺倒"的字体（Aegisub 的下拉里
# 列的和 ASS 里存的就是这个名字）。Qt 走 DirectWrite，压根没有 @ 这支（@新宋体
# 会被它当成宋体），所以字体照不带 @ 的名字去要、躺倒我们自己转 —— 转法是抄
# libass 的（ass_font.c 里 ass_get_glyph_outline，它就是照 GDI / VSFilter 抄的）：
#
#     横坐标 = 竖排前进量 + 字身下伸 + 字形原来的纵坐标
#     纵坐标 = -字身下伸 - 字形原来的横坐标
#
# 也就是把字形原地转 90°；排的时候一个字往前推"竖排前进量"（汉字就是 1 个 em），
# 所以看着还是横着排、但每个字都躺倒了 —— 跟 Aegisub 画面里一模一样。
_SHENG_SHU = 0x02f1     # 码点 >= 这个才躺倒（拉丁字母不躺）—— libass 的 VERTICAL_LOWER_BOUND

_SHU_DULIANG = {}       # 字体名 -> (em 方框, 字身下伸, vmtx 表, 竖排度量条数, QRawFont)
_SHU_ADV = {}           # (字体名, 字) -> 竖排前进量（字体单位）


def _shi_shu(ming):
    """字体名是不是竖排那一支（@ 开头）"""
    return str(ming or "").strip().startswith("@")


def _qu_shu(ming):
    """竖排名字去掉 @ 才是系统里那支字体（Qt 只认不带 @ 的）"""
    return str(ming or "").strip().lstrip("@")


def _ziti_xiangsu(ziti):
    """字体实际画出来多少像素高（点数值的字体也能问出来）"""
    try:
        px = float(QtGui.QFontInfo(ziti).pixelSize())
    except Exception:  # noqa
        px = float(ziti.pixelSize())
    return max(1.0, px)


def _shu_duliang(ming):
    """竖排那一支的度量：em 方框大小、字身下伸、每个字的竖排前进量

    都是字体单位（跟字号无关），读一次存着。读不出来（不是 Windows、字体没这
    几张表）就按"一个字一个 em"算 —— 汉字字体本来就是这么设计的。
    """
    ming = _qu_shu(ming)
    if ming in _SHU_DULIANG:
        return _SHU_DULIANG[ming]
    em, xia, vmtx, tiao, yuan = 1000, 0, b"", 0, None
    try:
        ziti = QtGui.QFont()
        ziti.setFamily(ming)
        yuan = QtGui.QRawFont.fromFont(ziti)
        if yuan.isValid():
            tou = bytes(yuan.fontTable("head"))
            if len(tou) >= 20:
                em = struct.unpack_from(">H", tou, 18)[0]
            o2 = bytes(yuan.fontTable("OS/2"))
            if len(o2) >= 72:
                xia = struct.unpack_from(">h", o2, 70)[0]
            vh = bytes(yuan.fontTable("vhea"))
            vmtx = bytes(yuan.fontTable("vmtx"))
            if len(vh) >= 36:
                tiao = struct.unpack_from(">H", vh, 34)[0]
        else:
            yuan = None
    except Exception as cuowu:      # noqa
        logger.error(f"视频工作台：读竖排字体度量失败 {cuowu}")
    if em <= 0:
        em = 1000
    jieguo = (em, xia, vmtx, tiao, yuan)
    _SHU_DULIANG[ming] = jieguo
    return jieguo


def _shu_adv(ming, zi):
    """一个字躺倒后往前推多少（字体单位）；表里没有就按一个 em"""
    jian = (_qu_shu(ming), zi)
    if jian in _SHU_ADV:
        return _SHU_ADV[jian]
    em, _xia, vmtx, tiao, yuan = _shu_duliang(ming)
    adv = em
    try:
        if yuan is not None and tiao > 0:
            zong = yuan.glyphIndexesForString(zi)
            if zong:
                hao = min(int(zong[0]), tiao - 1)
                if (hao + 1) * 4 <= len(vmtx):
                    adv = struct.unpack_from(">H", vmtx, hao * 4)[0]
    except Exception as cuowu:      # noqa
        logger.error(f"视频工作台：读竖排前进量失败 {cuowu}")
    if adv <= 0:
        adv = em
    _SHU_ADV[jian] = adv
    return adv


def _shu_kuan(ziti, ming, wen):
    """竖排文字排出来总共多宽（当前坐标像素）"""
    em = float(_shu_duliang(ming)[0])
    bi = _ziti_xiangsu(ziti) / em
    fm = QtGui.QFontMetricsF(ziti)
    kuan = 0.0
    for zi in str(wen):
        if ord(zi) >= _SHENG_SHU:
            kuan += _shu_adv(ming, zi) * bi
        else:
            kuan += fm.horizontalAdvance(zi)     # 拉丁字母不躺倒，照旧
    return kuan


def _shu_lu(ziti, ming, wen, x, di):
    """竖排文字 -> 路径：每个字躺倒、横向一个个排（算法跟 libass 一样）

    x / di 是这一段的起笔位置和基线（当前坐标）；返回拼好的路径。
    """
    em, xia_zi, _v, _t, _r = _shu_duliang(ming)
    bi = _ziti_xiangsu(ziti) / float(em)
    xia = xia_zi * bi                     # 字身下伸（像素，负的）
    fm = QtGui.QFontMetricsF(ziti)
    lu = QtGui.QPainterPath()
    x_dang = float(x)
    for zi in str(wen):
        if ord(zi) >= _SHENG_SHU:
            adv = _shu_adv(ming, zi) * bi
            yi = QtGui.QPainterPath()
            yi.addText(0.0, 0.0, ziti, zi)
            # 横坐标 = 前进量 + 下伸 + 字形纵坐标；纵坐标 = -下伸 - 字形横坐标
            yi = QtGui.QTransform(
                0.0, -1.0, 1.0, 0.0, adv + xia, -xia
            ).map(yi)
            yi.translate(x_dang, di)
            lu.addPath(yi)
            x_dang += adv
        else:
            lu.addText(x_dang, di, ziti, zi)
            x_dang += fm.horizontalAdvance(zi)
    return lu


def _ziti_kuan_biao(fangda, yi_hang):
    """一段文字 -> [(字, 格式, 宽度), ...]（宽度是当前坐标的像素，竖排按竖排算）"""
    jieguo = []
    for wen, gs in yi_hang:
        ziti = _hua_ziti(gs, fangda)
        fm = QtGui.QFontMetricsF(ziti)
        ming = str(gs.get("font") or "").strip()
        shu = _shi_shu(ming)
        bi = (_ziti_xiangsu(ziti) / float(_shu_duliang(ming)[0])) if shu else 1.0
        for ch in str(wen):
            if shu and ord(ch) >= _SHENG_SHU:
                jieguo.append((ch, gs, float(_shu_adv(ming, ch)) * bi))
            else:
                jieguo.append((ch, gs, float(fm.width(ch))))
    return jieguo


def _ziti_chidu(ming, cu=False):
    """(字体名, 走不走粗档) -> 字身倍率：ASS 里的 Fontsize 要乘这个才是真正的字身大小

    Aegisub（libass）把 Fontsize 当成"上伸 + 下伸"这么高，不是当成一个字身方框，
    所以真正画出来的字身 = Fontsize × 字身 ÷ (上伸 + 下伸)。这里在 1000 号字下量
    一次上伸下伸，把倍率除出来，跟 Aegisub 一模一样。
    粗档和细档的上伸下伸不一样，所以 cu 也得算进键里。
    """
    jian = (_qu_shu(ming), bool(cu))    # 竖排那一支（@xxx）跟 xxx 是同一份字体
    if jian in _ZITI_CHIDU:
        return _ZITI_CHIDU[jian]
    bili = 1.0
    try:
        ziti = QtGui.QFont()
        if jian[0]:
            ziti.setFamily(jian[0])
        ziti.setPixelSize(1000)
        ziti.setBold(bool(cu))
        fm = QtGui.QFontMetricsF(ziti)
        gao = fm.ascent() + fm.descent()
        if gao > 1.0:
            bili = 1000.0 / gao
    except Exception as cuowu:  # noqa
        logger.error(f"视频工作台：算字身倍率失败 {cuowu}")
    _ZITI_CHIDU[jian] = bili
    return bili


def _hua_ziti(gs, bi):
    """格式字典 + 画面缩放比 -> 字体（字号 / 粗斜体 / 字距都按比例算好）"""
    ziti = QtGui.QFont()
    ming = str(gs.get("font") or "").strip()
    if ming:
        # 直接用 ASS 里写的名字，让 Qt 去系统字体库里找（不写死任何路径）。
        # 竖排那一支（@xxx）Qt 没有，得去掉 @ 去要字体，躺倒由 _shu_lu 自己转。
        ziti.setFamily(_qu_shu(ming))
    # 走不走粗档：ASS 里写了 Bold，或者这个名字在系统里本身是粗体（新兰圆-B）
    cu = bool(gs.get("b")) or _ziti_ming_cu(ming)
    px = max(1.0, _ass_shuzi(gs.get("fs"), 1.0) * bi
             * (_ass_shuzi(gs.get("fscy"), 100) / 100.0)
             * _ziti_chidu(ming, cu))
    ziti.setPixelSize(max(1, int(round(px))))
    ziti.setBold(cu)
    ziti.setItalic(bool(gs.get("i")))
    ziti.setUnderline(bool(gs.get("u")))
    ziti.setStrikeOut(bool(gs.get("s")))
    if abs(_ass_shuzi(gs.get("fsp"), 0.0)) > 0.01:
        ziti.setLetterSpacing(
            QtGui.QFont.AbsoluteSpacing, _ass_shuzi(gs["fsp"], 0.0) * bi
        )
    return ziti


def _zhe_hang(fm, wen, kuan):
    """一行太长就按宽度折成几行（ASS 默认也会折）"""
    if kuan <= 0 or fm.width(wen) <= kuan:
        return [wen]
    jieguo = []
    dangqian = ""
    for ch in wen:
        if dangqian and fm.width(dangqian + ch) > kuan:
            jieguo.append(dangqian)
            dangqian = ch
        else:
            dangqian += ch
    if dangqian:
        jieguo.append(dangqian)
    return jieguo or [wen]


def _zhe_duan(fangda, yi_hang, kuan):
    """分段折行：[(文字, 格式), ...] -> [[(文字, 格式), ...], ...]

    一行里的字可能一段一个格式（{\\1c&H..&}某几个字{\\r}后面的字），所以宽度
    得一段一段量、折行只能折在字与字之间；折完把挨着同格式的字并回一段。
    """
    zifu = _ziti_kuan_biao(fangda, yi_hang)
    xing = [[]]
    kuan_hang = 0.0
    for ch, gs, w in zifu:
        if xing[-1] and kuan_hang + w > kuan:
            xing.append([])
            kuan_hang = 0.0
        xing[-1].append((ch, gs))
        kuan_hang += w
    jieguo = []
    for hang_zf in xing:
        duan = []
        for ch, gs in hang_zf:
            if duan and duan[-1][1] == gs:
                duan[-1][0] += ch
            else:
                duan.append([ch, gs])
        if duan:
            jieguo.append([(a, b) for a, b in duan])
    return jieguo or [[]]


def _gs_yanse(gs, jian, a_jian, bei=1.0):
    y = QtGui.QColor(str(gs.get(jian) or "#FFFFFF"))
    tou = _ass_shuzi(gs.get(a_jian), 255) * max(0.0, min(1.0, bei))
    y.setAlpha(max(0, min(255, int(round(tou)))))
    return y


def _hua_yi_hang(huabi, lu, gs, bi, fscx, zhong_x, di_y, bei=1.0):
    """画一行：先阴影、再描边、最后填字（跟 ASS 的层次一样）

    bei 是这一刻的不透明度倍数（\\fad 渐入渐出用），1 = 原样。
    """
    huabi.save()
    if abs(fscx - 1.0) > 0.001:
        # \fscx：横着拉伸，锚点（中点）不动
        huabi.translate(zhong_x, 0.0)
        huabi.scale(fscx, 1.0)
        huabi.translate(-zhong_x, 0.0)
    if abs(_ass_shuzi(gs.get("frz"), 0.0)) > 0.01:
        huabi.translate(zhong_x, di_y)
        huabi.rotate(_ass_shuzi(gs["frz"], 0.0))
        huabi.translate(-zhong_x, -di_y)
    shad = max(0.0, _ass_shuzi(gs.get("shad"), 0.0)) * bi
    if shad > 0.05:
        yin = QtGui.QPainterPath(lu)
        yin.translate(shad, shad)
        huabi.fillPath(yin, _gs_yanse(gs, "c4", "a4", bei))
    bord = max(0.0, _ass_shuzi(gs.get("bord"), 0.0)) * bi
    if bord > 0.05:
        bi_bi = QtGui.QPen(_gs_yanse(gs, "c3", "a3", bei))
        bi_bi.setWidthF(bord * 2.0)
        bi_bi.setJoinStyle(Qt.RoundJoin)
        bi_bi.setCapStyle(Qt.RoundCap)
        huabi.strokePath(lu, bi_bi)
    huabi.fillPath(lu, _gs_yanse(gs, "c1", "a1", bei))
    huabi.restore()


def _fad_bei(tiao):
    """\\fad(入ms, 出ms) -> 播到这一刻该压多少不透明度（0~1；没写就给 1）"""
    fad = (tiao.get("gs") or {}).get("fad")
    if not fad:
        return 1.0
    qi = int(tiao.get("qi") or 0)
    zhi = int(tiao.get("zhi") or 0)
    ms = int(tiao.get("ms") or 0)
    jin, chu = int(fad[0]), int(fad[1])
    bei = 1.0
    if jin > 0 and ms < qi + jin:
        bei = min(bei, (ms - qi) / float(jin))
    if chu > 0 and ms > zhi - chu:
        bei = min(bei, (zhi - ms) / float(chu))
    return max(0.0, min(1.0, bei))


def _hua_ass_tiao(huabi, qu, tiao, jizhun):
    """把一条字幕画进画面矩形 qu 里

    全程按 ASS 自己的坐标系（jizhun = PlayResX/PlayResY）算：字号就是样式/标签里
    写的那个数（一个字多少像素就是多少），位置就是 \\pos / 边距里写的那个数，行高
    按字体的上伸下伸算，最后整块等比缩到画面上，出了画面的切掉 —— 跟 Aegisub 一致。
    """
    if qu.width() <= 8 or qu.height() <= 8:
        return
    gs = tiao.get("gs") or dict(_ASS_MOREN_YANGSHI)
    hang = [x for x in (tiao.get("hang") or []) if x]
    if not hang:
        return
    bei = _fad_bei(tiao)
    if bei <= 0.004:
        return          # 渐入渐出到全透明了，这一帧干脆不画
    kuan_jz = float(max(1, int(jizhun[0])))      # 基准宽（PlayResX）
    gao_jz = float(max(1, int(jizhun[1])))       # 基准高（PlayResY）
    bi_x = qu.width() / kuan_jz                  # 基准 -> 画面，横竖各一个比例
    bi_y = qu.height() / gao_jz
    # 字号：按 ASS 里写的大小算（40 就是 40 个基准像素的"上伸+下伸"高，
    # 真正的字身再乘字体的倍率），跟 Aegisub 一致。
    # Qt 的字号只能填整数，直接按 32 号画会比 Aegisub 宽出 1% 还多（长行还可能
    # 凭空多折一行），所以先把字放大 FANGDA 倍画、再整条缩回来，宽度就准了。
    fangda = float(HUA_ZIMU_FANGDA)
    ziti = _hua_ziti(gs, fangda)
    fm = QtGui.QFontMetricsF(ziti)
    fscx = max(0.01, _ass_shuzi(gs.get("fscx"), 100) / 100.0)
    ml = max(0.0, _ass_shuzi(gs.get("ml"), 0.0))
    mr = max(0.0, _ass_shuzi(gs.get("mr"), 0.0))
    mv = max(0.0, _ass_shuzi(gs.get("mv"), 0.0))
    # 折行宽度：写了 \pos 的行按"整个画面宽"折——Aegisub（libass）就是这样，
    # 长出来的部分直接溢到画面外，不会缩回边距里折成两行；没写 \pos 的才扣
    # 掉左右边距再折。
    if gs.get("pos"):
        kuan = max(10.0, kuan_jz / fscx)
    else:
        kuan = max(10.0, (kuan_jz - ml - mr) / fscx)
    xing = []
    for x in hang:
        xing.extend(_zhe_duan(fangda, x, kuan * fangda))
    # 行高 = 上伸 + 下伸（不算 Qt 的 leading），跟 ASS 的行距一致
    lh = max(1.0, (fm.ascent() + fm.descent()) / fangda)
    zong_gao = lh * len(xing)
    an = int(_ass_shuzi(gs.get("an"), 2))
    an = an if 1 <= an <= 9 else 2
    lie = (an - 1) % 3               # 0 左 / 1 中 / 2 右
    pai = (an - 1) // 3              # 0 下 / 1 中 / 2 上
    pos = gs.get("pos")
    if pos:
        ax = float(pos[0])
        ay = float(pos[1])
    else:
        if lie == 0:
            ax = ml
        elif lie == 1:
            # 居中：在左右边距之间居中，左/右边距各推一半 —— libass / VSFilter
            # 就是这么算的（Aegisub 画面一模一样），不是死钉在画面正中。
            ax = (kuan_jz + ml - mr) / 2.0
        else:
            ax = kuan_jz - mr
        if pai == 0:
            ay = gao_jz - mv
        elif pai == 1:
            ay = gao_jz / 2.0
        else:
            ay = mv
    if pai == 0:
        ding = ay - zong_gao
    elif pai == 1:
        ding = ay - zong_gao / 2.0
    else:
        ding = ay
    huabi.save()
    huabi.setRenderHint(QtGui.QPainter.Antialiasing, True)
    # 基准坐标 -> 画面：挪到画面左上角、按两个方向的比例缩过去；画面外的切掉
    huabi.translate(qu.left(), qu.top())
    huabi.scale(bi_x, bi_y)
    huabi.setClipRect(QtCore.QRectF(0.0, 0.0, kuan_jz, gao_jz))
    if int(_ass_shuzi(gs.get("bs"), 1)) == 3:
        # 实底框：左右边距之间铺一条（ASS 的"不透明框"）
        kuang = QtCore.QRectF(
            ml, ding, max(1.0, kuan_jz - ml - mr), zong_gao,
        )
        huabi.fillRect(kuang, _gs_yanse(gs, "c3", "a3"))
    for i, duan in enumerate(xing):
        # 行宽 = 各段宽度之和；对齐照这一行的总宽算
        kuan_duan = []
        kuan_x = 0.0
        for wen_d, gs_d in duan:
            kuan_duan.append(
                sum(w for _c, _g, w in _ziti_kuan_biao(fangda, [(wen_d, gs_d)]))
                / fangda * fscx
            )
            kuan_x += kuan_duan[-1]
        if lie == 0:
            x0 = ax
        elif lie == 1:
            x0 = ax - kuan_x / 2.0
        else:
            x0 = ax - kuan_x
        di = ding + i * lh + fm.ascent() / fangda
        # 一段一段画：每段用自己那份格式（颜色 / 粗斜体 / 字号都能只有几个字不一样）
        x_dang = x0
        zhong_x = x0 + kuan_x / 2.0
        for (wen_d, gs_d), kuan_d in zip(duan, kuan_duan):
            ziti_d = _hua_ziti(gs_d, fangda)
            ming_d = str(gs_d.get("font") or "").strip()
            if _shi_shu(ming_d):
                # 竖排那一支：字一个躺倒横着排（照 libass 的算法）
                lu_da = _shu_lu(
                    ziti_d, ming_d, wen_d, x_dang * fangda, di * fangda
                )
            else:
                lu_da = QtGui.QPainterPath()
                lu_da.addText(x_dang * fangda, di * fangda, ziti_d, wen_d)
            lu = QtGui.QTransform().scale(1.0 / fangda, 1.0 / fangda).map(lu_da)
            _hua_yi_hang(huabi, lu, gs_d, 1.0, fscx, zhong_x, di, bei)
            x_dang += kuan_d
    huabi.restore()


def _zimu_qian(tiao_men, mubiao, jizhun):
    """给画面上这一屏字幕算个指纹：指纹一样就不用重画

    影响画面的全在这儿：折好的行、各段格式、这一刻的透明度、画面多大、
    ASS 的基准分辨率。毫秒不进指纹 —— 毫秒只用来算透明度，剔掉之后正常
    播放（没写 \\fad 的那种）每一帧的指纹都跟上一帧一模一样，图直接复用。
    """
    bu = []
    for tiao in tiao_men:
        bu.append(
            (
                repr(tiao.get("hang") or []),
                repr(tiao.get("gs") or {}),
                round(float(_fad_bei(tiao)), 3),
            )
        )
    return (
        tuple(bu),
        int(mubiao.width()),
        int(mubiao.height()),
        tuple(jizhun or ()),
    )


def _chai_ass(wenben):
    """把 ASS 拆开留着"原样"：骨架原文 + 每行 Dialogue 的完整字段

    {"tou": [Events 之前的所有原文行], "events": Events 段头原文,
     "geshi": Format 行原文, "ming": 字段名, "zai": Events 段里的其它行,
     "hang": [每行的 {字段名: 值}, ...]}。没有 Events 段就返回 None。
    """
    tou, zai, hang, ming = [], [], [], list(_ASS_ZIDUAN)
    events, geshi = "", ""
    zai_events = False
    you_events = False
    for yuan_hang in (wenben or "").splitlines():
        tiao = yuan_hang.strip()
        if tiao.startswith("[") and tiao.endswith("]"):
            zai_events = tiao.lower() == "[events]"
            if zai_events:
                you_events = True
                events = yuan_hang
            else:
                tou.append(yuan_hang)
            continue
        if not zai_events:
            tou.append(yuan_hang)
            continue
        if tiao.lower().startswith("format:"):
            geshi = yuan_hang
            lie = [x.strip() for x in tiao.split(":", 1)[1].split(",")]
            if lie:
                ming = lie
            continue
        if tiao.lower().startswith(("dialogue:", "comment:")):
            # Text 是最后一个字段、里面可以有逗号：只切前面那几个
            bu = tiao.split(":", 1)[1].strip().split(",", len(ming) - 1)
            bu += [""] * (len(ming) - len(bu))
            hang.append(
                dict(
                    zip(ming, bu),
                    _lei=(
                        "Comment"
                        if tiao.lower().startswith("comment:")
                        else "Dialogue"
                    ),
                )
            )
            continue
        zai.append(yuan_hang)
    if not you_events:
        return None
    return {
        "tou": tou,
        "events": events,
        "geshi": geshi,
        "ming": ming,
        "zai": zai,
        "hang": hang,
    }


def _xie_ass_baoliu(lu, jiegou, zimu, fu_liebiao=None):
    """按原 ASS 的结构写出去：骨架 / 样式表 / 每行的字段原样带过去

    只有时间、条数、文字会变：每条字幕优先认原来的那一行（先认文字、再认原来
    的时间），认到就把那行的 Style / Name / 边距 / Effect 一起带上；认不到
    （新加的行）就照原来第一条 Dialogue 的样子复制一份。

    fu_liebiao = 每条的字段表（样式 / 说话人 / 层 / 边距 / 特效 / 注释），给了就
    盖在认出来的那一行上。
    """
    from anylabeling.views.labeling.utils.video import (
        ASS_TEMPLATE,
        format_ass_time,
    )

    ming = jiegou.get("ming") or list(_ASS_ZIDUAN)
    lao = [dict(x) for x in (jiegou.get("hang") or [])]
    fu_men = list(fu_liebiao or [])
    yong = [False] * len(lao)
    muban = dict(lao[0]) if lao else {k: "" for k in ming}
    muban.setdefault("Style", "Default")

    hang_xin = []
    for xu, (qi, zhi, wenben) in enumerate(zimu):
        wen = str(wenben or "")
        chun = _ass_chun_wen(wen)
        zhao = -1
        for i, h in enumerate(lao):                     # 先按文字认
            if not yong[i] and _ass_chun_wen(h.get("Text")) == chun:
                zhao = i
                break
        if zhao < 0:                                    # 文字改过就按原时间认
            for i, h in enumerate(lao):
                if (
                    not yong[i]
                    and _ass_hao_ms(h.get("Start")) == int(qi)
                    and _ass_hao_ms(h.get("End")) == int(zhi)
                ):
                    zhao = i
                    break
        if zhao >= 0:
            yong[zhao] = True
            bu = dict(lao[zhao])
            # 文字没动过就沿用原来那一串：内联标签（\pos、\k 之类）原样留着，
            # 真改了字（或者改了标签）才换成新的那一串
            if _ass_yuan_wen(bu.get("Text")) != _ass_yuan_wen(wen):
                bu["Text"] = wen.replace("\n", "\\N")
        else:
            bu = dict(muban)
            bu["Text"] = wen.replace("\n", "\\N")
        bu["Start"] = format_ass_time(int(qi))
        bu["End"] = format_ass_time(int(zhi))
        # 编辑区上方那排改过的字段：盖在原行上（没给就照原样）
        fu = fu_men[xu] if xu < len(fu_men) else {}
        if fu:
            if fu.get("yang"):
                bu["Style"] = str(fu["yang"])
            if fu.get("shuo") is not None and "shuo" in fu:
                bu["Name"] = str(fu["shuo"] or "")
            if fu.get("ceng") is not None and "ceng" in fu:
                bu["Layer"] = str(int(fu.get("ceng") or 0))
            for jian, lie in (("zuo", "MarginL"), ("you", "MarginR"),
                              ("shu", "MarginV")):
                if jian in fu:
                    bu[lie] = f"{max(0, int(fu.get(jian) or 0)):04d}"
            if "texiao" in fu:
                bu["Effect"] = str(fu.get("texiao") or "")
            if "zhushi" in fu:
                bu["_lei"] = "Comment" if fu.get("zhushi") else "Dialogue"
        hang_xin.append(bu)

    lei_zhi = str(hang_xin[0].get("_lei") or "Dialogue") if hang_xin else ""
    with open(lu, "w", encoding="utf-8-sig") as f:
        tou = jiegou.get("tou") or []
        if tou:
            for tiao in tou:
                f.write(tiao + "\n")
        else:
            f.write(ASS_TEMPLATE)
        f.write((jiegou.get("events") or "[Events]") + "\n")
        if jiegou.get("geshi"):
            f.write(jiegou["geshi"] + "\n")
        for tiao in jiegou.get("zai") or []:
            f.write(tiao + "\n")
        for bu in hang_xin:
            f.write(
                str(bu.get("_lei") or lei_zhi or "Dialogue")
                + ": "
                + ",".join(str(bu.get(k, "")) for k in ming)
                + "\n"
            )


def _xie_zimu_dao_wenjian(lu, zimu, ass_yuan=None, fu_liebiao=None):
    """按扩展名挑写法：.ass / .ssa 有原结构就按原结构写，否则用模板"""
    if ass_yuan and osp.splitext(lu)[1].lower() in (".ass", ".ssa"):
        _xie_ass_baoliu(lu, ass_yuan, zimu, fu_liebiao)
    else:
        _xie_zimu_wenjian(lu, zimu)


# ---------------------------------------------------------------------
# 时间轴
# ---------------------------------------------------------------------
class ShijianZhou(QtWidgets.QWidget):
    """刻度尺 + 音频波形 + 播放头；点击/拖动定位；可缩放、可左右平移"""

    tiaozheng = pyqtSignal(int)     # 请求定位到 ms
    tuo_kaishi = pyqtSignal()       # 鼠标按在刻度尺上、准备拖播放头了
    tuo_jieshu = pyqtSignal()       # 手松了（拖播放头结束）
    shitu_bian = pyqtSignal()       # 视图（缩放/平移）变了 -> 外面刷新滚动条
    xuan_zhong = pyqtSignal(int)    # 点了字幕块 -> 第几条（跟右侧列表对齐）
    zimu_tuo = pyqtSignal(int, int, int)    # 拖完字幕块 -> 第几条, 新起ms, 新止ms
    shanchu_zimu = pyqtSignal(object)       # 请求删掉这几条（右键菜单 / Delete）
    hebing_zimu = pyqtSignal(object)        # 请求把这几条并成一条
    fangxiang_jian = pyqtSignal(int)        # 焦点在时间轴上按左右方向键 -> -1 后退 / +1 前进一帧
    zimu_nuo = pyqtSignal(object)           # 方向键挪了选中的块 -> [(第几条, 新起ms, 新止ms), ...]
    xiayitiao = pyqtSignal()                # 焦点在时间轴上按 Ins -> 换到下一条字幕
    fuzhi_kuai = pyqtSignal()               # 时间轴上 Ctrl+C -> 复制选中的块（外面写剪贴板）
    zhantie_kuai = pyqtSignal()             # 时间轴上 Ctrl+V -> 复制一份落在原块的下面一行
    kuang_xuan = pyqtSignal(object)         # 空白处拖框选了一批块 -> 第几条的列表（外面同步列表多选）

    CHIDU_GAO = 28
    ZUO_PAD = 12
    XIA_PAD = 6
    ZUI_XIAO_GAO = 70
    ZIMU_BIAN_KUAN = 6              # 离块边缘这么近就算点在边上（能左右拖）
    ZIMU_FEN_GE_KUAN = 4            # 离字幕带下边缘这么近就算点在下边缘上（能上下拖单行高度）
    ZIMU_BIAN_KUAN_DU = 2           # 字幕块描边粗细（奇数/偶数块交替两种颜色）
    ZIMU_HANG_ZUIDA = 8             # 字幕带最多摞几行（自己往上/往下挪，想摆哪行摆哪行）
    BO_SHOU_KUAN = 16               # 波形左上角调振幅的小把手多大（正方形边长）
    KUANG_DONG_PX = 4               # 空白处按住拖这么多像素才算框选（不到就算点了一下空白）
    XIFU_PX = 8                     # 拖块时离别的块边缘这么近（像素）就吸上去
    GENSUI_BI = 0.25                # 自动跟随时播放头放在视图左边这个比例处

    _BUCHANG_MIAO = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(self.ZUI_XIAO_GAO)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self._shichang = 0
        self._bofangtou = 0
        self._fps = 0.0
        self._tuodong = False
        self._kuaisu = False            # 外面正在拖播放头（进度条那种）：只画窄带
        self._xuanfu_x = None

        # 波形包络：每格（BO_HAO_MIAO 毫秒）一组 最大/最小振幅
        self._bo_max = np.zeros(0, dtype=np.int16)
        self._bo_min = np.zeros(0, dtype=np.int16)
        self._bo_you = 0                # 已解出来的格数
        self._bo_cankao = None          # (已算到第几格, 满量程参考幅值)
        self._bo_tishi = "正在解析音频…"
        self._bo_zong = 0               # 这条视频一共要解几格（算进度用）
        self._bo_zhengzai = False       # 音频正在解析：波形带中间画进度条
        self._bo_wancheng = ""          # 解完了在同一个位置闪一下"波形就绪"
        self._bo_wancheng_ji = QtCore.QTimer(self)
        self._bo_wancheng_ji.setSingleShot(True)
        self._bo_wancheng_ji.setInterval(BO_WANCHENG_MIAO)
        self._bo_wancheng_ji.timeout.connect(self._bo_wancheng_wan)
        self._shangci_shuaxin = 0.0     # 解析期间限频用
        self._bo_fangda = _du_bo_fangda() or BO_FANGDA_MOREN  # 波形振幅缩放（波形左下角的小把手拖它）
        self._tuo_bo = None             # 正在拖那个把手
        self._bo_shou_liang = False     # 鼠标是不是停在小把手上（停上去要亮起来）

        # 视图窗口
        self._beishu = 1.0              # 1 = 整条铺满；越大越放大
        self._shi_ms = 0                # 视图左边界
        self._shishi_gun = False        # 「时间轴实时滚动」：开着 = 播放时把播放头锁在视口中间
        self._zai_bofang = False        # 外面告诉的：现在在不在播放
        self._huatu_huancun = None      # (key, QPixmap)
        self._jing_huancun = None       # (key, QPixmap) 整条轨道的静层（见 _jing_tu）
        self._jing_ban = 0              # 静层版本号：字幕数据一改就 +1，让静层缓存作废

        # 字幕块（硬字幕提取出来的），叠在波形下半截
        self._zimu = []                 # [(起ms, 止ms, 文字), ...]
        self._zimu_tao = []             # 每块字幕摆在第几行（0 起，自己挪的）
        self._kuai_yanse = []           # 每块字幕的底色（外面按说话人用的样式色喂进来的）；None = 默认蓝
        self._xuan_zhong = -1           # 主选是第几段（跟右侧字幕列表对齐）
        self._xuan_duo = []             # 多选里都是第几段（有序；单选时只有一个）
        # 播放头现在压在哪几块上（只用来判断"要不要整条重画"：进 / 出块时那块
        # 的描边要换色，窄带擦不干净。画的时候还是实时看 _bofangtou）
        self._zai_kuai = None
        self._miao_dian = None          # Shift 连选的锚点（上一次主选那条）
        self._tuo_zimu = None           # 正在拖的那块（拖动中只改本地，松手才提交）
        self._xifu_ms = None            # 拖动中吸到了哪个时间点上（画那根吸附提示线）
        self._xuanfu_zimu = None        # 鼠标悬在哪块字幕上（描边 + 底下整条竖带）
        self._zimu_bian_yu = None       # 鼠标悬在哪个块的哪条边（换光标 / 画把手）
        self._zimu_gao_guding = _du_zimu_gao()   # 用户调过的每行字幕块高度；None = 用默认
        self._tuo_gao = None            # 正在拖字幕带下边缘调高度：{"an_y": y, "an_gao": 高度}
        self._fen_ge_liang = False      # 鼠标有没有停在字幕带下边缘上（只换光标，不画线）
        # 空白处按住拖出来的选框：{"x0","y0","x1","y1","dong"}；None = 没在框选
        self._kuang = None

    # ---- 字幕块 ----
    def shezhi_zimu(self, zimu):
        xin = [tuple(x) for x in (zimu or [])]
        # 每行位置跟着"内容"走：外面（右侧列表改完字、删完块）每次都是把整个
        # 列表重塞一遍，按索引留行号会错位，按内容认就乱不了。
        jiu = {}
        for i, t in enumerate(self._zimu):
            jiu.setdefault(t, []).append(
                self._zimu_tao[i] if i < len(self._zimu_tao) else 0
            )
        tao = []
        for t in xin:
            sheng = jiu.get(t)
            tao.append(sheng.pop(0) if sheng else 0)
        self._zimu = xin
        self._zimu_tao = tao
        if len(self._kuai_yanse) != len(self._zimu):
            self._kuai_yanse = []   # 条数变了，颜色表先作废（外面马上重喂一遍）
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        n = len(self._zimu)
        # 字幕条数变了（删了 / 合并了）：把越界的选中项清掉
        self._xuan_duo = [x for x in self._xuan_duo if 0 <= x < n]
        if self._xuan_zhong >= n:
            self._xuan_zhong = -1
        if self._xuan_zhong >= 0 and self._xuan_zhong not in self._xuan_duo:
            self._xuan_duo = [self._xuan_zhong]
        self._zai_kuai = None       # 块变了，播放头压着哪几块重新认
        self.update()

    def shezhi_kuai_yanse(self, men):
        """整表喂进每块字幕的底色（外面按说话人那个样式的主文字色算好）

        None = 这块照旧用默认蓝。长度跟现存块数对不上就整表作废 ——
        宁可全用默认色，也不要拿错颜色糊到别的块上。
        """
        men = list(men or [])
        if len(men) != len(self._zimu):
            men = []
        if men == self._kuai_yanse:
            return
        self._kuai_yanse = men
        self._jing_ban += 1     # 块色变了，静层缓存作废（见 _jing_tu）
        self.update()

    def tianjia_zimu(self, qi_ms, zhi_ms, wenben):
        """新建一块：按时间插到该在的位置，行号表跟着插一格；返回插在第几块

        原来选中的那块跟着往后挪一格，插在前面也不会选中别的块。
        """
        wei = zimu_charu_weizhi(self._zimu, qi_ms, zhi_ms)
        self._zimu.insert(wei, (int(qi_ms), int(zhi_ms), str(wenben or "")))
        self._zimu_tao.insert(wei, 0)   # 新块默认摆第一行
        if len(self._kuai_yanse) == len(self._zimu) - 1:
            self._kuai_yanse.insert(wei, None)  # 新块先按默认色，外面重喂再换
        self._jing_ban += 1             # 字幕数据变了，静层缓存作废（见 _jing_tu）
        if wei <= self._xuan_zhong:
            self._xuan_zhong += 1
        self._xuan_duo = [x + 1 if x >= wei else x for x in self._xuan_duo]
        self._zai_kuai = None       # 块变了，播放头压着哪几块重新认
        self.update()
        return wei

    def gai_zimu_wenben(self, xu, wenben):
        """只改这一块的文字，别的都不动

        字幕编辑区每敲一个字就走这儿：整份重塞一遍（shezhi_zimu）会把行号也
        重算一遍，敲字的时候没必要，也卡。
        """
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, jiu = self._zimu[xu]
        wenben = str(wenben or "")
        if wenben == jiu:
            return
        self._zimu[xu] = (qi, zhi, wenben)
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        self.update()

    def qingkong_zimu(self):
        if self._zimu:
            self._zimu = []
            self._zimu_tao = []
            self._kuai_yanse = []
            self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
            self._xuan_zhong = -1
            self._xuan_duo = []
            self._miao_dian = None
            self._zai_kuai = None
            self.update()

    def zimu_shu(self):
        return len(self._zimu)

    def zimu_liebiao(self):
        """时间轴上现在的字幕块（起, 止, 文字）"""
        return list(self._zimu)

    def shezhi_kuai_bian(self, qi_ms=None, zhi_ms=None):
        """把选中的那一条字幕块的开始 / 结束时间设成指定毫秒（帧级夹住）

        只认单选。给 qi_ms 就改开始，给 zhi_ms 就改结束；至少留一帧的宽度。
        播放头落在够不着的地方（给开始时间时人已经在块尾后面，或给结束时间时
        人还在块头前面）会做出一块倒挂的块，这种直接不认，原样不动。
        直接改本地块时间，再喊外面同步右侧列表 / SRT。
        """
        xuan = self.xuan_zhong_liebiao()
        if len(xuan) != 1 or self._shichang <= 0:
            return False
        fps = float(self._fps or 0)
        if fps <= 0:
            return False
        xu = xuan[0]
        qi, zhi, wen = self._zimu[xu]
        jian = max(1, int(round(1000.0 / fps)))
        if qi_ms is not None:
            xin = max(0, min(int(round(qi_ms)), self._shichang))
            if xin > zhi - jian:
                return False
            qi = xin
        if zhi_ms is not None:
            xin = max(0, min(int(round(zhi_ms)), self._shichang))
            if xin < qi + jian:
                return False
            zhi = xin
        if (qi, zhi) == self._zimu[xu][:2]:
            return False
        self._zimu[xu] = (qi, zhi, wen)
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        self.update()
        self.zimu_nuo.emit([(xu, qi, zhi)])
        return True

    def jin_tie(self, fangxiang):
        """把选中的那一条字幕块的边紧贴到相邻块的边上

        fangxiang=+1：块尾贴到下一块的块头；fangxiang=-1：块头贴到上一块的块尾。
        只认单选。邻块按开始时间认（内部列表顺序不一定排好），贴完会倒挂或者
        不够一帧宽就不认，原样不动。
        """
        xuan = self.xuan_zhong_liebiao()
        if len(xuan) != 1 or self._shichang <= 0:
            return False
        fps = float(self._fps or 0)
        if fps <= 0:
            return False
        xu = xuan[0]
        paixu = sorted(
            range(len(self._zimu)),
            key=lambda i: (self._zimu[i][0], self._zimu[i][1]),
        )
        wei = paixu.index(xu) + fangxiang
        if not 0 <= wei < len(paixu):
            return False
        lin = paixu[wei]
        qi, zhi, wen = self._zimu[xu]
        jian = max(1, int(round(1000.0 / fps)))
        if fangxiang > 0:
            zhi = int(self._zimu[lin][0])
            if zhi < qi + jian:
                return False
        else:
            qi = int(self._zimu[lin][1])
            if qi > zhi - jian:
                return False
        if (qi, zhi) == self._zimu[xu][:2]:
            return False
        self._zimu[xu] = (qi, zhi, wen)
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        self.update()
        self.zimu_nuo.emit([(xu, qi, zhi)])
        return True

    # ---- 选中（单选 / 多选）----
    def xuan_zhong_liebiao(self):
        """现在选中了哪几条（从小到大排好）"""
        return sorted(int(x) for x in self._xuan_duo)

    def nuo_xuan_zhong(self, zhen_shu):
        """选中的字幕块整体左右挪 zhen_shu 帧（撞到片头 / 片尾就整组停）

        直接改本地的块时间，不等外面回填 —— 连着调才能一下一下累加。
        改完喊外面同步右侧列表 / SRT。

        注意：方向键已经不走这条路了（方向键只挪播放头，块的起止用 Q/W/E/R
        拉），这里当前没有调用者，留着备用。
        """
        xuan = self.xuan_zhong_liebiao()
        if not xuan or self._shichang <= 0 or not zhen_shu:
            return False
        fps = float(self._fps or 0)
        if fps <= 0:
            return False
        dong = float(zhen_shu) * (1000.0 / fps)
        zui_zuo = min(self._zimu[xu][0] for xu in xuan)
        zui_you = max(self._zimu[xu][1] for xu in xuan)
        if dong > 0:
            dong = min(dong, self._shichang - zui_you)
        else:
            dong = -min(-dong, zui_zuo)
        if abs(dong) < 0.5:
            return False
        gai = []
        for xu in xuan:
            qi, zhi, wen = self._zimu[xu]
            xin_qi = max(0, min(self._shichang, int(round(qi + dong))))
            xin_zhi = max(0, min(self._shichang, int(round(zhi + dong))))
            self._zimu[xu] = (xin_qi, xin_zhi, wen)
            gai.append((xu, xin_qi, xin_zhi))
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        self.update()
        self.zimu_nuo.emit(gai)
        return True

    def shezhi_xuan_zhong(self, xu):
        """外部（右侧字幕列表）选了第几条，这边跟着高亮（只留这一条）"""
        xu = int(xu)
        if xu < 0 or xu >= len(self._zimu):
            xu = -1
        if xu != self._xuan_zhong or self._xuan_duo != (
            [] if xu < 0 else [xu]
        ):
            self._xuan_ze_dan(xu)
            self.update()

    def shezhi_xuan_zhong_duo(self, xu_liebiao):
        """外部（字幕列表多选 / 全选）选中了这一批，这边整批跟着高亮"""
        xu = sorted(
            {int(x) for x in (xu_liebiao or [])
             if 0 <= int(x) < len(self._zimu)}
        )
        if not xu:
            self._xuan_ze_dan(-1)
            self.update()
            return
        self._xuan_duo = list(xu)
        self._xuan_zhong = xu[-1]
        self._miao_dian = xu[-1]        # Shift 连选的锚点落在这批的最后一条
        self.update()

    def _xuan_ze_dan(self, xu):
        """只留某一条（-1 = 一条都不选）；不改外观（调用方自己刷）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            xu = -1
        self._xuan_zhong = xu
        self._xuan_duo = [] if xu < 0 else [xu]
        self._miao_dian = xu if xu >= 0 else None

    def _jia_jian_xuan(self, xu):
        """Ctrl 点击：这条没选中就加上，已选中就去掉"""
        xu = int(xu)
        if xu in self._xuan_duo:
            self._xuan_duo.remove(xu)
        else:
            self._xuan_duo.append(xu)
        self._xuan_zhong = self._xuan_duo[-1] if self._xuan_duo else -1
        self._miao_dian = xu
        self.update()

    def _lian_xuan(self, xu):
        """Shift 点击：从锚点到这条，中间整片一起选上"""
        xu = int(xu)
        mao = self._miao_dian
        if mao is None or not (0 <= mao < len(self._zimu)):
            mao = xu
        qi, zhi = (mao, xu) if mao <= xu else (xu, mao)
        self._xuan_duo = list(range(qi, zhi + 1))
        self._xuan_zhong = xu
        self.update()

    def quan_xuan_kuai(self):
        """Ctrl+A：整条轨道上的块全选上

        跟框选一条路：选中一批之后喊外面，让字幕列表也整批跟着选上。
        """
        n = len(self._zimu)
        if n <= 0:
            return
        self._xuan_duo = list(range(n))
        self._xuan_zhong = n - 1
        self._miao_dian = 0
        self.update()
        self.kuang_xuan.emit(list(self._xuan_duo))

    def gundong_dao_zimu(self, xu):
        """把第 xu 段滚进视野（只动视图，不动播放头）

        这块的起点已经在视野里就什么都别做 —— 贴着屏幕边缘、尾巴探出去一点
        的那种块本来就在眼前，再按"整块要装得下"去判，就会把它拽到正中间，
        看着就是"块自己乱跑"。只有起点也在视野外（比如在右侧列表里选了很后
        面的一条）才真滚过去。
        """
        if not (0 <= int(xu) < len(self._zimu)):
            return
        qi, zhi, _w = self._zimu[int(xu)]
        ke = self._keshi_ms()
        if ke > 0 and self._shi_ms <= int(qi) < self._shi_ms + ke:
            return
        self.gundong_dao_ms(qi, int(zhi) - int(qi))

    def gundong_dao_ms(self, qi_ms, chang_ms=0):
        """让 [qi_ms, qi_ms+chang_ms] 这段落在视野里；已经在视野里就不动"""
        ke = self._keshi_ms()
        if ke <= 0 or self._shichang <= 0:
            return
        qi_ms = int(qi_ms)
        jie = qi_ms + max(0, int(chang_ms))
        if qi_ms >= self._shi_ms and jie <= self._shi_ms + ke:
            return
        if jie - qi_ms <= ke:
            # 这一段装得下：摆中间
            zhong = (qi_ms + jie) / 2.0
            self._shi_ms = int(round(zhong - ke / 2.0))
        else:
            # 比视野还长：对齐段首，左边留一点余量
            self._shi_ms = qi_ms - int(round(ke * 0.1))
        self._qia_zheng()
        self._huatu_huancun = None
        self.shitu_bian.emit()
        self.update()

    # ---- 外部接口 ----
    def shezhi_shichang(self, ms):
        self._shichang = max(0, int(ms or 0))
        self._beishu = 1.0
        self._shi_ms = 0
        self._huatu_huancun = None
        self.shitu_bian.emit()
        self.update()

    def _bofangtou_zai_kuai(self):
        """播放头现在压在哪几块上（只给"要不要整条重画"用）

        画的时候不靠它 —— 那边直接看 _bofangtou 实时判断；这个只用来发现
        "进 / 出块"这个时刻。半开区间，跟画的时候一个口径。
        """
        t = self._bofangtou
        return tuple(
            xu
            for xu, (qi, zhi, _wen) in enumerate(self._zimu)
            if min(int(qi), int(zhi)) <= t < max(int(qi), int(zhi))
        )

    def shezhi_bofangtou(self, ms, gensui=True):
        if self._tuodong:
            # 正在用鼠标拖播放头：这时候只认鼠标位置。
            # 外面（帧回调定时器 / 异步定位回来的结果）还在不停回写位置，
            # 跟鼠标一抢，播放头就在原地来回抖，所以拖动期间一律不理。
            return
        self._nuo_bofangtou(ms, gensui)

    def _nuo_bofangtou(self, ms, gensui=True):
        """把播放头挪到 ms 并重画

        播放头只挪一点点的时候，只重画它扫过的那一条窄带，不整条时间轴重绘
        （拖动的时候每挪一个像素就调一次，整条重绘会卡）。
        """
        jiu_x = self._ms_to_x(self._bofangtou)
        jiu_shi = self._shi_ms
        self._bofangtou = max(0, min(int(ms or 0), self._shichang or int(ms or 0)))
        if gensui and self._beishu > 1.0:
            # 实时滚动开着 + 正在播（而且这会儿不是拿鼠标在拖）：播放头锁在视口
            # 正中间；其余时候还是老规矩 —— 跑出视野才挪一屏
            if (
                self._shishi_gun
                and self._zai_bofang
                and not (self._tuodong or self._kuaisu)
            ):
                self._suoding_bofangtou()
            else:
                self._gensui_bofangtou()
        xin_x = self._ms_to_x(self._bofangtou)
        if self._tuodong or self._kuaisu:
            # 鼠标正拖着播放头：只擦播放头扫过的那一条窄带。
            # 这里特意不算"播放头压着哪一块"（那要扫一遍所有字幕块），
            # 也不整条重绘 —— 鼠标一快就会频繁整条重绘，播放头就落在
            # 鼠标后头，看着完全不跟手。块上的描边高亮等松手再统一刷。
            self.update(
                min(jiu_x, xin_x) - 10, 0, abs(xin_x - jiu_x) + 20,
                self.height(),
            )
            return
        xin_zai = self._bofangtou_zai_kuai()
        if xin_zai != self._zai_kuai:
            # 播放头进 / 出了哪块：那块的描边要换色，窄带擦不干净 —— 整条重画
            self._zai_kuai = xin_zai
            self.update()
        elif self._shi_ms != jiu_shi:
            self.update()               # 视图跟着滚了，整条重画
        elif xin_x != jiu_x:
            # 播放头只挪了一点点：只重画它扫过的那一条窄带，别整条时间轴重绘。
            # 两边余量给够 10 像素：免得线宽和抗锯齿的残影擦不干净。
            self.update(
                min(jiu_x, xin_x) - 10, 0, abs(xin_x - jiu_x) + 20,
                self.height(),
            )
        else:
            self.update()

    def bofangtou_ms(self):
        """播放头现在停在哪一毫秒"""
        return int(self._bofangtou)

    def shezhi_kuaisu_hua(self, kai):
        """外面（视频下方那条进度条）正按着拖播放头：这期间只画窄带

        不然进度条一拖，时间轴整条来回重绘，播放头就跟不上手。
        进出这个状态各重画一次：把"正播到哪块"的红圈擦掉 / 补回来。
        """
        self._kuaisu = bool(kai)
        self.update()

    def shezhi_fps(self, fps):
        try:
            self._fps = max(0.0, float(fps or 0.0))
        except (TypeError, ValueError):
            self._fps = 0.0
        self.update()

    def shezhi_bo_tishi(self, wenben):
        """波形那行的提示 / 进度

        外面一共调三次：
          · 打开视频时「正在解析音频…」→ 开始解析，画进度条
          · 解完了给空串 → 进度条收掉，原地闪一下「波形就绪」
          · 出错给「没有波形：…」→ 进度条收掉，只留错误提示
        """
        wenben = str(wenben or "")
        self._bo_tishi = wenben
        if "正在解析" in wenben:
            self._bo_wancheng = ""
            self._bo_wancheng_ji.stop()
        else:
            self._bo_zhengzai = False
            self._bo_wancheng = "波形就绪" if not wenben else ""
            if self._bo_wancheng:
                self._bo_wancheng_ji.start()
        self._huatu_huancun = None
        self.update()

    def _bo_wancheng_wan(self):
        """「波形就绪」闪完收掉"""
        self._bo_wancheng = ""
        self.update()

    # ---- 波形数据 ----
    def zhunbei_bofang(self, zong_ge):
        """按视频长度先开好波形缓冲区"""
        zong_ge = max(16, int(zong_ge))
        self._bo_max = np.zeros(zong_ge, dtype=np.int16)
        self._bo_min = np.zeros(zong_ge, dtype=np.int16)
        self._bo_you = 0
        self._bo_zong = zong_ge             # 进度条按这个算百分比
        self._bo_zhengzai = True            # 开始解析：画进度条
        self._bo_wancheng = ""
        self._bo_wancheng_ji.stop()
        self._huatu_huancun = None
        self.update()

    def qingkong_bofang(self):
        self._bo_max = np.zeros(0, dtype=np.int16)
        self._bo_min = np.zeros(0, dtype=np.int16)
        self._bo_you = 0
        self._bo_zong = 0
        self._bo_zhengzai = False
        self._bo_wancheng = ""
        self._bo_wancheng_ji.stop()
        self._huatu_huancun = None
        self.update()

    def tianjia_bofang(self, ge_hao, mx, mn):
        ge_hao = max(0, int(ge_hao))
        xu = ge_hao + len(mx)
        if xu > len(self._bo_max):
            kuan = max(xu, len(self._bo_max) * 2, 16)
            a = np.zeros(kuan, dtype=np.int16)
            a[: len(self._bo_max)] = self._bo_max
            self._bo_max = a
            b = np.zeros(kuan, dtype=np.int16)
            b[: len(self._bo_min)] = self._bo_min
            self._bo_min = b
        self._bo_max[ge_hao:xu] = mx
        self._bo_min[ge_hao:xu] = mn
        self._bo_you = max(self._bo_you, xu)
        self._huatu_huancun = None
        # 解析音频时会一段接一段地送进来，不必每段都重绘整条时间轴，
        # 否则解析那十几秒里主线程全在画画，画面跟着掉帧。
        xian = time.perf_counter()
        if xian - self._shangci_shuaxin >= 0.2:
            self._shangci_shuaxin = xian
            self.update()

    def man_liang_cheng(self):
        """画波形时的满量程幅值（这条视频自己的响度）

        按 32768 满刻度画，动画片、录屏这类音量偏低的片源会缩成细细
        一条线；改用本条视频自己的响度当满量程，波形才能像音频软件
        那样撑满整条带子。取 99.5% 分位而非最大值：个别爆音（爆炸、
        拍手）不会把整条波形压扁。放大封顶 16 倍（2048），免得全程
        近乎无声的片子把底噪放大成一片假波形。
        """
        if self._bo_you <= 0:
            return 32768.0
        if self._bo_cankao is not None and self._bo_cankao[0] == self._bo_you:
            return self._bo_cankao[1]
        zhi = np.concatenate([
            np.abs(self._bo_max[: self._bo_you].astype(np.int32))[::16],
            np.abs(self._bo_min[: self._bo_you].astype(np.int32))[::16],
        ])
        can = float(np.percentile(zhi, 99.5)) if zhi.size else 32768.0
        can = max(can, 2048.0)
        self._bo_cankao = (self._bo_you, can)
        return can

    # ---- 缩放 / 平移 ----
    def beishu(self):
        return self._beishu

    def shezhi_beishu(self, bei, ding_wei_ms=None):
        """设置放大倍数；ding_wei_ms 是希望原地不动的时间点（默认拿播放头）"""
        bei = max(1.0, min(float(SUOFANG_ZUIDA), float(bei or 1.0)))
        if abs(bei - self._beishu) < 1e-9:
            return
        if ding_wei_ms is None:
            ding_wei_ms = self._bofangtou
        quan = self._keshi_ms()
        bi = 0.5
        if quan > 0:
            bi = (float(ding_wei_ms) - self._shi_ms) / float(quan)
            bi = max(0.0, min(1.0, bi))
        self._beishu = bei
        ke = self._keshi_ms()
        self._shi_ms = int(round(float(ding_wei_ms) - bi * ke))
        self._qia_zheng()
        self._huatu_huancun = None
        self.shitu_bian.emit()
        self.update()

    def shezhi_shitu_bili(self, bi):
        """按 0~1 的比例左右平移（滚动条用）"""
        bi = max(0.0, min(1.0, float(bi or 0.0)))
        self._shi_ms = int(round(bi * self._shi_shangxian()))
        self._qia_zheng()
        self._huatu_huancun = None
        self.update()

    def shitu_bili(self):
        xian = self._shi_shangxian()
        if xian <= 0:
            return 0.0
        return max(0.0, min(1.0, self._shi_ms / float(xian)))

    def _keshi_ms(self):
        if self._shichang <= 0:
            return 0
        return max(1, int(round(self._shichang / max(1.0, self._beishu))))

    def _shi_shangxian(self):
        return max(0, self._shichang - self._keshi_ms())

    def _qia_zheng(self):
        self._shi_ms = max(0, min(int(self._shi_ms), self._shi_shangxian()))

    def _gensui_bofangtou(self):
        """播放头跑出视野时把视图挪过去"""
        ke = self._keshi_ms()
        if ke <= 0:
            return
        if (
            self._bofangtou < self._shi_ms
            or self._bofangtou > self._shi_ms + ke
        ):
            self._shi_ms = self._bofangtou - int(round(ke * self.GENSUI_BI))
            self._qia_zheng()
            self._huatu_huancun = None
            self.shitu_bian.emit()

    def shezhi_shishi_gun(self, kai):
        """「时间轴实时滚动」开关（照 ARC）：开着 = 播放时播放头定在视口中间、轴往左滚"""
        self._shishi_gun = bool(kai)
        self.update()

    def shezhi_zai_bofang(self, zai):
        """外面告诉的：现在在播还是在暂停（实时滚动只在播的时候生效）"""
        self._zai_bofang = bool(zai)

    def _suoding_bofangtou(self):
        """照 ARC：把播放头摆在视口正中间 —— 播放头看着不动，是时间轴在往左走

        只有放大到一屏装不下整条视频时才有区别（1 倍铺满时整条都在视野里，
        播放头本来就跑不出画面）。
        """
        ke = self._keshi_ms()
        if ke <= 0:
            return
        jiu = self._shi_ms
        self._shi_ms = int(round(self._bofangtou - ke / 2.0))
        self._qia_zheng()
        if self._shi_ms != jiu:
            self._huatu_huancun = None
            self.shitu_bian.emit()

    # ---- 坐标换算 ----
    def _guidao_qu(self):
        return QtCore.QRect(
            self.ZUO_PAD,
            self.CHIDU_GAO + 6,
            max(0, self.width() - self.ZUO_PAD * 2),
            max(0, self.height() - self.CHIDU_GAO - 6 - self.XIA_PAD),
        )

    def _ms_to_x(self, ms):
        qu = self._guidao_qu()
        ke = self._keshi_ms()
        if ke <= 0 or qu.width() <= 0:
            return qu.left()
        bi = (float(ms) - self._shi_ms) / float(ke)
        return int(round(qu.left() + bi * qu.width()))

    def _x_to_ms(self, x, xifu=True):
        """横坐标 -> 毫秒

        xifu=True 时把结果吸附到帧格上（定位用，保证落在整帧）；
        正在拖动播放头时用 False —— 拖的时候要跟手，吸附会让播放头
        一格一格往外蹦，看着就是抖。
        """
        qu = self._guidao_qu()
        ke = self._keshi_ms()
        if qu.width() <= 0 or ke <= 0:
            return 0
        bi = (x - qu.left()) / float(qu.width())
        ms = self._shi_ms + bi * ke
        ms = int(round(max(0.0, min(float(self._shichang), ms))))
        if xifu and self._fps > 0:
            jian = 1000.0 / self._fps
            ms = int(round(round(ms / jian) * jian))
        return max(0, min(ms, self._shichang))

    def _xuan_buchang(self):
        qu = self._guidao_qu()
        ke = self._keshi_ms()
        if self._shichang <= 0 or qu.width() <= 0 or ke <= 0:
            return 1
        mubiao = (ke / 1000.0) * 96.0 / qu.width()
        for bu in self._BUCHANG_MIAO:
            if bu >= mubiao:
                return bu
        return self._BUCHANG_MIAO[-1]

    # ---- 绘制 ----
    def paintEvent(self, event):
        qu = self._guidao_qu()
        huabi = QtGui.QPainter(self)
        # 先贴静层：背景、波形、刻度、字幕块、重叠标记这些不跟鼠标走的东西，
        # 全在这张现成的图里（Qt 只合成要擦的那一小块，所以擦一条窄带就只贴一条）
        huabi.drawPixmap(0, 0, self._jing_tu())
        # 再画实时层：播放头、它正压着的那块、鼠标悬停的高亮、吸附线、鼠标处的时间线
        self._hua_kuai_zai(huabi, qu)
        self._hua_bofangtou(huabi, qu)
        self._hua_xuanfu_zimu(huabi, qu)
        self._hua_xifu(huabi, qu)
        self._hua_xuanfu(huabi)
        self._hua_xuan_kuang(huabi)
        huabi.end()

    def _hua_xuan_kuang(self, huabi):
        """框选时那个框：淡蓝填一层 + 主题色细边（照 ARC，画在最上面）"""
        if self._kuang is None or not self._kuang["dong"]:
            return
        se = QtGui.QColor(_ys()["zhuse"])
        tian = QtGui.QColor(se)
        tian.setAlpha(48)               # 底下一层淡淡的，字幕块还看得清
        huabi.setPen(QtGui.QPen(se, 1))
        huabi.setBrush(tian)
        huabi.drawRect(self._kuang_ju())
        huabi.setBrush(Qt.NoBrush)

    def _hua_bo_jindu(self, huabi, bo, c):
        """波形带中间那条解析进度：解到几成就填几成，解完写「波形就绪」

        进度按「已解格数 / 一共要解几格」算 —— 一共几格在 zhunbei_bofang
        里照视频长度算好了（每格 BO_HAO_MIAO 毫秒），跟 ffmpeg 这一下送出
        了多少无关，所以能一路走到 100%，不用去 CMD 看日志。
        """
        zong = max(1, int(self._bo_zong))
        if self._bo_wancheng:
            bili = 1.0
            shuo = self._bo_wancheng
        else:
            bili = min(1.0, self._bo_you / float(zong))
            shuo = f"正在解析音频 {int(round(bili * 100))}%"
        gao = 26
        kuan = max(220, min(460, bo.width() // 3))
        jing = QtCore.QRectF(
            bo.center().x() - kuan / 2.0,
            bo.center().y() - gao / 2.0,
            kuan,
            gao,
        )
        # 照「字号 N」那个浮标做：黑底 + 白字，整块不透明，压在白波形带上
        # 也一眼看得清。进度是块里一条绿从左往右长；整体形状走裁剪，所以
        # 长到一半时右端是平的，外框不会跟着变圆。
        kuang = QtGui.QPainterPath()
        kuang.addRoundedRect(jing, 4.0, 4.0)
        huabi.save()
        huabi.setRenderHint(QtGui.QPainter.Antialiasing, True)   # 圆角才不毛
        huabi.setPen(Qt.NoPen)
        huabi.setBrush(QtGui.QColor(c["wenzi"]))          # 黑底
        huabi.drawPath(kuang)
        if bili > 0:
            huabi.setClipPath(kuang)
            huabi.setBrush(QtGui.QColor(c["qing"]))       # 已经解好的部分
            huabi.drawRect(
                QtCore.QRectF(
                    jing.left(), jing.top(), jing.width() * bili, gao
                )
            )
        huabi.restore()
        # 白字：黑底白字本来就清楚。这里用 drawText 而不是画路径 —— 路径走的是
        # 图形绘制、不带字体抗锯齿，字会发虚（上面那版就是），drawText 走系统
        # 字体渲染，边缘才实。
        jiu_ziti = huabi.font()
        ziti = QtGui.QFont(jiu_ziti)
        ziti.setPixelSize(16)
        ziti.setBold(True)
        huabi.setFont(ziti)
        huabi.setPen(QtGui.QColor(c["beijing2"]))
        huabi.drawText(jing, Qt.AlignCenter, shuo)
        huabi.setFont(jiu_ziti)

    def _jing_tu(self):
        """整条轨道的"静层"：背景 + 波形 + 刻度 + 字幕块 + 重叠标记 + 波形把手

        这些东西每画一遍都要几十毫秒（几千块得一块块画、重叠区间得整个重算
        一遍），可它们只在"视图挪了 / 字幕改了 / 尺寸变了 / 播放头换了块"的
        时候才变。所以合成一张图存起来，重绘时只贴这张图；播放头、悬停高亮、
        吸附线这些跟着鼠标每一动都在变的东西，才实时盖在上面。
        """
        qu = self._guidao_qu()
        key = (
            self._jing_ban,                 # 字幕数据（内容 / 起止 / 摆第几行）改过
            self._shi_ms,                   # 视图左边界
            self._keshi_ms(),               # 一屏铺多少毫秒（= 缩放倍率）
            self._shichang,
            self._fps,
            self.width(),
            self.height(),
            self._bo_you,                   # 波形解到第几格
            self._bo_zhengzai,              # 正在解析：带中间那条进度条
            self._bo_wancheng,              # 解完闪的「波形就绪」
            self._bo_fangda,
            self._bo_tishi,
            self.man_liang_cheng(),
            self._zimu_gao(qu),
            self._xuan_zhong,
            tuple(self._xuan_duo),
            self._bo_shou_liang,
            ZHUTI,                          # 换主题了整张都得重画
        )
        if self._jing_huancun is not None and self._jing_huancun[0] == key:
            return self._jing_huancun[1]

        c = _ys()
        tu = QtGui.QPixmap(max(1, self.width()), max(1, self.height()))
        tu.fill(QtGui.QColor(c["beijing2"]))
        huabi = QtGui.QPainter(tu)

        bo = self._bo_qu(qu)
        huabi.fillRect(bo, QtGui.QColor(c["bo_di"]))

        # 中线：整条一条、颜色极淡（深色主题照 ARC 取色 #373839）。粗细看
        # BO_ZHONG_GAO。画在波形底下，有波形的地方让波形自己盖住
        if self._bo_you > 0:
            zhong_y = bo.top() + bo.height() / 2.0
            huabi.setPen(QtGui.QPen(QtGui.QColor(c["bo_zhong"]), BO_ZHONG_GAO))
            huabi.drawLine(bo.left(), int(zhong_y), bo.right(), int(zhong_y))

        huabi.drawPixmap(bo.left(), bo.top(), self._qu_bofang_tupian(bo))

        # 正在解析音频：波形带中间压一条进度（解完在原地闪一下「波形就绪」）
        if self._bo_zhengzai or self._bo_wancheng:
            self._hua_bo_jindu(huabi, bo, c)
        elif self._bo_you <= 0:
            # 视图里没有波形数据时给个提示
            huabi.setPen(QtGui.QColor(c["wenzi_ci"]))
            if self._shichang <= 0:
                tishi = "（还没打开视频）"
            else:
                tishi = self._bo_tishi or "（没有音频波形）"
            huabi.drawText(bo, Qt.AlignCenter, tishi)

        self._hua_zimu(huabi, qu)
        self._hua_chidu(huabi, qu)
        # 重叠标记放在刻度之后画：它的竖线要竖到刻度条里去，先画会被刻度条盖掉
        self._hua_chongdie(huabi, qu)
        self._hua_bo_shou(huabi)
        huabi.end()

        self._jing_huancun = (key, tu)
        return tu

    # ---- 字幕块 ----
    def _zimu_gao_shangxian(self, qu):
        """单行字幕块最高能到多少

        字幕块是压在波形上的上层，加高不占波形的地方，所以一行最高能盖满整条轨道区。
        """
        return max(ZIMU_GAO_ZUI_XIAO, min(ZIMU_GAO_ZUIDA, int(qu.height())))

    def _zimu_gao(self, qu):
        """字幕带占多高 = 每行高 × 行数

        存的是每一行的高（不是整条带子的总高）：挪块换行、行数一变，只有带子
        整体跟着变高变矮，每一行的高度雷打不动 —— 块不会因为换行就长高。
        默认 ZIMU_GAO_MOREN；自己拖过就按调好的来（记在配置里，下次打开还在）。
        """
        mei_hang = (
            ZIMU_GAO_MOREN
            if self._zimu_gao_guding is None
            else self._zimu_gao_guding
        )
        mei_hang = max(
            ZIMU_GAO_ZUI_XIAO, min(int(mei_hang), self._zimu_gao_shangxian(qu))
        )
        return mei_hang * max(1, self._zimu_hang_shu())

    def _zimu_qu(self, qu):
        """字幕块那条带子：轨道区最上面一条（波形上方）"""
        return QtCore.QRect(qu.left(), qu.top(), qu.width(), self._zimu_gao(qu))

    def _zimu_hang(self):
        """每块字幕摆在第几行（0 起），跟 _zimu 一一对应

        行是自己挪的（拖块上下 / 右键菜单），不按重叠自动排 —— 想摆哪行摆哪行。
        """
        n = len(self._zimu)
        tao = list(self._zimu_tao)
        if len(tao) < n:
            tao.extend([0] * (n - len(tao)))
        elif len(tao) > n:
            tao = tao[:n]
        return [max(0, min(int(x), self.ZIMU_HANG_ZUIDA - 1)) for x in tao]

    def _zimu_hang_shu(self):
        """字幕带现在有几行（= 最下面那一块的行号 + 1，至少一行）"""
        tao = self._zimu_hang()
        return max(1, max(tao) + 1) if tao else 1

    def _gai_zimu_hang(self, xu_liebiao, xin_hang):
        """把这几块挪到第 xin_hang 行（一次挪一整批，多选时一起动）"""
        n = len(self._zimu)
        xin_hang = max(0, min(int(xin_hang), self.ZIMU_HANG_ZUIDA - 1))
        gai = False
        for xu in xu_liebiao:
            xu = int(xu)
            if 0 <= xu < n and self._zimu_tao[xu] != xin_hang:
                self._zimu_tao[xu] = xin_hang
                self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
                gai = True
        if gai:
            self.update()
        return gai

    def zimu_hang(self, xu):
        """第 xu 块现在摆在第几行（重复行要照原件往下沉一行）"""
        tao = self._zimu_hang()
        xu = int(xu)
        return tao[xu] if 0 <= xu < len(tao) else 0

    def shezhi_hang(self, dui):
        """按 [(第几条, 第几行), ...] 把行号摆上去

        整份 shezhi_zimu 是按"内容"认行号的，重复出来的块跟原件内容一模一样，
        认不到旧行号会落到第一行；所以重复完得再喊这一声，把副本沉到原件下面。
        """
        tao = self._zimu_hang()
        n = len(self._zimu)
        gai = False
        for xu, hang in dui or ():
            xu = int(xu)
            hang = max(0, min(int(hang), self.ZIMU_HANG_ZUIDA - 1))
            if 0 <= xu < n and tao[xu] != hang:
                tao[xu] = hang
                gai = True
        if gai:
            self._zimu_tao = tao
            self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
            self.update()
        return gai

    def _yi_zimu_hang(self, fangxiang):
        """选中的块整批往上 / 往下挪一行（-1 上移，+1 下移）

        往下挪到底会再开一行（跟别人软件一样，字幕带是个能摞好几行的轨道层）。
        """
        xuan = self.xuan_zhong_liebiao()
        if not xuan or not self._zimu:
            return
        gai = False
        for xu in xuan:
            if 0 <= xu < len(self._zimu_tao):
                xin = max(
                    0,
                    min(
                        self._zimu_tao[xu] + int(fangxiang),
                        self.ZIMU_HANG_ZUIDA - 1,
                    ),
                )
                if xin != self._zimu_tao[xu]:
                    self._zimu_tao[xu] = xin
                    self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
                    gai = True
        if gai:
            self.update()

    def _zimu_hang_kuang(self, tiao, hang, hangshu):
        """第 hang 行（一共 hangshu 行）在字幕带里占的那一条

        整条字幕带按行数等分：行数不变时把字幕带拉高，每行就更高、块更大更清楚。
        """
        hangshu = max(1, int(hangshu))
        gao = tiao.height() / float(hangshu)
        top = tiao.top() + int(round(int(hang) * gao))
        di = tiao.top() + int(round((int(hang) + 1) * gao))
        return QtCore.QRect(tiao.left(), top, tiao.width(), max(1, di - top))

    def _zai_fen_ge_xian(self, y):
        """鼠标是不是落在字幕带下边缘上（上下拖它调高度，不画任何线）"""
        if self._shichang <= 0:
            return False
        qu = self._guidao_qu()
        if qu.height() <= 0:
            return False
        tiao = self._zimu_qu(qu)
        return abs(y - tiao.bottom()) <= self.ZIMU_FEN_GE_KUAN

    def _tuo_gao_dao(self, y):
        """拖动中：按鼠标上下位置改每行字幕块的高度（夹在允许范围内）

        鼠标拖的是整条带子的底边，落到"每行"上要除以行数 —— 存的才是每行高。
        """
        if self._tuo_gao is None:
            return
        qu = self._guidao_qu()
        hangshu = max(1, self._zimu_hang_shu())
        zong = self._tuo_gao["an_gao"] + (y - self._tuo_gao["an_y"])
        zui_di = ZIMU_GAO_ZUI_XIAO * hangshu
        zong = max(zui_di, min(int(zong), max(zui_di, int(qu.height()))))
        mei_hang = int(round(zong / float(hangshu)))
        if mei_hang != self._zimu_gao_guding:
            self._zimu_gao_guding = mei_hang
            self.update()

    def _bo_qu(self, qu):
        """波形占满整条轨道区，画在最底层

        字幕块是上层，半透明地盖在波形上 —— 加高字幕块只是往上多盖一点，
        不会把波形压矮。两个图层互不干扰（跟别人的软件一样）。
        """
        return QtCore.QRect(qu.left(), qu.top(), qu.width(), qu.height())

    def _bo_shou_qu(self):
        """波形左下角那个小把手占的位置（按住上下拖它 = 调波形振幅缩放）

        放左下角是因为左上角被字幕块压着（字幕块是上层），放上去会被盖住。
        """
        if self._shichang <= 0:
            return QtCore.QRect()
        bo = self._bo_qu(self._guidao_qu())
        if bo.width() < self.BO_SHOU_KUAN * 2:
            return QtCore.QRect()
        if bo.height() < self.BO_SHOU_KUAN + 6:
            return QtCore.QRect()
        return QtCore.QRect(
            bo.left() + 2,
            bo.bottom() - 1 - self.BO_SHOU_KUAN,
            self.BO_SHOU_KUAN,
            self.BO_SHOU_KUAN,
        )

    def _shezhi_bo_fangda(self, fang):
        """设波形振幅缩放倍率（改了就作废波形缓存，下次重画）"""
        fang = max(BO_FANGDA_ZUI_XIAO, min(float(fang), BO_FANGDA_ZUIDA))
        if abs(fang - self._bo_fangda) < 1e-6:
            return
        self._bo_fangda = fang
        self._huatu_huancun = None
        self.update()

    def _bo_fangda_dao(self, y):
        """拖把手：往上拖 = 放大，往下拖 = 缩小（挪 60 像素翻一倍）"""
        if self._tuo_bo is None:
            return
        self._shezhi_bo_fangda(
            self._tuo_bo["an_fang"]
            * (1.0 + (self._tuo_bo["an_y"] - y) / 60.0)
        )

    def _hua_bo_shou(self, huabi):
        """波形左上角的小把手：上下双箭头，提示这里能按住上下拖"""
        shou = self._bo_shou_qu()
        if shou.isEmpty():
            return
        c = _ys()
        liang = self._bo_shou_liang or self._tuo_bo is not None
        di = QtGui.QColor(c["beijing2"])
        di.setAlpha(230 if liang else 150)
        huabi.setPen(Qt.NoPen)
        huabi.setBrush(di)
        huabi.drawRoundedRect(shou, 3, 3)

        yanse = QtGui.QColor(c["zhuse"] if liang else c["wenzi_ci"])
        cx = shou.center().x()
        huabi.setBrush(yanse)
        huabi.setPen(QtGui.QPen(yanse, 1))
        huabi.drawLine(cx, shou.top() + 5, cx, shou.bottom() - 5)
        huabi.setPen(Qt.NoPen)
        huabi.drawPolygon(
            QtGui.QPolygon(
                [
                    QtCore.QPoint(cx, shou.top() + 3),
                    QtCore.QPoint(cx - 3, shou.top() + 7),
                    QtCore.QPoint(cx + 3, shou.top() + 7),
                ]
            )
        )
        huabi.drawPolygon(
            QtGui.QPolygon(
                [
                    QtCore.QPoint(cx, shou.bottom() - 3),
                    QtCore.QPoint(cx - 3, shou.bottom() - 7),
                    QtCore.QPoint(cx + 3, shou.bottom() - 7),
                ]
            )
        )
        huabi.setBrush(Qt.NoBrush)

    def _hua_zimu(self, huabi, qu):
        if not self._zimu or self._shichang <= 0 or qu.width() <= 0:
            return
        tiao = self._zimu_qu(qu)
        hang = self._zimu_hang()
        hangshu = self._zimu_hang_shu()

        # 默认所有块一个样：半透明蓝底 + 白字，选不选中都不变。
        # 套过槽位（说话人非空）的块，外面会按它那个说话人用的样式的主文字色
        # 喂一份颜色进来（_kuai_yanse），没喂到的照旧用这个默认蓝。
        moren = QtGui.QColor(YANSE_ZIMU)
        moren.setAlpha(ZIMU_TOUMING_DU)
        ziti = huabi.font()
        ziti.setPointSize(8)
        huabi.setFont(ziti)
        for xu, (qi, zhi, wenben) in enumerate(self._zimu):
            hang_qu = self._zimu_hang_kuang(tiao, hang[xu], hangshu)
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 < x1:
                x1, x2 = x2, x1
            if x2 - x1 < 3:
                x2 = x1 + 3
            if x2 < qu.left() or x1 > qu.right():
                continue
            kuan = x2 - x1
            kuai = QtCore.QRect(
                x1, hang_qu.top(), max(1, kuan), hang_qu.height()
            )
            kuai = kuai.intersected(tiao)
            if kuai.width() <= 0:
                continue
            # 这块的底色：外面喂了就用外面的（说话人那个样式的色），没有就默认蓝
            yanse = moren
            if xu < len(self._kuai_yanse) and self._kuai_yanse[xu] is not None:
                yanse = self._kuai_yanse[xu]
            xuan = xu in self._xuan_duo
            if xuan:
                # 选中：只在块内部描一圈绿框，底色和文字都保持原样（跟别人
                # 软件一样，选中只是"圈一下"，内容一个像素都不动）。
                # 绿框 5 像素，比普通块那圈 2 像素的描边粗，一眼看得出选中。
                bian = QtGui.QColor(YANSE_ZIMU_XUAN)
                kuan_bi = 5
            else:
                # 紧挨着的块靠描边分个数：奇数块、偶数块交替用两种深色描边，
                # 交界处永远是"这块的深蓝边 | 那块的深棕边"，一眼数得清几块。
                # （播放头正压着的那块的红边不画在这儿 —— 那是跟着播放头每帧
                # 在变的，画在这儿会让整条轨道静层反复作废，见 _hua_kuai_zai）
                bian = QtGui.QColor(
                    YANSE_ZIMU_BIAN_A if xu % 2 == 0 else YANSE_ZIMU_BIAN_B
                )
                kuan_bi = self.ZIMU_BIAN_KUAN_DU
            # 底色直接铺在时间轴背景上（半透明），再沿块的四条内边刷一圈描边色。
            # 描边全刷在自己块里，不糊到隔壁块上——两块紧挨着时交界就是
            # "这块的描边 | 那块的描边"，一眼数得清块数。
            huabi.setPen(Qt.NoPen)
            huabi.fillRect(kuai, yanse)
            if kuan_bi * 2 + 1 > kuai.width():
                # 块太窄，四条边会互相压住、还容易伸到隔壁块上，整块刷描边色
                huabi.fillRect(kuai, bian)
            else:
                huabi.fillRect(
                    QtCore.QRect(kuai.left(), kuai.top(), kuai.width(), kuan_bi), bian
                )
                huabi.fillRect(
                    QtCore.QRect(
                        kuai.left(), kuai.bottom() - kuan_bi + 1, kuai.width(), kuan_bi
                    ),
                    bian,
                )
                huabi.fillRect(
                    QtCore.QRect(kuai.left(), kuai.top(), kuan_bi, kuai.height()), bian
                )
                huabi.fillRect(
                    QtCore.QRect(
                        kuai.right() - kuan_bi + 1, kuai.top(), kuan_bi, kuai.height()
                    ),
                    bian,
                )
            # 有文字的那块：左上角点一个小圆球，扫一眼就知道哪几段已经填了词、
            # 哪几段还是空的（空块没有球）。块太窄放不下球就不点，免得压到隔壁块。
            if wenben and kuai.width() >= 14 and kuai.height() >= 12:
                huabi.setPen(Qt.NoPen)
                huabi.setBrush(QtGui.QBrush(QtGui.QColor(YANSE_ZIMU_QIU)))
                huabi.drawEllipse(
                    QtCore.QPoint(kuai.left() + 6, kuai.top() + 6), 3, 3
                )
            if kuai.width() > 44 and wenben and kuai.height() >= 12:
                huabi.setPen(QtGui.QColor(YANSE_ZIMU_ZI))
                wen = huabi.fontMetrics().elidedText(
                    wenben, Qt.ElideRight, kuai.width() - 6
                )
                huabi.drawText(
                    kuai.adjusted(3, 0, -3, 0),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    wen,
                )
        huabi.setBrush(Qt.NoBrush)

    def _chongdie_qujian(self):
        """一行里头哪几段时间上摞了不止一块字幕 -> [(第几行, 起ms, 止ms), ...]

        首尾正好接上的（前一块的止 == 后一块的起）不算摞在一起。
        """
        if len(self._zimu) < 2:
            return []
        hang = self._zimu_hang()
        fen = {}
        for xu, h in enumerate(hang):
            qi, zhi = int(self._zimu[xu][0]), int(self._zimu[xu][1])
            if zhi < qi:
                qi, zhi = zhi, qi
            fen.setdefault(h, []).append((qi, zhi))
        jieguo = []
        for h, lie in fen.items():
            if len(lie) < 2:
                continue
            bian = []
            for qi, zhi in lie:
                bian.append((qi, 1))
                bian.append((zhi, -1))
            # 同一时刻先算结束（-1 排在 +1 前面）：首尾相接的两块不算摞
            bian.sort(key=lambda x: (x[0], x[1]))
            duo = 0
            qishi = None
            for t, d in bian:
                duo += d
                if duo >= 2 and qishi is None:
                    qishi = t
                elif duo < 2 and qishi is not None:
                    if t > qishi:
                        jieguo.append((h, qishi, t))
                    qishi = None
        jieguo.sort()
        return jieguo

    def _hua_chongdie(self, huabi, qu):
        """摞在一起的那几段：块的颜色压深，时间轴顶端再画个 L 形标出起止

        一眼就能看出哪几段撞上了、从哪儿撞到哪儿。
        """
        qujian = self._chongdie_qujian()
        if not qujian or qu.width() <= 0:
            return
        tiao = self._zimu_qu(qu)
        hangshu = self._zimu_hang_shu()
        shen = QtGui.QColor(0, 0, 0)
        shen.setAlpha(CHONGDIE_SHEN_ALPHA)
        huabi.setPen(Qt.NoPen)
        for h, qi, zhi in qujian:
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 <= x1:
                x2 = x1 + 1
            if x2 < qu.left() or x1 > qu.right():
                continue
            hang_qu = self._zimu_hang_kuang(tiao, h, hangshu)
            # 在块上面再压一层暗色 —— 重叠的那一截看着就是深了一截
            huabi.fillRect(
                QtCore.QRect(x1, hang_qu.top(), x2 - x1, hang_qu.height()), shen
            )
        # L 形标记：横线贴着字幕带顶边横跨整个重叠区间，竖线从横线的左端往上
        # 竖进时间轴的刻度条里（之前是往下垂进字幕块，正好画反了）
        huabi.save()
        huabi.setPen(QtGui.QPen(QtGui.QColor(YANSE_CHONGDIE), 2))
        y_ding = max(0, qu.top() - CHONGDIE_L_GAO)
        for _h, qi, zhi in qujian:
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 <= x1:
                x2 = x1 + 1
            if x2 < qu.left() or x1 > qu.right():
                continue
            x1 = max(x1, qu.left())
            x2 = min(x2, qu.right())
            huabi.drawLine(x1, qu.top(), x2, qu.top())
            huabi.drawLine(x1, qu.top(), x1, y_ding)
        huabi.restore()
        huabi.setBrush(Qt.NoBrush)

    def _zimu_zai_ms(self, ms):
        for qi, zhi, wenben in self._zimu:
            if qi <= ms <= zhi:
                return (qi, zhi, wenben)
        return None

    def _zimu_xu_zai_ms(self, ms):
        """这个时刻落在第几段字幕上（找不到返回 -1）"""
        for xu, (qi, zhi, _w) in enumerate(self._zimu):
            if qi <= ms <= zhi:
                return xu
        return -1

    def _zimu_zai_dian(self, x, y):
        """按像素找鼠标点在哪块字幕上 -> (第几条, 落在哪部分)

        部分："zuo" = 左边缘（拖它改开头）、"you" = 右边缘（拖它改结尾）、
        "yi" = 中间（拖它整块平移）。没点中返回 (-1, None)。
        按像素而不是按毫秒找，是为了"贴着块边点"能稳稳命中边缘。
        """
        if not self._zimu or self._shichang <= 0:
            return -1, None
        qu = self._guidao_qu()
        tiao = self._zimu_qu(qu)
        if not (tiao.top() <= y <= tiao.bottom()):
            return -1, None
        # 从后往前找：后画的块压在前面的上面，跟绘制顺序一致
        hang = self._zimu_hang()
        hangshu = self._zimu_hang_shu()
        for xu in range(len(self._zimu) - 1, -1, -1):
            hang_qu = self._zimu_hang_kuang(tiao, hang[xu], hangshu)
            if not (hang_qu.top() <= y <= hang_qu.bottom()):
                continue            # 多行轨道：点到的行不对，不是这块
            qi, zhi, _w = self._zimu[xu]
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 < x1:
                x1, x2 = x2, x1
            if x2 - x1 < 3:
                x2 = x1 + 3
            if not (x1 <= x <= x2):
                continue
            # 块太窄时分不出左右，一律当中间
            if x2 - x1 > self.ZIMU_BIAN_KUAN * 2:
                if x - x1 <= self.ZIMU_BIAN_KUAN:
                    return xu, "zuo"
                if x2 - x <= self.ZIMU_BIAN_KUAN:
                    return xu, "you"
            return xu, "yi"
        return -1, None

    def _ms_xifu_rongcha(self):
        """吸附的范围有多大：屏幕上 XIFU_PX 像素换成毫秒是多少

        按像素算，视图放大缩小后手感一样——永远都是"离得就差几个像素才吸"。
        """
        qu = self._guidao_qu()
        ke = self._keshi_ms()
        if qu.width() <= 0 or ke <= 0:
            return 0.0
        return ke * float(self.XIFU_PX) / qu.width()

    def _xifu_dao(self, ms, paichu):
        """把一个时间点吸到别的字幕块的起止边上（离得够近才吸）

        返回 (吸完的时间, 吸到了哪个时间点)；附近没别人的边就是 (ms, None)。
        paichu 是正拖的那块，不能吸到它自己的另一头上去。
        """
        rong = self._ms_xifu_rongcha()
        if rong <= 0:
            return ms, None
        zui, zui_cha = None, None
        for xu, (qi, zhi, _w) in enumerate(self._zimu):
            if xu == paichu:
                continue
            for dian in (int(qi), int(zhi)):
                cha = abs(ms - dian)
                if cha <= rong and (zui_cha is None or cha < zui_cha):
                    zui, zui_cha = dian, cha
        return (ms, None) if zui is None else (zui, zui)

    def _xifu_zheng_kuai(self, xin_qi, chang, paichu):
        """整块平移时：左右两条边都试着吸，哪条离得近就按哪条对齐

        返回 (吸完的新开头, 吸到了哪个时间点)。
        """
        a, shun_a = self._xifu_dao(xin_qi, paichu)
        b, shun_b = self._xifu_dao(xin_qi + chang, paichu)
        if shun_a is None and shun_b is None:
            return xin_qi, None
        if shun_b is None:
            return a, shun_a
        if shun_a is None:
            return b - chang, shun_b
        if abs(shun_a - xin_qi) <= abs(shun_b - (xin_qi + chang)):
            return a, shun_a
        return b - chang, shun_b

    def _zimu_tuo_dao(self, x, y=None):
        """拖动中：左右 = 改本块起止时间，上下 = 换到别的行（松手才提交）

        拖着走过别的字幕块的起止边附近时，自动吸上去对齐。
        """
        tuo = self._tuo_zimu
        if tuo is None or not (0 <= tuo["xu"] < len(self._zimu)):
            return
        if not tuo["dong"]:
            # 还在"点一下"的范围内（左右都没挪够 3 像素），先不算拖
            an_y = tuo.get("an_y")
            if abs(x - tuo["an_x"]) < 3 and (
                an_y is None or y is None or abs(y - an_y) < 3
            ):
                return
            tuo["dong"] = True

        if y is not None:
            self._zimu_tuo_hang(tuo, y)

        ms = self._x_to_ms(x)
        zui_xiao = 1
        if self._fps > 0:
            zui_xiao = max(1, int(round(1000.0 / self._fps)))   # 最短一帧
        qi, zhi = tuo["qi"], tuo["zhi"]
        shun = None
        if tuo["buwei"] == "zuo":
            qi = max(0, min(ms, zhi - zui_xiao))
            qi, shun = self._xifu_dao(qi, tuo["xu"])
            qi = max(0, min(qi, zhi - zui_xiao))
        elif tuo["buwei"] == "you":
            zhi = min(self._shichang, max(ms, qi + zui_xiao))
            zhi, shun = self._xifu_dao(zhi, tuo["xu"])
            zhi = min(self._shichang, max(zhi, qi + zui_xiao))
        else:
            chang = zhi - qi
            xin_qi = ms - (tuo["an_ms"] - tuo["qi"])
            xin_qi = max(0, min(self._shichang - chang, xin_qi))
            xin_qi, shun = self._xifu_zheng_kuai(xin_qi, chang, tuo["xu"])
            xin_qi = max(0, min(self._shichang - chang, xin_qi))
            qi, zhi = xin_qi, xin_qi + chang
        self._xifu_ms = shun
        self._zimu[tuo["xu"]] = (int(qi), int(zhi), self._zimu[tuo["xu"]][2])
        self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）
        self.update()

    def _zimu_tuo_hang(self, tuo, y):
        """拖动中上下移动：按鼠标走过的距离把这块挪到第几行

        行高用按下那一刻的行高，出发点用按下时那块原来在第几行 —— 不拿实时行高
        算，不然目标行一超出现有行数、行数就涨、行高就变，下一帧又算回原来那行，
        块就卡在最下面一行拖不出去了（原来就是这么卡住的）。
        往下拖过半行高就落到下一行，一直往下拖就一直往下，最多 ZIMU_HANG_ZUIDA 行。
        """
        tiao = self._zimu_qu(self._guidao_qu())
        if tiao.height() <= 0:
            return
        an_y = tuo.get("an_y")
        if an_y is None:
            return
        hangshu = max(1, int(tuo.get("hangshu") or 1))
        hang_gao = max(1.0, tiao.height() / float(hangshu))
        an_hang = int(tuo.get("an_hang") or 0)
        xin = an_hang + int(round((y - an_y) / hang_gao))
        xin = max(0, min(xin, self.ZIMU_HANG_ZUIDA - 1))
        xu = int(tuo["xu"])
        if xu >= len(self._zimu_tao):
            self._zimu_tao.extend([0] * (xu + 1 - len(self._zimu_tao)))
        if xin != self._zimu_tao[xu]:
            self._zimu_tao[xu] = xin
            self._jing_ban += 1     # 字幕数据变了，静层缓存作废（见 _jing_tu）

    def _qu_bofang_tupian(self, qu):
        """把当前视图这段波形画成一张图缓存起来（只画一次，之后直接贴）"""
        if qu.width() <= 0 or qu.height() <= 0:
            return QtGui.QPixmap()
        ke = self._keshi_ms()
        man = self.man_liang_cheng()
        key = (
            self._shi_ms,
            ke,
            qu.width(),
            qu.height(),
            self._bo_you,
            man,
            self._bo_fangda,
        )
        if self._huatu_huancun is not None and self._huatu_huancun[0] == key:
            return self._huatu_huancun[1]

        c = _ys()
        tupian = QtGui.QPixmap(qu.width(), qu.height())
        tupian.fill(QtCore.Qt.transparent)
        if self._bo_you <= 0 or ke <= 0:
            self._huatu_huancun = (key, tupian)
            return tupian

        # 「毫秒 -> 格号」的除数：一格 = BO_HAO_MIAO 毫秒（解码线程就是按这个
        # 格长把采样点分组的）。以前这里误用了「每格的采样点数」，等于把时间轴
        # 压缩了 BO_CAN_YANG_LV/1000 倍：屏幕上每一列只取到零点几毫秒的一个瞬间，
        # 语音那种忽高忽低的波形大半被跳过去，只剩持续大音量才画得出来。
        mei_ge_haomiao = float(max(1, BO_HAO_MIAO))
        kuan = qu.width()
        gao = qu.height()
        ban = gao / 2.0 - 2.0

        # 每一列像素覆盖的毫秒区间 -> 取这段的最大/最小振幅
        bian = np.linspace(
            self._shi_ms, self._shi_ms + ke, kuan + 1, dtype=np.float64
        )
        qi = np.clip(
            np.floor(bian / mei_ge_haomiao).astype(np.int64), 0, self._bo_you
        )
        shang = np.zeros(kuan, dtype=np.float32)
        xia = np.zeros(kuan, dtype=np.float32)
        for i in range(kuan):
            a = int(qi[i])
            b = int(qi[i + 1])
            if b <= a:
                b = a + 1
            if a >= self._bo_you:
                break
            b = min(b, self._bo_you)
            shang[i] = self._bo_max[a:b].max()
            xia[i] = self._bo_min[a:b].min()

        xian = []
        for i in range(kuan):
            # 振幅乘上缩放倍率：倍率大了波形就顶到上下边缘（超出部分被图片裁掉）
            y1 = gao / 2.0 - shang[i] * self._bo_fangda / man * ban
            y2 = gao / 2.0 - xia[i] * self._bo_fangda / man * ban
            if y2 - y1 < 1.0:
                y1 -= 0.5
                y2 += 0.5
            xian.append(QtCore.QLineF(i + 0.5, y1, i + 0.5, y2))

        hb = QtGui.QPainter(tupian)
        hb.setRenderHint(QtGui.QPainter.Antialiasing, False)
        hb.setPen(QtGui.QPen(QtGui.QColor(c["bo"]), 1))
        hb.drawLines(xian)
        hb.end()

        self._huatu_huancun = (key, tupian)
        return tupian

    def _hua_chidu(self, huabi, qu):
        c = _ys()
        if self._shichang <= 0 or qu.width() <= 0:
            return
        bu_miao = self._xuan_buchang()
        bu_ms = bu_miao * 1000
        ziti = huabi.font()
        ziti.setPointSize(8)
        huabi.setFont(ziti)
        fm = QtGui.QFontMetrics(ziti)

        huabi.fillRect(
            QtCore.QRect(qu.left(), 0, qu.width(), self.CHIDU_GAO),
            QtGui.QColor(c["mian"]),
        )

        fu = QtGui.QColor(c["biankuang_liang"])
        fu.setAlpha(70)

        # 主刻度之间再补一层小刻度（每格一分为五）：只有粗刻度时，两个标签
        # 之间一大片空，看不出时间走到哪儿了。小刻度太挤（不到 8 像素）就不画。
        ke = self._keshi_ms()
        xi_ms = int(bu_ms) // 5
        if xi_ms > 0 and ke > 0 and qu.width() * xi_ms / float(ke) >= 8:
            xiao = QtGui.QColor(c["biankuang_liang"])
            xiao.setAlpha(160)
            huabi.setPen(xiao)
            xi = int(self._shi_ms // xi_ms * xi_ms)
            xi = max(0, min(xi, self._shichang))
            while xi <= self._shichang:
                if int(xi) % int(bu_ms):
                    x = self._ms_to_x(xi)
                    if x > qu.right() + 4:
                        break
                    huabi.drawLine(x, self.CHIDU_GAO - 8, x, self.CHIDU_GAO)
                xi += xi_ms

        ms = int(self._shi_ms // bu_ms * bu_ms)
        ms = max(0, min(ms, self._shichang))
        shang_ge_y = qu.left() - 999
        while ms <= self._shichang:
            x = self._ms_to_x(ms)
            if x > qu.right() + 4:
                break
            huabi.setPen(QtGui.QColor(c["wenzi_ci"]))
            huabi.drawLine(x, self.CHIDU_GAO - 13, x, self.CHIDU_GAO)
            huabi.setPen(fu)
            huabi.drawLine(x, self.CHIDU_GAO, x, qu.bottom())
            huabi.setPen(QtGui.QColor(c["wenzi_ci"]))
            wenben = _shijian_wenben(ms, self._fps).rsplit(":", 1)[0]
            kuan = fm.horizontalAdvance(wenben)
            tx = max(qu.left(), min(qu.right() - kuan, x - kuan // 2))
            if tx > shang_ge_y + 14:
                huabi.drawText(tx, fm.ascent() + 3, wenben)
                shang_ge_y = tx + kuan
            ms += bu_ms

    def _hua_xuanfu_zimu(self, huabi, qu):
        """鼠标悬在哪块字幕上：那块描一圈亮边，底下整个轨道也亮出一条带子

        不点也能看出鼠标对着的是哪一块、这块占了多长（一直贯到轨道底下）。
        """
        xu = self._xuanfu_zimu
        if xu is None or not (0 <= xu < len(self._zimu)) or self._shichang <= 0:
            return
        qi, zhi, _w = self._zimu[xu]
        x1 = self._ms_to_x(qi)
        x2 = self._ms_to_x(zhi)
        if x2 < x1:
            x1, x2 = x2, x1
        if x2 - x1 < 3:
            x2 = x1 + 3
        tiao = self._zimu_qu(qu)
        hang = self._zimu_hang()
        hangshu = self._zimu_hang_shu()
        hang_qu = self._zimu_hang_kuang(tiao, hang[xu], hangshu)
        kuan = x2 - x1
        kuang = QtCore.QRect(x1, hang_qu.top(), max(1, kuan), hang_qu.height())

        c = _ys()
        dai = QtGui.QColor(YANSE_ZIMU)
        dai.setAlpha(50)
        # 竖带只从字幕块下边缘往下画（波形那一段），别糊到字幕块上
        huabi.fillRect(
            QtCore.QRect(
                kuang.left(),
                tiao.bottom() + 1,
                kuang.width(),
                max(0, qu.bottom() - tiao.bottom()),
            ),
            dai,
        )
        if xu in self._xuan_duo:
            # 这块已经选中：选中本身就是一圈绿框，再描一圈会正好压在绿框上，
            # 鼠标一放上去就看不见选中了。所以只留底下的竖带，不再描边。
            return
        # 描边用主题的前景色（浅色主题深、深色主题浅），两个主题下都看得见。
        # 往里缩 1 像素再画：2 像素的笔是以边界为中心铺开的，不缩就会糊到块外面去。
        huabi.setPen(QtGui.QPen(QtGui.QColor(c["wenzi"]), 2))
        huabi.setBrush(Qt.NoBrush)
        huabi.drawRect(kuang.adjusted(1, 1, -1, -1))

    def _hua_xifu(self, huabi, qu):
        """拖着字幕块吸到别的块边上时，在那条边上画一根实线

        不然只是块自己对齐上了，眼睛看不出来到底是吸上了还是差一点点。
        """
        if self._xifu_ms is None:
            return
        x = self._ms_to_x(self._xifu_ms)
        if x < qu.left() - 8 or x > qu.right() + 8:
            return
        c = _ys()
        huabi.save()
        huabi.setPen(QtGui.QPen(QtGui.QColor(c["zhuse"]), 2))
        huabi.drawLine(x, qu.top(), x, qu.bottom())
        huabi.restore()

    def _hua_kuai_zai(self, huabi, qu):
        """播放头正压着的那几块：描边换成"正播到这块"的红

        这一圈不能画进静层（见 _jing_tu）：播放的时候播放头每一换块都要重画，
        塞进静层就是每换块整条轨道重画一遍，白等一百多毫秒。所以挪到实时层，
        静层那张图一动不动，只重画这几块所在的那一小片。
        拖动播放头的时候不画：那会儿鼠标说了算，红圈跟着手乱跑没意义。
        """
        if self._tuodong or self._kuaisu or not self._zai_kuai:
            return
        tiao = self._zimu_qu(qu)
        hang = self._zimu_hang()
        hangshu = self._zimu_hang_shu()
        bian = QtGui.QColor(YANSE_ZIMU_ZAI)
        huabi.setPen(Qt.NoPen)
        for xu in self._zai_kuai:
            if not (0 <= xu < len(self._zimu)) or xu in self._xuan_duo:
                continue            # 选中的那圈绿框优先，别盖上去
            qi, zhi, _w = self._zimu[xu]
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 < x1:
                x1, x2 = x2, x1
            if x2 - x1 < 3:
                x2 = x1 + 3
            hang_qu = self._zimu_hang_kuang(tiao, hang[xu], hangshu)
            kuai = QtCore.QRect(
                x1, hang_qu.top(), max(1, x2 - x1), hang_qu.height()
            )
            kuai = kuai.intersected(tiao)
            if kuai.width() <= 0:
                continue
            kuan_bi = 5     # 播放头压着的红框 5 像素（跟选中那圈绿一样粗）
            if kuan_bi * 2 + 1 > kuai.width():
                huabi.fillRect(kuai, bian)
            else:
                huabi.fillRect(
                    QtCore.QRect(kuai.left(), kuai.top(), kuai.width(), kuan_bi), bian
                )
                huabi.fillRect(
                    QtCore.QRect(
                        kuai.left(), kuai.bottom() - kuan_bi + 1, kuai.width(), kuan_bi
                    ),
                    bian,
                )
                huabi.fillRect(
                    QtCore.QRect(kuai.left(), kuai.top(), kuan_bi, kuai.height()), bian
                )
                huabi.fillRect(
                    QtCore.QRect(
                        kuai.right() - kuan_bi + 1, kuai.top(), kuan_bi, kuai.height()
                    ),
                    bian,
                )

    def _hua_bofangtou(self, huabi, qu):
        c = _ys()
        x = self._ms_to_x(self._bofangtou)
        if x < qu.left() - 8 or x > qu.right() + 8:
            return
        huabi.setPen(QtGui.QPen(QtGui.QColor(c["hong"]), 2))
        huabi.drawLine(x, 0, x, qu.bottom())

    def _hua_xuanfu(self, huabi):
        if self._xuanfu_x is None or self._shichang <= 0:
            return
        c = _ys()
        ms = self._x_to_ms(self._xuanfu_x)
        x = self._ms_to_x(ms)
        yu = QtGui.QColor(c["zhuse"])
        yu.setAlpha(170)
        huabi.save()
        huabi.setPen(QtGui.QPen(yu, 1, Qt.PenStyle.DashLine))
        huabi.drawLine(x, self.CHIDU_GAO, x, self._guidao_qu().bottom())
        huabi.restore()

    # ---- 鼠标 ----
    def _kai_shi_tuo_zimu(self, xu, buwei, x, bao_xuan=False, y=None):
        """按下字幕块：选中它，并进入拖动态（松手才提交）

        bao_xuan=True 表示这块本来就在多选里，保持多选不动（只拖它这一块）。
        y 是按下时的上下位置：上下拖 = 把这块挪到别的行。
        """
        qi, zhi, _w = self._zimu[xu]
        if bao_xuan:
            self._miao_dian = int(xu)
            self._xuan_zhong = int(xu)
        else:
            self._xuan_ze_dan(xu)
            self.xuan_zhong.emit(xu)
        self._tuo_zimu = {
            "xu": xu,
            "buwei": buwei,
            "an_x": x,
            "an_y": y,
            "an_ms": self._x_to_ms(x),
            "qi": int(qi),
            "zhi": int(zhi),
            "dong": False,
            # 行数和这块原来在第几行都按下时锁死：拖动中行数一变行高就变，
            # 拿实时行高去算目标行会自己往上跳，越拖越乱。
            "hangshu": self._zimu_hang_shu(),
            "an_hang": self._zimu_hang()[xu],
        }
        self._xifu_ms = None

    def _fanyie(self, bu):
        """PgDn / PgUp：前后翻一屏，拿当前屏里头 / 尾那条字幕当对齐点

        往后翻（bu>0）：新视野左边界 = 当前屏最后一条字幕的**起点** —— 翻过去
        那条贴在新屏最左，后面接着的内容都露出来。
        往前翻（bu<0）：新视野右边界 = 当前屏第一条字幕的**结束点** —— 块是从
        起点往右长的，拿起点去顶右边界等于整块掉到屏外面；用结束点顶，那条才
        完整贴在新屏最右，前面接着的内容都露出来。
        屏里一条字幕都没有（大段空白）就按一屏滚，不至于按了没反应。
        只挪视野，播放头不动。
        """
        ke = self._keshi_ms()
        if ke <= 0 or ke >= self._shichang:
            return
        jie = self._shi_ms + ke
        zai = [i for i, (qi, _zhi, _w) in enumerate(self._zimu)
               if self._shi_ms <= qi < jie]
        if zai:
            if bu > 0:
                xin = self._zimu[zai[-1]][0]
            else:
                xin = self._zimu[zai[0]][1] - ke
        else:
            xin = self._shi_ms + (ke if bu > 0 else -ke)
        jiu = self._shi_ms
        self._shi_ms = int(xin)
        self._qia_zheng()
        if self._shi_ms == jiu:
            return
        self._huatu_huancun = None
        self.shitu_bian.emit()
        self.update()

    def keyPressEvent(self, event):
        # 焦点在时间轴上时，左右方向键逐帧移动播放头。时间轴自己拿到焦点后，
        # 按键不会再冒泡到外层窗口，得在这儿接住再喊外面去调帧。
        jian = event.key()
        # PgUp / PgDn：前后翻一屏（带字幕对齐，见 _fanyie）
        if jian == Qt.Key_PageUp:
            self._fanyie(-1)
            event.accept()
            return
        if jian == Qt.Key_PageDown:
            self._fanyie(1)
            event.accept()
            return
        if jian == Qt.Key_Left:
            self.fangxiang_jian.emit(-1)
            event.accept()
            return
        if jian == Qt.Key_Right:
            self.fangxiang_jian.emit(1)
            event.accept()
            return
        # Ins = 换到下一条字幕（跟字幕列表回车一个手感）
        if jian == Qt.Key_Insert:
            self.xiayitiao.emit()
            event.accept()
            return
        # Ctrl+A / Ctrl+C / Ctrl+V：全选整条轨道 / 复制选中的块 / 粘一份到下面一行
        if event.modifiers() & Qt.ControlModifier:
            if jian == Qt.Key_A:
                self.quan_xuan_kuai()
                event.accept()
                return
            if jian == Qt.Key_C:
                self.fuzhi_kuai.emit()
                event.accept()
                return
            if jian == Qt.Key_V:
                self.zhantie_kuai.emit()
                event.accept()
                return
        # 回车不在这儿接：让它冒泡到窗口那边（回车 = 建一条新字幕块）
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # 点一下就把键盘焦点拿过来：后面的空格 / 方向键 / 回车才接得住
        self.setFocus(Qt.MouseFocusReason)
        if event.button() == Qt.LeftButton and self._shichang > 0:
            x, y = event.pos().x(), event.pos().y()
            if self._bo_shou_qu().contains(x, y):
                # 按在波形左上角的小把手上：上下拖 = 调波形振幅缩放
                self._tuo_bo = {"an_y": y, "an_fang": self._bo_fangda}
                self.update()
                event.accept()
                return
            # 点在字幕块上 -> 先选中；接着拖 = 改时间
            xu, buwei = self._zimu_zai_dian(x, y)
            xiu = event.modifiers()
            # Ctrl / Shift 点击只改选择：不拖动，也不跳转
            if xu >= 0 and (xiu & Qt.ControlModifier):
                self._jia_jian_xuan(xu)
                event.accept()
                return
            if xu >= 0 and (xiu & Qt.ShiftModifier):
                self._lian_xuan(xu)
                event.accept()
                return
            # 点在已经多选中的一块上：选择保持不动，只拖这一块；
            # 没拖动的话松手只是跳到这一段开头，选择照样不变。
            bao = xu >= 0 and len(self._xuan_duo) > 1 and xu in self._xuan_duo
            if xu >= 0 and buwei != "yi":
                # 贴着块的左 / 右边缘按下的，一定是想改长短，优先当拖块
                self._kai_shi_tuo_zimu(xu, buwei, x, bao_xuan=bao, y=y)
                event.accept()
                return
            if self._zai_fen_ge_xian(y):
                # 点在下边缘的分隔线上：上下拖 = 调字幕带多高
                self._tuo_gao = {
                    "an_y": y,
                    "an_gao": self._zimu_gao(self._guidao_qu()),
                }
                self.update()
                event.accept()
                return
            if xu >= 0:
                self._kai_shi_tuo_zimu(xu, buwei, x, bao_xuan=bao, y=y)
                event.accept()
                return
            # 刻度尺那一条：只有点这儿（或按住拖）才定位播放头，
            # 跟别人的软件一样，播放头只认时间刻度。选中照旧不动。
            if y <= self.CHIDU_GAO:
                self._tuodong = True
                self.tuo_kaishi.emit()      # 先跟外面说一声：鼠标接管播放头了
                ms = self._x_to_ms(x)
                self._bofangtou = max(0, min(int(ms), self._shichang or int(ms)))
                self.tiaozheng.emit(int(ms))
                self.update()
                event.accept()
                return
            # 其它空白（字幕带的空处、下面波形那一片）：不动播放头。
            # 按住拖 = 拉一个框框选字幕块（照 ARC）；原地松手（没拖）= 清空选中。
            self._kuang = {"x0": x, "y0": y, "x1": x, "y1": y, "dong": False}
            self._xuanfu_zimu = None
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._bo_shou_qu().contains(event.pos()):
                # 双击小把手：波形振幅缩放回到默认
                self._tuo_bo = None
                self._shezhi_bo_fangda(BO_FANGDA_MOREN)
                _cun_gao_ji(bo_fangda=None)
                event.accept()
                return
            if self._zai_fen_ge_xian(event.pos().y()):
                # 双击分隔线：字幕带高度恢复默认
                self._tuo_gao = None
                self._zimu_gao_guding = None
                _cun_zimu_gao(None)
                self.update()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        self._xuanfu_x = event.pos().x()
        if self._tuo_bo is not None:
            self._bo_fangda_dao(event.pos().y())
        elif self._tuo_gao is not None:
            self._tuo_gao_dao(event.pos().y())
        elif self._tuo_zimu is not None:
            self._zimu_tuo_dao(event.pos().x(), event.pos().y())
        elif self._tuodong and self._shichang > 0:
            self._xuanfu_zimu = None
            # 拖播放头：本地先按鼠标位置画（不吸附帧格，跟着手走），
            # 再把位置告诉外面去定位视频画面。拖动期间外面的回写会被
            # shezhi_bofangtou 挡掉，所以播放头不会被拉回去。
            # 只擦播放头扫过的那一条窄带（别整条重画 —— 波形加所有字幕块
            # 每挪一个像素画一遍，就是拖起来卡的原因）。
            ms = self._x_to_ms(event.pos().x(), xifu=False)
            self._nuo_bofangtou(ms, gensui=False)
            self.tiaozheng.emit(int(ms))
        elif self._kuang is not None:
            # 正在空白处拉框选
            self._kuang_dao(event.pos().x(), event.pos().y())
        else:
            # 光标提示：块左右边缘 = 左右拉伸（拖长短）、字幕带下边缘 =
            # 上下拉伸（拖高度）、波形左上角小把手 = 上下拉伸（调振幅）、
            # 别的地方 = 手型
            x, y = event.pos().x(), event.pos().y()
            xu, buwei = self._zimu_zai_dian(x, y)
            # 鼠标对着哪块：那块描边、底下整条轨道也亮出来（不点也有反馈）
            self._xuanfu_zimu = xu if xu >= 0 else None
            bian = (xu, buwei) if (xu >= 0 and buwei != "yi") else None
            zai_shou = bian is None and self._bo_shou_qu().contains(x, y)
            zai_fen = bian is None and not zai_shou and self._zai_fen_ge_xian(y)
            huan = False
            if bian != self._zimu_bian_yu:
                self._zimu_bian_yu = bian
                huan = True
            if zai_fen != self._fen_ge_liang:
                self._fen_ge_liang = zai_fen
                huan = True
            if zai_shou != self._bo_shou_liang:
                self._bo_shou_liang = zai_shou
                huan = True
            if huan:
                if zai_shou:
                    self.setCursor(Qt.SizeVerCursor)
                elif bian is not None:
                    self.setCursor(Qt.SizeHorCursor)
                elif zai_fen:
                    self.setCursor(Qt.SizeVerCursor)
                else:
                    self.setCursor(Qt.PointingHandCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._tuo_bo is not None:
                # 松手：振幅缩放就这样了（拖动过程里已经实时改过），记下来下次打开还是这么大
                self._tuo_bo = None
                _cun_gao_ji(bo_fangda=round(self._bo_fangda, 4))
                self.update()
            if self._tuo_gao is not None:
                # 松手：把这次调好的高度记下来，下次打开还是这么高
                self._tuo_gao = None
                if self._zimu_gao_guding is not None:
                    _cun_zimu_gao(self._zimu_gao_guding)
                self.update()
            tuo = self._tuo_zimu
            if tuo is not None:
                self._tuo_zimu = None
                self._xifu_ms = None
                xu = tuo["xu"]
                if tuo["dong"] and 0 <= xu < len(self._zimu):
                    qi, zhi, _w = self._zimu[xu]
                    if (int(qi), int(zhi)) != (tuo["qi"], tuo["zhi"]):
                        self.zimu_tuo.emit(int(xu), int(qi), int(zhi))
                self.update()
            if self._tuodong:
                # 松手了：这时才把位置吸附到整帧并提交一次，
                # 播放头落到精确帧上，也把外面的回写解禁。
                self._tuodong = False
                ms = self._x_to_ms(event.pos().x())
                self._bofangtou = max(0, min(int(ms), self._shichang or int(ms)))
                self.tuo_jieshu.emit()       # 先跟外面说一声：手松了
                self.tiaozheng.emit(int(ms))  # 这一下才真定位（画面 + 音频）
                self.update()
            self._tuodong = False
            self._kuang_fang()
        super().mouseReleaseEvent(event)

    def _kuang_dao(self, x, y):
        """拉框选中：起点到鼠标之间那个矩形，四个方向都能拉

        照 ARC：拖的过程中**碰到就选中**，不用等松手 —— 框拉到哪儿，
        被框压住的块当场就亮起来，往回缩也会当场取消。
        """
        k = self._kuang
        if k is None:
            return
        k["x1"], k["y1"] = x, y
        if not k["dong"] and (
            abs(k["x1"] - k["x0"]) > self.KUANG_DONG_PX
            or abs(k["y1"] - k["y0"]) > self.KUANG_DONG_PX
        ):
            # 拖过几个像素才算框选（手抖一下不算）
            k["dong"] = True
            self.setCursor(Qt.CrossCursor)
        if k["dong"]:
            ming = self._kuang_ming_zhong()
            if ming != self._xuan_duo:
                # 只有"框到的块变了"才重设选中并通知外面：每挪一个像素都
                # 去刷一遍字幕列表没必要，那一批本来就是同一次选择
                self._xuan_duo = list(ming)
                self._xuan_zhong = ming[-1] if ming else -1
                self._miao_dian = ming[-1] if ming else None
                self.kuang_xuan.emit(list(ming))
        self.update()

    def _kuang_ju(self):
        """选框的像素矩形（起点终点对角，不分成谁大谁小）"""
        k = self._kuang
        if k is None:
            return QtCore.QRect()
        return QtCore.QRect(
            QtCore.QPoint(min(k["x0"], k["x1"]), min(k["y0"], k["y1"])),
            QtCore.QPoint(max(k["x0"], k["x1"]), max(k["y0"], k["y1"])),
        )

    def _kuang_ming_zhong(self):
        """框套住了哪几块（按屏幕上块的像素矩形相交算，跨行也能框）"""
        if self._kuang is None or not self._zimu:
            return []
        ju = self._kuang_ju()
        qu = self._guidao_qu()
        tiao = self._zimu_qu(qu)
        hang = self._zimu_hang()
        hangshu = self._zimu_hang_shu()
        ming = []
        for xu in range(len(self._zimu)):
            hang_qu = self._zimu_hang_kuang(tiao, hang[xu], hangshu)
            qi, zhi, _w = self._zimu[xu]
            x1 = self._ms_to_x(qi)
            x2 = self._ms_to_x(zhi)
            if x2 < x1:
                x1, x2 = x2, x1
            if x2 - x1 < 3:
                x2 = x1 + 3
            kuai = QtCore.QRect(
                x1, hang_qu.top(), max(1, x2 - x1), hang_qu.height()
            )
            if ju.intersects(kuai):
                ming.append(xu)
        return ming

    def _kuang_fang(self):
        """松手：框选结果按拖动中实时选好的定下（没拖过 = 就是点了一下空白）"""
        if self._kuang is None:
            return
        dong = self._kuang["dong"]
        self._kuang = None
        self.setCursor(Qt.PointingHandCursor)
        if dong:
            return          # 选谁在拖动中已经实时选好了，这儿不再动
        # 没拖（就是点了一下空白）：清空选中
        if self._xuan_duo or self._xuan_zhong >= 0:
            self._xuan_ze_dan(-1)
            self.xuan_zhong.emit(-1)
        self.update()

    def contextMenuEvent(self, event):
        """右键字幕块：合并 / 删除（只动选中的那几块）"""
        if self._shichang <= 0 or not self._zimu:
            return super().contextMenuEvent(event)
        xu, _buwei = self._zimu_zai_dian(event.pos().x(), event.pos().y())
        if xu < 0:
            return super().contextMenuEvent(event)
        if xu not in self._xuan_duo:
            # 右键点在一块没选中的块上：先选中它，再弹菜单
            self._xuan_ze_dan(xu)
            self.xuan_zhong.emit(xu)
            self.update()
        xuan = self.xuan_zhong_liebiao()

        cai = QtWidgets.QMenu(self)
        biao = cai.addAction(f"操作 {len(xuan)} 段字幕")
        biao.setEnabled(False)
        cai.addSeparator()
        hang = self._zimu_hang()
        a_shang = cai.addAction("上移一行")
        a_shang.setEnabled(any(hang[xu] > 0 for xu in xuan))
        a_xia = cai.addAction("下移一行（到底会再开一行）")
        a_xia.setEnabled(
            all(hang[xu] < self.ZIMU_HANG_ZUIDA - 1 for xu in xuan)
        )
        cai.addSeparator()
        a_he = cai.addAction("合并字幕块（Ctrl+M）")
        a_he.setEnabled(len(xuan) >= 2)
        cai.addSeparator()
        a_shan = cai.addAction(
            "删除字幕块（Delete）" if len(xuan) <= 1
            else f"删除这 {len(xuan)} 段字幕块（Delete）"
        )
        dian = cai.exec_(event.globalPos())
        if dian is a_shang:
            self._yi_zimu_hang(-1)
        elif dian is a_xia:
            self._yi_zimu_hang(1)
        elif dian is a_he:
            self.hebing_zimu.emit(xuan)
        elif dian is a_shan:
            self.shanchu_zimu.emit(xuan)

    def leaveEvent(self, event):
        self._xuanfu_x = None
        self._xuanfu_zimu = None
        gai = self._fen_ge_liang or self._bo_shou_liang
        self._fen_ge_liang = False
        self._bo_shou_liang = False
        if self._zimu_bian_yu is not None:
            self._zimu_bian_yu = None
            gai = True
        if gai:
            self.setCursor(Qt.PointingHandCursor)
        self.update()
        return super().leaveEvent(event)

    def wheelEvent(self, event):
        """滚轮：左右平移整条时间轴，看前面 / 后面的段落用

        往下滚 = 往右看（往后走），往上滚 = 往左看（往前走）。
        一格滚轮挪视野的 1/10 左右，按住滚轮连续滚就是快速拖。
        没放大时（整条视频都在视野里）本来就没得挪，什么都不做。
        """
        if self._shichang <= 0 or self._shi_shangxian() <= 0:
            event.accept()
            return
        jiao = event.angleDelta().y()
        if jiao == 0:
            jiao = event.angleDelta().x()
        if jiao == 0:
            event.accept()
            return
        # 按滚动量成比例地挪（鼠标一格是 120；触摸板一次给一点点，
        # 按比例算就不会一划拉就飞出去）
        bu = int(round(self._keshi_ms() * GUNDONG_BU_BI * abs(jiao) / 120.0))
        bu = max(1, bu)
        self._shi_ms += bu if jiao < 0 else -bu
        self._qia_zheng()
        self._huatu_huancun = None
        self.shitu_bian.emit()
        self.update()
        event.accept()


# ---------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------
class _JiazaiMoxingXiancheng(QtCore.QThread):
    """后台加载检测 / OCR 模型

    模型初始化要好几秒，搁主线程点「开始」整个界面会冻住，
    所以丢后台跑，加载完把结果发回来再真正开始。
    """

    jiazai_wan = pyqtSignal(object, object, object, object, str)
    # 检测模型, OCR 模型, 类别 id, 类别名, 错误说明（空串 = 成功）

    def __init__(self, zhu, canyu, parent=None):
        super().__init__(parent)
        self.zhu = zhu                # 主窗口：借它的 _jiazai_moxing
        self.canyu = canyu
        self.quxiao = False

    def run(self):
        moxing = ocr_moxing = None
        leibie_id = []
        leibie_ming = {}
        cuowu = ""
        try:
            moxing, ocr_moxing, leibie_id, leibie_ming, cuowu = (
                self.zhu._jiazai_moxing(self.canyu)
            )
        except Exception as yi_chang:  # noqa
            logger.error(f"视频工作台：加载模型出错 {yi_chang}")
            cuowu = f"加载模型出错：{yi_chang}"
        self.jiazai_wan.emit(moxing, ocr_moxing, leibie_id, leibie_ming,
                             cuowu)


class VideoWorkDialog(QtWidgets.QDialog):
    """视频工作台窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("YsgVideoWork")
        self.setWindowTitle("视频工作台")
        # 标题栏只留 最小化 / 最大化 / 关闭，去掉那个问号
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        # 界面记忆（窗口位置 / 分栏 / 标签页 / 时间轴缩放）跟「谁在下面」存同一个地方：
        # 软件根目录的 gongzuotai.ini
        self._weizhi_peizhi = buju_qsettings()
        # 上次关窗时的窗口位置 + 大小：存过就还原，头一回给 1280x800
        if self._weizhi_peizhi.value(JIEMIAN_ZHENGGE_JIAN):
            self.restoreGeometry(self._weizhi_peizhi.value(JIEMIAN_ZHENGGE_JIAN))
        else:
            self.resize(1280, 800)
        self.setStyleSheet(_yangshi_quanju())

        self.lujing = ""
        self.fps = 0.0
        self.shichang_ms = 0
        self.zong_zhen = 0
        self.zhen_kuan = 0
        self.zhen_gao = 0
        self.beisu = MO_REN_BEISU
        self.zhengzai_bofang = False
        self.duqu = None
        self.bo_xiancheng = None
        # 播放内核：解码 / 声音 / 定位 / 变速 / 时钟全归 mpv
        self.mpv = MpvBofang(self)
        self._daowei_bao = False     # 结尾这件事只报一次
        self.tuili_qu = None         # 跟随播放做推理时那一路取帧（只喂模型）

        # 推理 / 硬字幕提取
        self.chuliqi = None          # ZhenChuli：单帧处理
        self.yunsuan = None          # 正在跑的后台线程
        self.yunsuan_moshi = ""      # "gensui" 跟随播放 / "saomiao" 全片扫描
        self._jiazai_xc = None       # 正在后台加载模型的线程
        self._jiazai_quxiao = False  # 加载期间点了停止 / 关了窗口
        self._kaishi_paofa = ""      # 这次开始选的是哪种跑法
        self._kaishi_canyu = None    # 这次开始的参数快照（等模型加载完用）
        self._moxing_huancun = {}    # 加载过的模型留着复用：key=(用途, YAML 绝对路径)
        self._bofang_ms = 0          # 播放头当前毫秒（画面下方字幕条用）
        # 方向键微调：按一下走一帧，按住不放越走越快
        self._shangci_fangxiang = 0      # 上一次按的是哪个方向（+1 右 / -1 左 / 0 刚起步）
        self._lianxu_cishu = 0           # 这一串里连着按了多少下（决定快慢）
        self._shangci_shike = 0.0        # 上一次按下的时刻（隔得久就当新的一串）
        self._zou_dao_ms = 0.0           # 这一串已经走到哪儿了（下一次从这儿接着走）
        # 定位守卫：值 = (要定位到的帧号, 请求时刻)。
        # 定位是后台异步做的，刚按方向键时取帧线程手里可能还攥着上一帧，
        # 那一帧一回来就把播放头往回拽 —— 于是播放头在"新位置"和"旧位置"
        # 之间来回蹦。定着位的时候不认旧帧的回写就不会抖。
        self._mubiao_zhen = None
        # 上次已经让读取线程跳过去的帧号：同一个帧号不重复跳（拖播放头时省事）
        self._shang_ci_tiaozheng_zhen = None
        # 鼠标是不是正按着拖播放头（时间轴刻度尺 / 进度条）
        self._tuo_bt_zhong = False
        # 拖动期间攒着的最新帧号：等节拍到点才发给 mpv（见 _fa_tuo_dingwei）
        self._tuo_dingwei_zhen = None
        # 「只播当前字幕块」要播到哪停：值 = 块尾毫秒；None = 一路播下去
        self._bo_dao_ms = None
        self._zimu_tiao_wen = None   # 字幕条现在显示的是哪句话（避免每帧重设）
        # 撤销 / 重做（照 AEG）：一步存一整版字幕的快照，Ctrl+Z 往回退、Ctrl+Y 再往前。
        # 栈里的元素 = (说明, 快照, 动作类别, 时刻)；快照 = (字幕表, 字段表, 选中行)。
        self._chexiao_zhan = []
        self._chongzuo_zhan = []
        self._chexiao_zhong = False  # 正在撤销 / 重做：这期间的写盘别再重复记账
        self.zimu_wenjian = ""       # 当前字幕文件（打开 / 另存过就是它）
        self.zimu_geshi = ""         # "srt" / "ass"，决定存回去用哪种格式
        # 自动保存 / 自动备份（照 AEG 那套）：
        #   备份 = 动一份字幕之前，先把磁盘上的原件抄一份，同名覆盖、只留一份
        #   自动保存 = 定时存一份带时间戳的，一次一份、从不覆盖，攒成历史版本
        self._zidong_beifen_guo = set()   # 这一趟已经备份过的文件（同一个只备份一次）
        self._zidong_gai_dong = False     # 字幕改过、还没自动存过
        self._zidong_shange = None        # 上次自动保存时字幕长什么样
        self._zidong_jishi = QtCore.QTimer(self)
        self._zidong_jishi.setInterval(ZIDONG_BAOCUN_MIAO * 1000)
        self._zidong_jishi.timeout.connect(self._zidong_baocun)
        self._zidong_jishi.start()
        # 打开的是 ASS 时留一份原结构（骨架 / 样式表 / 每行字段）：保存按它写，
        # 只有时间和条数会变，样式之类原样保留
        self._ass_yuan = None
        # 画面里叠字幕：样式表缓存 + 每条字幕配的字段表 / ASS 原文缓存
        self._zimu_hua_ji = None      # (原结构, 基准分辨率, 样式表) 读一次存着
        self._zimu_pei_ji = []        # 每条配的（样式, 说话人, ASS 原文, 字段）
        self._zimu_fu_ji = []         # 每条的字段表（原 ASS 的 + 编辑区上方改过的）
        self._fu_wai = {}             # 外面直接指定字段的行（粘贴进来的那种）
                                      # 键 = tuple(起, 止, 文字)
        # 画面区高度：上半栏按画面比例贴着来，上下不留黑边（省出来的给时间轴）。
        # _shang_ci_he 存上次算过的（画面区宽, 画面原始尺寸），变了才主动重调；
        # 改分栏这个动作一律延到下一轮事件循环，不在 resize 里直接动（会打架）。
        self._shang_ci_he = None
        self._he_dai = None          # 待改成的上半栏高度
        self._he_sile = None         # (想改成的高, 当时的高)：动不了的组合，别再试
        self._he_jishi = QtCore.QTimer(self)
        self._he_jishi.setSingleShot(True)
        self._he_jishi.timeout.connect(self._shishi_he_gao)

        # 定时处理 mpv 事件及片尾检测；播放头等新视频帧实际渲染后再更新。
        self.xianshi_dingshi = QtCore.QTimer(self)
        self.xianshi_dingshi.setInterval(16)
        self.xianshi_dingshi.timeout.connect(self._qu_zhen_xianshi)

        # 拖播放头：按 TUO_SEEK_JIAN_GE_MS 的节拍把最新位置交给定位队列
        # —— 一次只让一发 seek 在飞，落地了才发下一发（见 _fa_tuo_dingwei）
        self.tuo_dingshi = QtCore.QTimer(self)
        self.tuo_dingshi.setInterval(TUO_SEEK_JIAN_GE_MS)
        self.tuo_dingshi.timeout.connect(self._fa_tuo_dingwei)

        # 声音也归 mpv：这里留个代理，外面还是照老样子调 setVolume / play /
        # pause / stop / setPlaybackRate，实际都转到 mpv 上
        self.yinpin = None
        if self.mpv.h is not None:
            self.yinpin = _YinpinDaiLi(self.mpv)
            self.yinpin.setVolume(0 if JINGYIN else int(MO_REN_YINLIANG))

        # 「时间轴」和「字幕列表」谁在下面（右键点这两块能切）。
        # 上次切过就按上次的来，头一回默认时间轴在下（打轴的摆法）。
        self._liebiao_zai_xia = (
            str(self._weizhi_peizhi.value(WEIZHI_PEIZHI_JIAN, "shijianzhou"))
            == "liebiao"
        )
        # 上下大分栏你亲手拖过没有：拖过就听你的，不再自动贴合画面比例
        self._shangxia_duoguo = str(
            self._weizhi_peizhi.value(JIEMIAN_SHANGXIA_DUO, "0")
        ) in ("1", "true", "True")
        # 标签页里面的分栏：那一页藏着的时候摆不准，先记着，等它露出来再摆
        self._ye_bili = {}

        self._jian_ui()

        # 两个界面开关（照 AEG / ARC）：
        #   选中字幕时画面跟着跳 = 在字幕列表里点一条，画面要不要跟过去
        #   时间轴实时滚动 = 播放时视图跟着播放头滚，还是播放头自己在轴上走
        self._gen_tiao = self._du_kaiguan(JIEMIAN_GEN_TIAO, True)
        self._shishi_gun = self._du_kaiguan(JIEMIAN_SHISHI_GUN, False)
        self.shijianzhou.shezhi_shishi_gun(self._shishi_gun)
        for ming in ("bianji_mianban", "zimu_mianban"):
            mian = getattr(self, ming, None)
            if mian is not None and hasattr(mian, "shezhi_zidong_tiao"):
                mian.shezhi_zidong_tiao(self._gen_tiao)
        self.bianji_mianban.gongjulan.shezhi_kaiguan(
            self._gen_tiao, self._shishi_gun
        )

        # 你亲手拖过上下大分栏 -> 记下来，以后不再自动贴合画面比例（听你的）
        self.shang_xia.splitterMoved.connect(self._shangxia_dong_le)

        # 拖 .srt / .ass 进这个窗口就直接载入（见 dragEnterEvent / dropEvent）
        self.setAcceptDrops(True)
        # 编辑框自己默认吃文件拖放（会把路径当文字插进去），关掉，
        # 这样拖到它身上也照样能落到窗口上载入字幕
        self.bianji_mianban.kuang.setAcceptDrops(False)

        # 按钮 / 勾选框 / 单选框 / 下拉框一律不吃键盘焦点。
        # 焦点一旦停在它们身上，空格就被它们自己吃掉了（按钮被当成"点一下"），
        # 根本冒泡不到本窗口，空格自然不是播放 / 暂停。设成 NoFocus 之后
        # 鼠标点击照常能用，键盘事件会冒泡上来交给下面的 keyPressEvent。
        # 输入框、数值框不在此列（要在里面打字 / 调数值）。
        for w in self.findChildren(QtWidgets.QAbstractButton):
            w.setFocusPolicy(Qt.NoFocus)
        for w in self.findChildren(QtWidgets.QComboBox):
            w.setFocusPolicy(Qt.NoFocus)
        # 滑块（音量、时间轴缩放）同理：焦点落在上面，方向键会被它拿去
        # 调滑块，轮不到微调帧。鼠标拖它照样好使。
        for w in self.findChildren(QtWidgets.QAbstractSlider):
            w.setFocusPolicy(Qt.NoFocus)

        # 主界面菜单那批 action 都注册成"应用程序级"快捷键（焦点在子窗口上也响应），
        # 视频工作台里按 Q/W/E/R 会跟主界面一起抢：Qt 打 Ambiguous shortcut overload
        # 警告，两边还可能同时响应。打开本窗口期间把它们临时降成"窗口级"，
        # 主窗口不激活就不响应；关窗口再原样还回去。
        self._beiping_kuangjie = []
        self._biping_zhujiemian_kuangjie()

        # 再补一道保险：空格快捷键。
        # QShortcut 的优先级高于控件的按键处理，万一焦点还是落在上面那些
        # 控件上，空格照样走播放 / 暂停。输入框里空格还是空格。
        self.kongge_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Space), self
        )
        self.kongge_jian.setContext(Qt.WindowShortcut)
        self.kongge_jian.activated.connect(self._kongge_an_xia)

        # 字幕编辑的时候空格得是空格，播放改用两个键：
        # Tab = 播放 / 暂停；` 或 ~ = 只播当前这一条字幕块（播完就停，再按重播）。
        # 这两个键在视频工作台里是全局的：焦点在字幕编辑框里也一样响应，
        # 而且绝不许落进框里变成 Tab 缩进 / 波浪号。所以不走 QShortcut ——
        # QPlainTextEdit 会先把 Tab、波浪号认成"我自己要用的字"，快捷键轮不上。
        # 改成在应用级拦按键，见 eventFilter。
        _yingyong = QtWidgets.QApplication.instance()
        if _yingyong is not None:
            _yingyong.installEventFilter(self)
            self._yingyong_lan_jian = _yingyong
        else:
            self._yingyong_lan_jian = None

        # 左右方向键 = 后退 / 前进一帧，按住不放越走越快。
        # 同样做成整窗口快捷键：焦点在时间轴、按钮、勾选框上都能用；
        # 焦点在输入框 / 数值框里时不抢（它们自己要拿方向键移光标、调数值）。
        self.zuo_fangxiang_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Left), self
        )
        self.zuo_fangxiang_jian.setContext(Qt.WindowShortcut)
        self.zuo_fangxiang_jian.setAutoRepeat(True)
        self.zuo_fangxiang_jian.activated.connect(
            lambda: self._fangxiangjian_tiaobu(-1)
        )
        self.you_fangxiang_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Right), self
        )
        self.you_fangxiang_jian.setContext(Qt.WindowShortcut)
        self.you_fangxiang_jian.setAutoRepeat(True)
        self.you_fangxiang_jian.activated.connect(
            lambda: self._fangxiangjian_tiaobu(1)
        )

        # Q/W/E/R + A/D = 打轴（只认单选的字幕块），Aegisub 那套：
        # Q 播放头跳到块开头、W 跳到块结尾、E 把块开头设到播放头、R 把块结尾设到播放头，
        # A 把块尾紧贴下一块的块头、D 把块头紧贴上一块的块尾。
        # 同样做成本窗口的快捷键（打开本窗口时主界面那批应用级快捷键已被降级，抢不走）。
        self.dazhou_jian = {}
        for jian_ming, dongzuo in (
            (Qt.Key_Q, "Q"), (Qt.Key_W, "W"),
            (Qt.Key_E, "E"), (Qt.Key_R, "R"),
            (Qt.Key_A, "A"), (Qt.Key_D, "D"),
        ):
            jian = QtWidgets.QShortcut(QtGui.QKeySequence(jian_ming), self)
            jian.setContext(Qt.WindowShortcut)
            jian.setAutoRepeat(True)
            jian.activated.connect(
                lambda d=dongzuo: self._dazhou_an(d)
            )
            self.dazhou_jian[dongzuo] = jian

        # Ctrl+S = 保存字幕：有打开过 / 另存过的文件就直接存回去，没有就弹另存为
        self.cun_zimu_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+S"), self
        )
        self.cun_zimu_jian.setContext(Qt.WindowShortcut)
        self.cun_zimu_jian.activated.connect(self._cun_zimu)

        # Delete = 删掉时间轴上选中的字幕块；Ctrl+M = 把选中的并成一条。
        # 同样是整窗口快捷键，焦点在时间轴 / 列表上都能用；
        # 焦点在输入框里时不抢（那儿 Delete 就该删字符）。
        self.shanchu_zimu_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Delete), self
        )
        self.shanchu_zimu_jian.setContext(Qt.WindowShortcut)
        self.shanchu_zimu_jian.activated.connect(self._shanchu_xuan_zhong_zimu)
        self.hebing_zimu_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+M"), self
        )
        self.hebing_zimu_jian.setContext(Qt.WindowShortcut)
        self.hebing_zimu_jian.activated.connect(self._hebing_xuan_zhong_zimu)

        # S = 切分字幕块：游标（播放头）停着的那条，从游标位置一分为二，
        # 两条文字一模一样，只是时间被切开（硬字幕提取把两句连成一条时用）。
        # 同样是整窗口快捷键，焦点在输入框里时不抢（那儿 S 就该是字母 s）。
        self.qiefen_zimu_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_S), self
        )
        self.qiefen_zimu_jian.setContext(Qt.WindowShortcut)
        self.qiefen_zimu_jian.activated.connect(self._qiefen_zimu_kuai)

        # Alt+S = 开关注释：把当前这条（列表里多选就这一批）在"注释 / 普通"
        # 之间翻一下，跟用鼠标勾那个「注释」勾选框一个效果。同样是整窗口
        # 快捷键，只在本窗口生效；输入框里 Alt+S 本来也没别的用处，不抢。
        self.zhushi_zimu_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence("Alt+S"), self
        )
        self.zhushi_zimu_jian.setContext(Qt.WindowShortcut)
        self.zhushi_zimu_jian.activated.connect(self._qiehuan_zhushi)

        # F1~F12 = 套槽位（说话人 + 样式）：选中字幕按 F1 就是套「槽位01」，
        # 跟 AEG 那个 Lua 脚本一个手感。槽位在样式编辑器下方「自动化脚本」
        # 里配（存在软件根目录的配置文件里）。同样只在本窗口生效，
        # 主界面没占 F1~F12，不冲突。
        self.caowei_jian = []
        for hao in range(CAOWEI_SHU):
            jian = QtWidgets.QShortcut(
                QtGui.QKeySequence(getattr(Qt, f"Key_F{hao + 1}")), self
            )
            jian.setContext(Qt.WindowShortcut)
            jian.activated.connect(lambda h=hao: self._tao_caowei(h))
            self.caowei_jian.append(jian)

        # Ctrl+R = 选择、Ctrl+F = 查找、Ctrl+H = 替换（照 AEG 那三个窗口）。
        # 都是非模态窗口（show 不是 exec）：开着照样能播放、拖时间轴、改字幕。
        # 同样只在本窗口生效，主界面那批快捷键打开工作台时已经降级了。
        self.sousuo_jian = {}
        for jian_ming, na in (
            ("Ctrl+R", "xuan"),
            ("Ctrl+F", "zhao"),
            ("Ctrl+H", "huan"),
        ):
            jian = QtWidgets.QShortcut(QtGui.QKeySequence(jian_ming), self)
            jian.setContext(Qt.WindowShortcut)
            jian.activated.connect(lambda n=na: self._kai_sousuo(n))
            self.sousuo_jian[na] = jian

        # Ctrl+Z = 撤销、Ctrl+Y = 重做（照 AEG 那套：一步一整版字幕快照）。
        # 焦点在输入框里时让给输入框自己（那儿 Ctrl+Z 该撤文字，不该撤字幕）。
        # 同样只在本窗口生效，主界面那批快捷键打开工作台时已经降级了。
        self.chexiao_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+Z"), self
        )
        self.chexiao_jian.setContext(Qt.WindowShortcut)
        self.chexiao_jian.activated.connect(self._chexiao)
        self.chongzuo_jian = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+Y"), self
        )
        self.chongzuo_jian.setContext(Qt.WindowShortcut)
        self.chongzuo_jian.activated.connect(self._chongzuo)

    # --------------------------------------------------------------
    # 界面
    # --------------------------------------------------------------
    def _jian_ui(self):
        gen = QtWidgets.QVBoxLayout(self)
        gen.setContentsMargins(12, 12, 12, 12)
        gen.setSpacing(0)

        # 上下分栏：上半（画面+参数） / 下半（时间轴 或 字幕列表），中间可拖着改高度
        self.shang_xia = QtWidgets.QSplitter(Qt.Vertical)
        self.shang_xia.setHandleWidth(9)
        self.shang_xia.setChildrenCollapsible(False)

        # 左右分栏：画面 / 参数区，中间可拖着改宽度
        self.zuo_you = QtWidgets.QSplitter(Qt.Horizontal)
        self.zuo_you.setHandleWidth(9)
        self.zuo_you.setChildrenCollapsible(False)
        self.zuo_you.addWidget(self._jian_yulan_mianban())
        self.zuo_you.addWidget(self._jian_canshu_mianban())
        self.zuo_you.setStretchFactor(0, 1)
        self.zuo_you.setStretchFactor(1, 0)

        self.shang_xia.addWidget(self.zuo_you)
        self.shang_xia.addWidget(self._jian_shijianzhou_mianban())
        self.shang_xia.setStretchFactor(0, 1)
        self.shang_xia.setStretchFactor(1, 0)

        # 「时间轴」和「字幕列表」按上次记住的位置摆好（默认时间轴在下），
        # 再给这两块挂上右键菜单，随时能对调
        self._bai_weizhi(chi_cun=False)
        self._gua_youjian_caidan()

        gen.addWidget(self.shang_xia, 1)

        # 分栏比例等窗口真正显示出来再定，不然尺寸还没算好
        QtCore.QTimer.singleShot(0, self._shezhi_chushi_bili)
        self._shuaxin_anniu()

    def _shezhi_chushi_bili(self):
        try:
            # 下面那一栏装的是谁，就按谁给高度：时间轴矮，字幕列表要高些
            self.shang_xia.setSizes(
                [
                    560,
                    ZIMU_LIEBIAO_GAO
                    if self._liebiao_zai_xia
                    else SHIJIANZHOU_GAO + 44,
                ]
            )
            self.zuo_you.setSizes([900, 340])
            if self._liebiao_zai_xia:
                # 列表在下面时，时间轴缩进右栏上方，给它够用的高度
                self.bianji_mianban.setSizes([SHIJIANZHOU_GAO + 44, 220])
        except Exception:  # noqa
            pass
        self._huan_jiemian_bili()       # 上次拖过的分栏位置盖回来
        # 上面刚把比例定死，画面区可能还是矮的（或者一开始就开了视频）：
        # 等这一步的尺寸落到控件上，再按画面比例把上半栏调一遍
        self._shang_ci_he = None
        QtCore.QTimer.singleShot(0, self._he_huamian_gao)

    # --------------------------------------------------------------
    # 「时间轴」和「字幕列表」对调位置
    # --------------------------------------------------------------
    def _bai_weizhi(self, chi_cun=True):
        """按当前模式把「时间轴块」和「字幕列表块」摆到各自该在的位置

        两块都是整块搬（各自带的工具栏 / 标题一起走）。Qt 搬控件只换父级，
        对象本身不动，所以信号连接、选中状态、字体缓存这些全都还在。
        chi_cun=True 时先把两处的尺寸记下来、搬完原样放回去 —— 就是换个位置，
        大小一点都不跟着变。
        """
        ce = self.bianji_mianban             # 右栏那个上下分栏
        lie = ce.lie_biao_kuang              # 字幕列表整块（标题 + 那张表）
        zhou = self.shijianzhou_kuang        # 时间轴整块（含打开/保存/缩放那排）
        shang_chi = self.shang_xia.sizes() if chi_cun else None
        ce_chi = ce.sizes() if chi_cun else None
        if self._liebiao_zai_xia:
            # 做字幕的摆法（照 AEG）：列表铺在窗口下方，时间轴收进右栏上方
            self.shang_xia.insertWidget(1, lie)
            ce.insertWidget(0, zhou)
        else:
            # 打轴 / 调轴的摆法（照 ACRtime）：时间轴在窗口下方，列表回右栏
            self.shang_xia.insertWidget(1, zhou)
            ce.insertWidget(0, lie)
        if chi_cun:
            self.shang_xia.setSizes(shang_chi)
            ce.setSizes(ce_chi)
        # 列表铺在窗口下方时，把「字幕列表 + 条数」那行收掉（照 AEG 顶格显示）
        ce.shezhi_liebiao_dingge(self._liebiao_zai_xia)

    def _qiehuan_weizhi(self):
        """把「时间轴」和「字幕列表」换个位置，并把这次的选择记下来"""
        self._liebiao_zai_xia = not self._liebiao_zai_xia
        self._weizhi_peizhi.setValue(
            WEIZHI_PEIZHI_JIAN,
            "liebiao" if self._liebiao_zai_xia else "shijianzhou",
        )
        self._bai_weizhi()
        self.shezhi_mianban.zhuangtai_shezhi(
            "字幕列表挪到窗口下方（时间轴收进右栏）"
            if self._liebiao_zai_xia
            else "时间轴挪到窗口下方（字幕列表回到右栏）"
        )

    # --------------------------------------------------------------
    # 界面记忆：窗口位置 / 标签页 / 各处分栏 / 时间轴缩放
    # --------------------------------------------------------------
    @staticmethod
    def _bili_wenben(daxiao):
        """分栏尺寸 -> 存进配置的一小段文本（不记的写「-」）"""
        if not daxiao:
            return "-"
        return ",".join(str(int(x)) for x in daxiao)

    @staticmethod
    def _wenben_bili(wenben):
        """配置里那段文本 -> 分栏尺寸（读不出来返回 None，那处就按默认）"""
        wenben = str(wenben or "").strip()
        if not wenben or wenben == "-":
            return None
        try:
            da = [int(x) for x in wenben.split(",")]
        except ValueError:
            return None
        return da if len(da) >= 2 else None

    def _huan_jiemian_bili(self):
        """按上次关窗时记下的分栏位置还原（没存过 / 摆法换过了就什么都不做）

        「画面 ↔ 右栏」「上半 ↔ 下半」在窗口上，直接摆就行；标签页里面那两处
        分栏（列表 ↔ 编辑框、推理设置 ↔ 字幕列表）得等那一页露出来才量得出高度，
        藏着的时候摆不准，所以先记着，等那页第一次显出来再摆。
        """
        duan = str(
            self._weizhi_peizhi.value(JIEMIAN_BILI_JIAN, "") or ""
        ).split(";")
        if len(duan) != 5:
            return
        bai = "liebiao" if self._liebiao_zai_xia else "shijianzhou"
        if duan[0] != bai:
            return              # 摆法跟上次不一样，那套尺寸对不上，按默认来
        for kuang, zhi in ((self.zuo_you, duan[1]), (self.shang_xia, duan[2])):
            da = self._wenben_bili(zhi)
            if da:
                kuang.setSizes(da)
        dang_qian = self.you_tabs.currentIndex()
        for ye, kuang, zhi in (
            (0, self.tiqu_ce, duan[4]),
            (1, self.bianji_mianban, duan[3]),
        ):
            da = self._wenben_bili(zhi)
            if not da:
                continue
            if ye == dang_qian:
                # 等这一轮布局落定再摆，不然页面尺寸还是 0，摆不准
                QtCore.QTimer.singleShot(0, lambda k=kuang, d=da: k.setSizes(d))
            else:
                self._ye_bili[ye] = (kuang, da)

    def _tab_huan_le(self, ye):
        """切到某个标签页：把上次记下的、这页里面的分栏位置放回去"""
        zai = self._ye_bili.pop(int(ye), None)
        if zai:
            QtCore.QTimer.singleShot(0, lambda: zai[0].setSizes(zai[1]))

    def _shangxia_dong_le(self, _wei=0, _hao=0):
        """你亲手拖过上下大分栏：记一笔，以后不再自动贴合画面比例"""
        self._shangxia_duoguo = True

    def _cun_jiemian_buju(self):
        """关窗时把窗口位置、标签页、各处分栏位置、时间轴缩放记下来，下次原样还原"""
        try:
            pei = self._weizhi_peizhi
            pei.setValue(JIEMIAN_ZHENGGE_JIAN, self.saveGeometry())
            pei.setValue(JIEMIAN_TAB_JIAN, int(self.you_tabs.currentIndex()))
            pei.setValue(JIEMIAN_SHANGXIA_DUO, "1" if self._shangxia_duoguo else "0")
            # 记数值框上的倍数：没打开过视频时时间轴内部倍数还是 1，
            # 直接记它会把默认的 ×15 弄丢
            pei.setValue(
                JIEMIAN_SUOFANG_JIAN,
                float(self.shijianzhou_suofang.value()),
            )
            bai = "liebiao" if self._liebiao_zai_xia else "shijianzhou"
            pei.setValue(
                JIEMIAN_BILI_JIAN,
                ";".join((
                    bai,
                    self._bili_wenben(self.zuo_you.sizes()),
                    # 上下大分栏只在你亲手拖过时才记：没拖过的话那个高度是
                    # 按画面比例算出来的，下次开视频它自己会重算
                    self._bili_wenben(self.shang_xia.sizes())
                    if self._shangxia_duoguo
                    else "-",
                    self._bili_wenben(self.bianji_mianban.sizes()),
                    self._bili_wenben(self.tiqu_ce.sizes()),
                )),
            )
            pei.sync()
        except Exception as cuowu:  # noqa
            logger.warning(f"视频工作台：记界面摆放失败 {cuowu}")

    def _jian_hang_caidan(self):
        """字幕列表右键里那批「整行编辑」（照 AEG 的网格右键菜单）

        action 只建一份长住：右键菜单挂一遍，字幕列表控件上也挂一遍 —— 挂
        控件是为了让快捷键只在焦点落在列表上时才生效（挂窗口会跟编辑框自己
        的 Ctrl+C / Ctrl+V 抢键），所以在列表里直接按也一样管用。
        """
        biao = self.bianji_mianban.biao
        ding = [
            ("插入(之前)", "Alt+J", lambda _=False: self._hang_charu(zai_qian=True)),
            ("插入(之后)", "", lambda _=False: self._hang_charu(zai_qian=False)),
            ("以视频时间插入(之前)", "",
             lambda _=False: self._hang_charu_shipin(zai_qian=True)),
            ("以视频时间插入(之后)", "Ctrl+K",
             lambda _=False: self._hang_charu_shipin(zai_qian=False)),
            None,
            ("重复行", "Ctrl+L", lambda _=False: self._hang_fuzhi()),
            ("以当前帧前分割行", "",
             lambda _=False: self._hang_qiege(xiang_hou=False)),
            ("以当前帧后分割行", "Ctrl+D",
             lambda _=False: self._hang_qiege(xiang_hou=True)),
            None,
            ("互换行", "", lambda _=False: self._hang_huhuan()),
            ("合并(连接)", "", lambda _=False: self._shoudao_zimu_hebing()),
            ("合并(保留首行)", "",
             lambda _=False: self._shoudao_zimu_hebing(liu_shou_hang=True)),
            None,
            ("使时间连续(更改开始时间)", "",
             lambda _=False: self._hang_shijian_lianxu(gai_qi=True)),
            ("使时间连续(更改结束时间)", "",
             lambda _=False: self._hang_shijian_lianxu(gai_qi=False)),
            ("重组行", "", lambda _=False: self._hang_chongzu()),
            None,
            ("剪切行", "Ctrl+X", lambda _=False: self._hang_jianqie(shan=True)),
            ("复制行", "Ctrl+C", lambda _=False: self._hang_jianqie(shan=False)),
            ("粘贴行", "Ctrl+V", lambda _=False: self._hang_zhantie()),
            None,
            ("删除行", "Ctrl+Delete", lambda _=False: self._shoudao_zimu_shanchu()),
        ]
        self._hang_actions = []
        for xiang in ding:
            if xiang is None:
                self._hang_actions.append(None)     # None = 菜单里那条分隔线
                continue
            wenzi, jian, dong = xiang
            a = QtWidgets.QAction(wenzi, self)
            if jian:
                a.setShortcut(jian)
                a.setShortcutContext(Qt.WidgetShortcut)
            a.triggered.connect(dong)
            biao.addAction(a)
            self._hang_actions.append(a)

    def _gua_youjian_caidan(self):
        """给这两块挂右键菜单

        字幕列表：批量加「」+ 位置切换
        时间轴：只有位置切换
        里层那块（列表的表、时间轴的画布）也要挂：点在它们身上时，
        事件不会冒到外框来，只挂外框的话点在表上没反应。
        """
        for w in (self.bianji_mianban.lie_biao_kuang,
                  self.bianji_mianban.biao):
            w.setContextMenuPolicy(Qt.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, x=w: self._tan_liebiao_caidan(x, pos)
            )
        for w in (self.shijianzhou_kuang, self.shijianzhou):
            w.setContextMenuPolicy(Qt.CustomContextMenu)
            w.customContextMenuRequested.connect(self._tan_weizhi_caidan)
        self._jian_hang_caidan()

    def _tan_liebiao_caidan(self, kuang, pos):
        """字幕列表右键：批量加「」+ 位置切换

        在行上点的时候：这一行要是已经在选中里，就对整批选中的下手（Ctrl+A
        全选后右键 = 一次全处理）；要是点在没选中过的行上，就只对点到的这一
        行下手，界面上已经选好的那几行不动。

        「说话人+样式」不放这儿：F1~F12 直接套，配置在样式编辑器里，
        右键再挂一份是多余的。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        mu = [int(x) for x in self.bianji_mianban.xuan_zhong_hang()]
        if kuang is self.bianji_mianban.biao and zimu:
            hang = self.bianji_mianban.biao.indexAt(pos).row()
            if 0 <= hang < len(zimu) and hang not in mu:
                mu = [hang]
        mu = [i for i in mu if 0 <= i < len(zimu)]
        # 这轮右键要处理哪几行：菜单里的动作触发时来这儿取（快捷键触发时
        # 这儿是 None，回去拿列表里真正选中的那几行）
        self._youjian_mu = list(mu)

        caidan = QtWidgets.QMenu(self)
        if mu:
            kuo = caidan.addAction("【-批量添加方括号】")
            kuo.triggered.connect(
                lambda _=False, h=list(mu): self._zhixing_kuohao(h)
            )
        else:
            wu = caidan.addAction("先选中要处理的字幕")
            wu.setEnabled(False)
        caidan.addSeparator()
        # AEG 那套整行编辑（插入 / 重复 / 分割 / 合并 / 时间 / 剪贴板 …）
        for a in self._hang_actions:
            if a is None:
                caidan.addSeparator()
            else:
                caidan.addAction(a)
        caidan.addSeparator()
        zhou, lie = self._tian_weizhi_caidan(caidan)
        dian = caidan.exec_(QtGui.QCursor.pos())
        self._youjian_mu = None          # 菜单关了，之后只认列表里选中的
        self._caidan_dian_weizhi(dian, zhou, lie)

    def _caidan_dian_weizhi(self, dian, zhou, lie):
        """点了菜单里的「谁放在下方」才翻位置

        必须确认点中的是这两项中的一项：菜单里还有别的项（批量加「」那些），
        不排除的话，点它们也会走到这儿来，位置就跟着乱翻。
        """
        if dian is None or (dian is not zhou and dian is not lie):
            return
        if bool(dian is lie) != self._liebiao_zai_xia:
            self._qiehuan_weizhi()

    def _tian_weizhi_caidan(self, caidan):
        """往菜单末尾加「谁放在下方」那两项，返回这两个 action"""
        zhou = caidan.addAction("时间轴放在下方")
        lie = caidan.addAction("字幕列表放在下方")
        for a in (zhou, lie):
            a.setCheckable(True)
        zhou.setChecked(not self._liebiao_zai_xia)
        lie.setChecked(self._liebiao_zai_xia)
        zu = QtWidgets.QActionGroup(caidan)
        zu.addAction(zhou)
        zu.addAction(lie)
        return zhou, lie

    def _tan_weizhi_caidan(self, _dian=None):
        """时间轴右键弹出的那个小菜单：选谁放到下方"""
        caidan = QtWidgets.QMenu(self)
        zhou, lie = self._tian_weizhi_caidan(caidan)
        dian = caidan.exec_(QtGui.QCursor.pos())
        self._caidan_dian_weizhi(dian, zhou, lie)

    def _jian_yulan_mianban(self):
        mianban = QtWidgets.QFrame()
        mianban.setObjectName("YsgPreview")
        self._yulan = mianban        # 画面这一栏整块（算上半栏该多高要用它）
        bu = QtWidgets.QVBoxLayout(mianban)
        bu.setContentsMargins(0, 0, 0, 0)
        bu.setSpacing(0)

        # ---- 画面（顶上不留标题栏，直接顶格） ----
        self.huamian = HuamianQu()
        self.huamian.shezhi_mpv(self.mpv)      # 视频由 mpv 渲染进这块画布
        # mpv 每真正画出一帧，播放头和字幕才一起跟到那一帧；这样时间轴不会
        # 按 mpv 的时钟先走，而画面还停在上一帧。
        self.huamian.shezhi_huan_zhen_hui(self._mpv_huan_zhen)
        self.huamian.chicun_bian.connect(self._tongbu_suo_dao)
        self.huamian.chicun_bian.connect(self._he_huamian_gao)
        bu.addWidget(self.huamian, 1)

        # ---- 画面下方：当前这一段的字幕（跟画面对照用）；底栏有个开关能收起来 ----
        self.zimu_tiao = QtWidgets.QLabel("")
        self.zimu_tiao.setObjectName("YsgZimuTiao")
        self.zimu_tiao.setAlignment(Qt.AlignCenter)
        self.zimu_tiao.setFixedHeight(ZIMU_TIAO_GAO)
        self.zimu_tiao.setTextFormat(Qt.PlainText)
        self.zimu_tiao.setVisible(self._du_zimu_tiao_kai())
        bu.addWidget(self.zimu_tiao, 0)

        # ---- 底栏（进度条 + 播放控制） ----
        di = QtWidgets.QFrame()
        di.setObjectName("YsgPreviewFooter")
        di_zhong = QtWidgets.QVBoxLayout(di)
        di_zhong.setContentsMargins(8, 2, 8, 2)
        di_zhong.setSpacing(0)

        # 上排：整条片子的进度（刻度 + 指针），点 / 拖定位
        self.jindu_tiao = ShipinJinduTiao()
        self.jindu_tiao.dingwei.connect(self._shoudao_tiaozheng)
        self.jindu_tiao.tuo_kaishi.connect(self._tuo_bofangtou_kaishi)
        self.jindu_tiao.tuo_jieshu.connect(self._tuo_bofangtou_wancheng)
        di_zhong.addWidget(self.jindu_tiao)

        # 下排：小按钮 + 当前时间 - 帧号（照 Aegisub：按钮在最左，时间紧跟其后）
        xia_pai = QtWidgets.QWidget()
        di_bu = QtWidgets.QHBoxLayout(xia_pai)
        di_bu.setContentsMargins(0, 0, 0, 0)
        di_bu.setSpacing(2)

        self.a_tui1miao = _tubiao_anniu(
            B.SP_MediaSeekBackward, "后退 1 秒", lambda: self._tiaobu(-1000)
        )
        self.a_shang_yizhen = _tubiao_anniu(
            B.SP_MediaSkipBackward, "上一帧", lambda: self._tiaobu_zhen(-1)
        )
        self.a_bofang = _tubiao_anniu(
            B.SP_MediaPlay, "播放 / 暂停", self._qiehuan_bofang
        )
        self.a_xia_yizhen = _tubiao_anniu(
            B.SP_MediaSkipForward, "下一帧", lambda: self._tiaobu_zhen(1)
        )
        self.a_jin1miao = _tubiao_anniu(
            B.SP_MediaSeekForward, "前进 1 秒", lambda: self._tiaobu(1000)
        )
        self.a_bofang_dangqian = _tubiao_anniu(
            "button_playline_16.png",     # Aegisub 自带的「播放当前行」图标
            "播放当前行（快捷键 ~）：从这条字幕的头上播到末尾就停",
            self._bofang_dangqian_kuai,
        )
        for a in (
            self.a_tui1miao,
            self.a_shang_yizhen,
            self.a_bofang,
            self.a_xia_yizhen,
            self.a_jin1miao,
            self.a_bofang_dangqian,
        ):
            di_bu.addWidget(a)

        di_bu.addSpacing(8)
        self.shijian_dangqian = QtWidgets.QLabel("0:00:00.000 - 0")
        self.shijian_dangqian.setObjectName("YsgTimeNow")
        self.shijian_dangqian.setToolTip("当前时间 - 当前帧号")
        di_bu.addWidget(self.shijian_dangqian)
        di_bu.addStretch(1)

        self.a_dakai_zimu = QtWidgets.QPushButton("打开字幕")
        self.a_dakai_zimu.setStyleSheet(_ys_ci_anniu_xiao())
        self.a_dakai_zimu.setCursor(Qt.PointingHandCursor)
        self.a_dakai_zimu.setToolTip(
            "载入一份 SRT / ASS 字幕到时间轴上（打轴、调轴、看轴）"
        )
        self.a_dakai_zimu.clicked.connect(self._dakai_zimu_wenjian)
        di_bu.addWidget(self.a_dakai_zimu)

        self.a_cun_zimu = QtWidgets.QPushButton("保存字幕")
        self.a_cun_zimu.setStyleSheet(_ys_ci_anniu_xiao())
        self.a_cun_zimu.setCursor(Qt.PointingHandCursor)
        self.a_cun_zimu.setToolTip(
            "另存为（弹框选路径和格式）；Ctrl+S 直接存回打开过的那个文件"
        )
        self.a_cun_zimu.clicked.connect(self._ling_cun_zimu)
        di_bu.addWidget(self.a_cun_zimu)
        di_bu.addSpacing(6)

        self.a_zimu_tiao = QtWidgets.QCheckBox("字幕条")
        self.a_zimu_tiao.setToolTip("显示 / 隐藏画面下方那条独立字幕")
        self.a_zimu_tiao.setChecked(self._du_zimu_tiao_kai())
        self.a_zimu_tiao.toggled.connect(self._qiehuan_zimu_tiao)
        di_bu.addWidget(self.a_zimu_tiao)
        di_bu.addSpacing(6)

        di_bu.addWidget(QtWidgets.QLabel("时间轴缩放"))
        self.shijianzhou_suofang = QtWidgets.QSpinBox()
        self.shijianzhou_suofang.setRange(
            int(SUOFANG_ZUI_XIAO), int(SUOFANG_ZUIDA)
        )
        self.shijianzhou_suofang.blockSignals(True)
        self.shijianzhou_suofang.setValue(int(round(self._suofang_qishi())))
        self.shijianzhou_suofang.blockSignals(False)
        self.shijianzhou_suofang.setStyleSheet(_ys_xiao_xiang())
        self.shijianzhou_suofang.setFixedWidth(52)
        self.shijianzhou_suofang.setToolTip(
            f"时间轴放大倍数（{int(SUOFANG_ZUI_XIAO)} ~ {int(SUOFANG_ZUIDA)}）：\n"
            "直接敲数字，或者点上下箭头 / 把鼠标放框上滚轮调"
        )
        self.shijianzhou_suofang.valueChanged.connect(self._shijianzhou_suofang)
        di_bu.addWidget(self.shijianzhou_suofang)
        di_bu.addSpacing(6)

        self.beisu_kuang = QtWidgets.QComboBox()
        for b in BEISU_LIEBIAO:
            self.beisu_kuang.addItem(f"{b:g}", b)
        # 默认必须停在 MO_REN_BEISU，否则会按列表第一项（0.5x）慢速播放
        if MO_REN_BEISU in BEISU_LIEBIAO:
            self.beisu_kuang.setCurrentIndex(BEISU_LIEBIAO.index(MO_REN_BEISU))
        self.beisu_kuang.setStyleSheet(_ys_xiao_xiang())
        self.beisu_kuang.setFixedWidth(BEISU_KUANG_KUAN)
        self.beisu_kuang.currentIndexChanged.connect(self._qiehuan_beisu)
        di_bu.addWidget(self.beisu_kuang)

        self.a_jingyin = _tubiao_anniu(
            B.SP_MediaVolumeMuted, "静音 / 取消静音", self._qiehuan_jingyin
        )
        di_bu.addWidget(self.a_jingyin)
        self.yinliang_tiao = QtWidgets.QSlider(Qt.Horizontal)
        self.yinliang_tiao.setRange(0, 100)
        self.yinliang_tiao.setValue(0 if JINGYIN else MO_REN_YINLIANG)
        self.yinliang_tiao.setFixedWidth(90)
        self.yinliang_tiao.setStyleSheet(_ys_huadong_tiao())
        self.yinliang_tiao.valueChanged.connect(self._qiehuan_yinliang)
        di_bu.addWidget(self.yinliang_tiao)

        self.a_dakai = _tubiao_anniu(
            B.SP_DialogOpenButton, "打开其他视频", self._xuan_video
        )
        di_bu.addWidget(self.a_dakai)

        di_zhong.addWidget(xia_pai)
        bu.addWidget(di, 0)
        return mianban

    def _jian_canshu_mianban(self):
        # 右栏 = 选项卡：一组功能占一个标签页，各管各的，互不干扰。
        # 以后加新功能就往这里 addTab，别往老标签里塞东西。
        tabs = QtWidgets.QTabWidget()
        tabs.setTabBar(_JunfenTabBar(tabs))     # 标签均分整条右栏
        tabs.setObjectName("YsgRightTabs")
        tabs.setMinimumWidth(330)
        # 先把「字幕编辑」建出来再挂标签：它要跟着时间轴的选中走，
        # 得赶在别的东西开始发信号之前存在。
        bianji = self._jian_zimubianji_tab()
        tabs.addTab(self._jian_yingzimu_tab(), "硬字幕提取")
        tabs.addTab(bianji, "字幕编辑")
        # 上次关窗时停在哪个标签页，这次打开就停哪个（头一回 = 硬字幕提取）
        try:
            ye = int(self._weizhi_peizhi.value(JIEMIAN_TAB_JIAN, 0) or 0)
        except (TypeError, ValueError):
            ye = 0
        if 0 <= ye < tabs.count():
            tabs.setCurrentIndex(ye)
        # 切页时，把上次记下的、属于那一页的分栏位置摆回去
        tabs.currentChanged.connect(self._tab_huan_le)
        self.you_tabs = tabs
        return tabs

    def _jian_yingzimu_tab(self):
        """「硬字幕提取」标签页：推理 / 提取的全部设置 + 时间轴和字幕列表"""
        # 两块面板之间也能拖着改占比
        ce = QtWidgets.QSplitter(Qt.Vertical)
        ce.setHandleWidth(9)
        ce.setChildrenCollapsible(False)
        self.tiqu_ce = ce         # 留着引用：关窗时要记它的分栏位置

        # 上块：实时推理 / 硬字幕提取的所有设置
        shang = QtWidgets.QFrame()
        shang.setObjectName("YsgPanel")
        shang_bu = QtWidgets.QVBoxLayout(shang)
        shang_bu.setContentsMargins(0, 0, 0, 0)
        shang_bu.setSpacing(0)
        self.shezhi_mianban = ShezhiMianban(self)
        self.shezhi_mianban.kaishi_qingqiu.connect(self._kaishi_yunsuan)
        self.shezhi_mianban.tingzhi_qingqiu.connect(self._tingzhi_yunsuan)
        self.shezhi_mianban.kuang_xuan_qingqiu.connect(self._qiehuan_kuang_xuan)
        self.shezhi_mianban.quyu_biangeng.connect(self.huamian.shezhi_quyu)
        self.shezhi_mianban.huamian_jingzhi.connect(
            self._shoudao_huamian_jingzhi
        )
        self.huamian.quyu_bian.connect(self._shoudao_kuang_xuan)
        shang_bu.addWidget(self.shezhi_mianban)
        ce.addWidget(shang)

        # 下块：OCR 的时间轴 + 字幕列表
        xia = QtWidgets.QFrame()
        xia.setObjectName("YsgPanel")
        xia_bu = QtWidgets.QVBoxLayout(xia)
        xia_bu.setContentsMargins(0, 0, 0, 0)
        xia_bu.setSpacing(0)
        self.zimu_mianban = ZimuMianban()
        self.zimu_mianban.tiaozheng.connect(self._shoudao_tiaozheng)
        self.zimu_mianban.xuan_zhong.connect(self._shoudao_xuan_zhong_zimu)
        self.zimu_mianban.zimu_xiugai.connect(self._shoudao_zimu_xiugai)
        xia_bu.addWidget(self.zimu_mianban)
        ce.addWidget(xia)

        ce.setSizes([460, 240])
        return ce

    def _jian_zimubianji_tab(self):
        """「字幕编辑」标签页：上面字幕列表，下面字幕编辑区

        面板自己就是上下分栏（中间那条能拖着改占比），不用再套一层。
        """
        self.bianji_mianban = ZimuBianjiMianban()
        self.bianji_mianban.xuan_zhong.connect(self._shoudao_xuan_zhong_zimu)
        self.bianji_mianban.xuan_zhong_duo.connect(
            self._shoudao_xuan_zhong_duo
        )
        self.bianji_mianban.tiaozheng.connect(self._shoudao_tiaozheng)
        self.bianji_mianban.wenben_gaile.connect(
            self._shoudao_zimu_zhengzai_gai
        )
        self.bianji_mianban.bianji_wancheng.connect(
            self._shoudao_zimu_bianji_wancheng
        )
        self.bianji_mianban.gongju_gaile.connect(self._shoudao_gongju)
        return self.bianji_mianban

    # --------------------------------------------------------------
    # 在画面上框选检测区域
    # --------------------------------------------------------------
    def _qiehuan_kuang_xuan(self):
        """面板点了「框选区域」：进/出框选状态都在这里切"""
        if self.huamian.zai_kuang_xuan():
            self.huamian.tingzhi_kuang_xuan()
            self.shezhi_mianban.kuang_xuan_zhuangtai(False)
            return
        if self.huamian.xianshi_chicun() is None:
            QtWidgets.QMessageBox.warning(self, "提示", "视频画面还没准备好")
            return
        self.huamian.kaishi_kuang_xuan()
        self.shezhi_mianban.kuang_xuan_zhuangtai(True)

    def _shoudao_kuang_xuan(self, quyu):
        """在画面上拖完了"""
        self.shezhi_mianban.kuang_xuan_zhuangtai(False)
        if quyu:
            self.shezhi_mianban.shezhi_quyu(quyu)

    def _jian_shijianzhou_mianban(self):
        kuang = QtWidgets.QFrame()
        kuang.setObjectName("YsgTimeline")
        bu = QtWidgets.QVBoxLayout(kuang)
        bu.setContentsMargins(0, 0, 0, 0)
        bu.setSpacing(0)

        self.shijianzhou = ShijianZhou()
        self.shijianzhou.tiaozheng.connect(self._shoudao_tiaozheng)
        self.shijianzhou.tuo_kaishi.connect(self._tuo_bofangtou_kaishi)
        self.shijianzhou.tuo_jieshu.connect(self._tuo_bofangtou_wancheng)
        self.shijianzhou.xuan_zhong.connect(self._shoudao_xuan_zhong_zimu)
        self.shijianzhou.zimu_tuo.connect(self._shoudao_zimu_tuo)
        self.shijianzhou.shanchu_zimu.connect(self._shoudao_zimu_shanchu)
        self.shijianzhou.hebing_zimu.connect(self._shoudao_zimu_hebing)
        self.shijianzhou.shitu_bian.connect(self._shuaxin_shitu_tiao)
        self.shijianzhou.fangxiang_jian.connect(self._fangxiangjian_tiaobu)
        self.shijianzhou.zimu_nuo.connect(self._shoudao_zimu_nuo)
        self.shijianzhou.xiayitiao.connect(self._shoudao_xiayitiao)
        self.shijianzhou.fuzhi_kuai.connect(self._shoudao_kuai_fuzhi)
        self.shijianzhou.zhantie_kuai.connect(self._shoudao_kuai_zhantie)
        self.shijianzhou.kuang_xuan.connect(self._shoudao_kuang_xuan_kuai)
        # 时间轴能拿键盘焦点：点它之后空格 / 方向键 / 回车才落到这儿
        self.shijianzhou.setFocusPolicy(Qt.StrongFocus)
        bu.addWidget(self.shijianzhou, 1)

        self.shijianzhou_tiao = QtWidgets.QScrollBar(Qt.Horizontal)
        self.shijianzhou_tiao.setObjectName("YsgShijianZhouTiao")
        self.shijianzhou_tiao.setFixedHeight(10)
        self.shijianzhou_tiao.setRange(0, 1000)
        self.shijianzhou_tiao.setValue(0)
        self.shijianzhou_tiao.setEnabled(False)
        self.shijianzhou_tiao.valueChanged.connect(self._shijianzhou_pingyi)
        bu.addWidget(self.shijianzhou_tiao, 0)
        # 整块留着给外面搬：默认在窗口最下方，也能跟「字幕列表」换位置
        self.shijianzhou_kuang = kuang
        return kuang

    # --------------------------------------------------------------
    # 打开 / 关闭视频
    # --------------------------------------------------------------
    def dakai(self, lujing):
        if not lujing or not osp.isfile(lujing):
            QtWidgets.QMessageBox.warning(self, "提示", "视频文件不存在")
            return False

        self._guanbi_video()
        self.lujing = lujing
        self.setWindowTitle(f"视频工作台 - {osp.basename(lujing)}")

        cap = cv2.VideoCapture(lujing)
        if not cap.isOpened():
            cap.release()
            QtWidgets.QMessageBox.warning(
                self, "提示", f"打不开这个视频：\n{osp.basename(lujing)}"
            )
            logger.error(f"视频工作台：打不开视频 {lujing}")
            return False
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.zong_zhen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.zhen_kuan = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.zhen_gao = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        # 让取帧线程按画面区的实际显示尺寸缩放，主线程就不必每帧做大图缩放
        self.huamian.shezhi_shipin_chicun(self.zhen_kuan, self.zhen_gao)
        # 换视频了：退出框选、清掉上一部片子的检测区域
        self.huamian.tingzhi_kuang_xuan()
        self.huamian.shezhi_quyu(None)
        self.shezhi_mianban.kuang_xuan_zhuangtai(False)

        if self.fps <= 0:
            self.fps = 30.0
        if self.zong_zhen <= 0:
            self.zong_zhen = int(round(self.fps * 1))
        self.shichang_ms = int(round(self.zong_zhen / self.fps * 1000))

        self.jindu_tiao.shezhi_shichang(self.shichang_ms)
        self.jindu_tiao.shezhi_bofangtou(0)
        self.shijianzhou.shezhi_shichang(self.shichang_ms)
        self.shijianzhou.shezhi_fps(self.fps)
        self.bianji_mianban.shezhi_fps(self.fps)
        self.shijianzhou.shezhi_bofangtou(0)
        self.shijianzhou.qingkong_bofang()
        self.shijianzhou.shezhi_bo_tishi("正在解析音频…")
        # 放大倍数：上次拖到多少就用多少（头一回 ×15，×1 会把整条视频塞进屏里，
        # 字幕块缩成指甲盖大小看不清）
        self.shijianzhou.shezhi_beishu(self._suofang_qishi(), 0)
        self._shuaxin_shitu_tiao()

        if self.yinpin is not None:
            self.yinpin.setVolume(
                0 if JINGYIN else int(self.yinliang_tiao.value())
            )

        self._qidong_duqu()

        # 后台解音轨画波形
        self.shijianzhou.zhunbei_bofang(
            int(self.shichang_ms / max(1, BO_HAO_MIAO)) + 64
        )
        self.bo_xiancheng = _BofangXiancheng(lujing, self)
        self.bo_xiancheng.kuaijie.connect(self._shoudao_bofang)
        self.bo_xiancheng.jieshu.connect(self._bo_jieshu)
        self.bo_xiancheng.start()

        _shezhi_tubiao(self.a_bofang, QtWidgets.QStyle.SP_MediaPlay)
        self._shuaxin_jingyin_tubiao()
        self.zhengzai_bofang = False
        self._shuaxin_anniu()

        # 通知右侧两块面板 + 清掉上一部视频的框和字幕块
        self.shezhi_mianban.shezhi_video(
            lujing, self.fps, self.zhen_kuan, self.zhen_gao, self.zong_zhen
        )
        self.zimu_mianban.qingkong()
        self._chexiao_qingkong()
        self.zimu_mianban.shezhi_shichang(self.shichang_ms)
        self.shijianzhou.qingkong_zimu()
        self._shezhi_bianji_zimu([])
        self.huamian.qingkong_kuang()
        # 字幕整个换了一批：上一份打开 / 另存过的文件不能再当 Ctrl+S 的目标
        self.zimu_wenjian = ""
        self.zimu_geshi = ""
        self._ass_yuan = None
        self._zidong_gai_dong = False
        self._zidong_shange = None
        logger.info(
            f"视频工作台：已打开 {lujing}  "
            f"{self.zhen_kuan}x{self.zhen_gao} {self.fps:.3f}fps "
            f"{self.zong_zhen}帧"
        )
        return True

    def _guanbi_video(self):
        self.xianshi_dingshi.stop()
        self.tuo_dingshi.stop()
        self._tuo_bt_zhong = False
        self._tuo_dingwei_zhen = None
        # 模型还在后台加载：标记取消，加载完别再去碰已经清空的界面
        if self._jiazai_xc is not None and self._jiazai_xc.isRunning():
            self._jiazai_quxiao = True
        # 先停推理，再停取帧（顺序反了的话，推理槽位里还会留着一帧没主的图）
        if self.yunsuan is not None:
            self.yunsuan.tingzhi()
            if not self.yunsuan.isRunning():
                self.yunsuan = None
        self.chuliqi = None
        self.yunsuan_moshi = ""
        self._tingzhi_tuili_qu()
        if self.duqu is not None:
            self.duqu.tingzhi()
            self.duqu = None
        if self.bo_xiancheng is not None:
            self.bo_xiancheng.tingzhi()
            self.bo_xiancheng = None
        self.mpv.zanting()
        self.zhengzai_bofang = False
        self.huamian.qingkong()
        self.huamian.qingkong_kuang()
        self.huamian.tingzhi_kuang_xuan()
        # 换片子 / 关窗口：字幕块和画面下方那条都清掉
        self.zimu_mianban.qingkong()
        self._chexiao_qingkong()
        self.shijianzhou.qingkong_zimu()
        self._shezhi_bianji_zimu([])
        self.zimu_wenjian = ""
        self.zimu_geshi = ""
        self._ass_yuan = None
        self._zidong_gai_dong = False
        self._zidong_shange = None
        self._zimu_hua_ji = None
        self._zimu_pei_ji = []
        self._zimu_fu_ji = []
        self._fu_wai = {}
        self.huamian.shezhi_zimu([])
        self._bofang_ms = 0
        self._mubiao_zhen = None
        self._zimu_tiao_wen = None
        if hasattr(self, "zimu_tiao"):
            self.zimu_tiao.setText("")
            self.zimu_tiao.setToolTip("")
        self.shezhi_mianban._suoding(False)

    def _shoudao_bofang(self, ge_hao, mx, mn):
        """波形解码线程送来的每一段包络"""
        self.shijianzhou.tianjia_bofang(ge_hao, mx, mn)

    def _bo_jieshu(self, cuowu):
        if cuowu:
            logger.warning(f"视频工作台：波形解析失败（{cuowu}）")
            self.shijianzhou.shezhi_bo_tishi(f"没有波形：{cuowu}")
        else:
            logger.info("视频工作台：波形解析完成")
            self.shijianzhou.shezhi_bo_tishi("")

    def _qidong_duqu(self):
        """把片子交给 mpv：解码 / 声音 / 定位 / 时钟全归它"""
        if self.mpv.h is None:
            logger.error(
                f"视频工作台：没有播放内核，播不了（{_MPV_JIAZAI_CUO}）"
            )
            return
        # 渲染上下文必须先建好，不然后面 mpv 会自己去开窗口
        self.huamian.zhunbei_mpv()
        if self.mpv.dakai(self.lujing):
            self.mpv.fps = float(self.fps or 30.0)
            self.mpv.shezhi_beisu(self.beisu)
            self.duqu = self.mpv      # 外面那些 if self.duqu is None 的判断照旧好使
            self._daowei_bao = False
            self.xianshi_dingshi.start()
        else:
            logger.error(f"视频工作台：播放内核载入失败 {self.lujing}")

    def _tongbu_suo_dao(self):
        """把画面区该显示的尺寸告诉取帧线程（缩放由线程做，主线程只负责贴图）"""
        if self.duqu is None:
            return
        self.duqu.suo_dao = self.huamian.xianshi_chicun()

    def _he_huamian_gao(self):
        """画面多大，上半栏就只留多大 —— 上下不留黑边

        画面是等比居中贴的：上半栏比画面高，上下就各多出一条黑边。这里按画面
        自己的比例把上半栏收一收，省出来的高度自动落到时间轴上 —— 时间轴想拖
        高照样拖，往上顶到"刚好"就停住，不会拖出黑边来。

        只有画面变宽 / 换了别的片子（宽高比变了）时才主动把上半栏调到刚好；
        纯上下拖分栏不主动调，免得跟人抢。这里只算，动手统一交给定时器。
        """
        if self._shangxia_duoguo:
            return              # 上下大分栏你自己拖过：听你的，不自动贴合
        if getattr(self, "shang_xia", None) is None:
            return
        yuan = self.huamian.yuan_chicun()
        kuan = self.huamian.width()
        if not yuan or kuan <= 8:
            self._shang_ci_he = None        # 没画面：下次重新算
            self._he_dai = None
            return
        # 画面以外那几块（标题栏 / 字幕条 / 播放控制）占多高；它们不随画面变
        guding = self._yulan.height() - self.huamian.height()
        if guding <= 0:
            return
        hua = max(1, int(round(yuan[1] * float(kuan) / float(yuan[0]))))
        mubiao = guding + hua
        da_xiao = self.shang_xia.sizes()
        if len(da_xiao) < 2:
            return
        dangqian = (kuan, yuan)
        zhudong = dangqian != self._shang_ci_he     # 换宽 / 换片 -> 主动调到刚好
        if not zhudong and da_xiao[0] <= mubiao:
            return                          # 已经刚好 / 人自己拖矮了：不动
        if self._he_sile == (mubiao, da_xiao[0]):
            return                          # 这个高度收不动也放不动，别白排
        xia = self.shang_xia.widget(1)
        xia_zui_xiao = xia.minimumHeight() if xia is not None else 0
        if sum(da_xiao) - mubiao < xia_zui_xiao:
            return                          # 时间轴没地方让了，别硬收
        self._shang_ci_he = dangqian
        if da_xiao[0] != mubiao:
            self._he_dai = mubiao
            self._he_jishi.start(0)

    def _shishi_he_gao(self):
        """真去改上半栏的高度（延到下一轮做，不在 resize 里面直接动分栏）"""
        mubiao = self._he_dai
        self._he_dai = None
        if not mubiao or getattr(self, "shang_xia", None) is None:
            return
        da_xiao = self.shang_xia.sizes()
        if len(da_xiao) < 2 or da_xiao[0] == mubiao:
            return
        self.shang_xia.setSizes([mubiao, sum(da_xiao) - mubiao])
        xian = self.shang_xia.sizes()[0]
        # 分栏没给这个高度（被最小高度顶住了）：记下来，别再反复试
        self._he_sile = (mubiao, xian) if xian == da_xiao[0] else None

    def _qu_zhen_xianshi(self):
        """定时处理 mpv 事件及片尾检测；播放头由新帧回调更新。"""
        if self.mpv.h is None:
            return
        self.mpv.pai_shijian()
        # 播到结尾：位置得真在尾巴上才算。刚按播放时要是从结尾跳回开头，
        # mpv 的 eof-reached 会晚一拍才翻过来，那会儿位置已经在开头了 ——
        # 拿位置一起卡住，就不会一按播放又被判成"已经播完"。
        if (
            self.zhengzai_bofang
            and self.mpv.daowei_ma()
            and self.shichang_ms > 0
            and self.mpv.shijian_ms() >= self.shichang_ms - 300
        ):
            self._daowei_bao = True
            self._bofang_daowei()
            return

    # --------------------------------------------------------------
    # 播放控制
    # --------------------------------------------------------------
    def _kongge_an_xia(self):
        """按了空格：播放 / 暂停

        焦点在输入框里的话不抢 —— 那儿空格就该是空格。
        """
        zhu = QtWidgets.QApplication.focusWidget()
        if isinstance(
            zhu,
            (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
             QtWidgets.QPlainTextEdit),
        ):
            return
        self._qiehuan_bofang()

    def _qiehuan_bofang(self):
        if self.duqu is None:
            return
        self._bo_dao_ms = None      # 手动按播放 = 一路播下去，不设终点
        if self.zhengzai_bofang:
            self._zanting()
        else:
            self._bofang()

    # --------------------------------------------------------------
    # 独立字幕条的开关
    # --------------------------------------------------------------
    def _du_zimu_tiao_kai(self):
        """上次把画面下方那条独立字幕条开着没（头一回 = 开着）"""
        zhi = self._weizhi_peizhi.value(JIEMIAN_ZIMU_TIAO_JIAN, None)
        if zhi is None:
            return True
        return str(zhi).strip().lower() in ("1", "true", "yes", "on")

    def _du_kaiguan(self, jian, moren):
        """读一个界面开关（存 gongzuotai.ini，跟字幕条开关一处）；头一回用 moren"""
        try:
            zhi = self._weizhi_peizhi.value(jian, None)
        except Exception:  # noqa
            return bool(moren)
        if zhi is None:
            return bool(moren)
        return str(zhi).strip().lower() in ("1", "true", "yes", "on")

    def _cun_kaiguan(self, jian, kai):
        """存一个界面开关"""
        try:
            self._weizhi_peizhi.setValue(jian, bool(kai))
            self._weizhi_peizhi.sync()
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：记录开关失败 {cuowu}")

    def _qiehuan_zimu_tiao(self, kai):
        """显示 / 隐藏画面下方那条独立字幕；关了就收掉那块地方，视频直接贴着底栏"""
        self.zimu_tiao.setVisible(bool(kai))
        try:
            self._weizhi_peizhi.setValue(JIEMIAN_ZIMU_TIAO_JIAN, bool(kai))
        except Exception:  # noqa
            pass

    def _tab_an_xia(self):
        """按了 Tab：播放 / 暂停

        视频工作台里哪儿按都好使（在字幕编辑框里打字时也一样），
        按键在 eventFilter 里已经被吃掉了，不会变成 Tab 缩进。
        """
        self._qiehuan_bofang()

    def _dangqian_zimu_kuai(self):
        """现在说的是哪一条字幕 -> (开始 ms, 结束 ms)；没有给 None

        编辑页里选中的那条优先（人正对着它改字），其次看时间轴上选中的。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        xu = self.bianji_mianban.dangqian_xu()
        if not (0 <= xu < len(zimu)):
            xuan = self.shijianzhou.xuan_zhong_liebiao()
            xu = xuan[0] if xuan else -1
        if not (0 <= xu < len(zimu)):
            return None
        return int(zimu[xu][0]), int(zimu[xu][1])

    def _bofang_dangqian_kuai(self):
        """按 ` 或 ~：只把当前这条字幕块播一遍

        从块头开播，播到块尾自动停下；再按一次，还是从块头重播。
        视频工作台里哪儿按都好使，按键在 eventFilter 里已经被吃掉了，
        不会落进编辑框变成波浪号。
        """
        if self.duqu is None or self.fps <= 0:
            return
        kuai = self._dangqian_zimu_kuai()
        if kuai is None:
            return
        qi, zhi = kuai
        self._bo_dao_ms = int(zhi)
        self._zanting()
        self._shoudao_tiaozheng(qi)
        self._bofang()

    def _shezhi_zai_bofang(self, zai):
        """改「在不在播」并告诉时间轴 —— 实时滚动那个开关只在播的时候生效"""
        self.zhengzai_bofang = bool(zai)
        zhou = getattr(self, "shijianzhou", None)
        if zhou is not None:
            zhou.shezhi_zai_bofang(bool(zai))

    def _bofang(self):
        if self.duqu is None:
            return
        if self.shichang_ms > 0 and self._dangqian_ms() >= self.shichang_ms - 1:
            self._tiaozheng_zhen(0)
        self._mubiao_zhen = None    # 播放中播放头就该跟着帧走，不设守卫
        self._shezhi_zai_bofang(True)
        self._daowei_bao = False
        _shezhi_tubiao(self.a_bofang, QtWidgets.QStyle.SP_MediaPause)
        self.mpv.jixu()

    def _zanting(self):
        self._shezhi_zai_bofang(False)
        _shezhi_tubiao(self.a_bofang, QtWidgets.QStyle.SP_MediaPlay)
        self.mpv.zanting()

    def _tiaobu_zhen(self, zhen_shu):
        """按钮上的上一帧 / 下一帧：点一下走一帧"""
        if self.duqu is None:
            return
        self._zanting()
        mubiao = max(0, min(self.zong_zhen - 1, self.duqu.zhen_hao() + zhen_shu))
        self._tiaozheng_zhen(mubiao)

    def _fangxiangjian_tiaobu(self, fangxiang):
        """方向键微调播放位置

        按一下走一帧；按住不放（键盘自动重复）越走越快：
        1 帧 -> 2 帧 -> 5 帧 -> 10 帧 -> 20 帧。
        所以"精确对轴"和"大范围快退快进"用同一个键就行。

        累加的基准不能用"视频当前帧号"——定位是后台异步做的，
        帧号更新会慢半拍，连按十下全从同一个旧帧号加 1，
        结果是原地不动。所以记住这一串走到哪儿了，下一下接着走。
        """
        if self.duqu is None or self.fps <= 0:
            return
        if self.zhengzai_bofang:
            self._zanting()         # 一动就先停下，免得播放和微调抢播放头
        xian = time.perf_counter()
        if (
            self._shangci_fangxiang != fangxiang
            or xian - self._shangci_shike > FANGXIANG_JIAN_GE_MIAO
        ):
            # 换方向了，或者停顿够久：当成新的一串，从当前位置起步
            self._lianxu_cishu = 0
            self._zou_dao_ms = float(self._bofang_ms)
        self._shangci_fangxiang = fangxiang
        self._shangci_shike = xian
        self._lianxu_cishu += 1

        # 连着按得越多，每下迈的步子越大
        if self._lianxu_cishu <= 4:
            bu = 1
        elif self._lianxu_cishu <= 12:
            bu = 2
        elif self._lianxu_cishu <= 24:
            bu = 5
        elif self._lianxu_cishu <= 40:
            bu = 10
        else:
            bu = 20

        # 方向键永远挪播放头（选中字幕块也一样）：打轴时是"选中块 + 一帧帧挪
        # 播放头看硬字幕对不对齐"，对不上就用 E / R 把块的起止帧拉过来。
        # 挪块那套（nuo_xuan_zhong）在这条路上不用了。
        mubiao = self._zou_dao_ms + fangxiang * bu * (1000.0 / self.fps)
        mubiao = max(0.0, min(float(self.shichang_ms), mubiao))
        zhen = int(round(mubiao / 1000.0 * self.fps))
        zhen = max(0, min(max(0, self.zong_zhen - 1), zhen))
        self._zou_dao_ms = float(self._tiaozheng_zhen(zhen))

    def _dazhou_an(self, dongzuo):
        """Q / W / E / R / A / D 打轴：只认选中的那一条字幕块（单选）

        Q 播放头 -> 块开始帧；W 播放头 -> 块结束帧；
        E 块开始 <- 播放头；R 块结束 <- 播放头；
        A 块尾贴下一块块头；D 块头贴上一块块尾。
        """
        if self.duqu is None or self.fps <= 0:
            return
        xuan = self.shijianzhou.xuan_zhong_liebiao()
        if len(xuan) != 1:
            return
        xu = xuan[0]
        zimu = self.shijianzhou.zimu_liebiao()
        if not (0 <= xu < len(zimu)):
            return
        qi, zhi, _w = zimu[xu]
        if dongzuo == "Q":
            self._zanting()
            self._tiaozheng_zhen(int(round(qi / 1000.0 * self.fps)))
        elif dongzuo == "W":
            self._zanting()
            self._tiaozheng_zhen(int(round(zhi / 1000.0 * self.fps)))
        elif dongzuo == "E":
            self.shijianzhou.shezhi_kuai_bian(qi_ms=self._bofang_ms)
        elif dongzuo == "R":
            self.shijianzhou.shezhi_kuai_bian(zhi_ms=self._bofang_ms)
        elif dongzuo == "A":
            self.shijianzhou.jin_tie(1)
        elif dongzuo == "D":
            self.shijianzhou.jin_tie(-1)

    def _tiaobu(self, ms):
        if self.duqu is None:
            return
        self._zanting()
        mubiao_ms = max(0, min(self.shichang_ms, self._dangqian_ms() + ms))
        zhen = int(round(mubiao_ms / 1000.0 * self.fps))
        self._tiaozheng_zhen(max(0, min(self.zong_zhen - 1, zhen)))

    def _tiaozheng_zhen(self, zhen_hao, yinpin=True, jingque=None):
        """定位到第 zhen_hao 帧；返回吸附到该帧后的毫秒数

        jingque 不给（None）就看是不是正拖着播放头：
          · 拖着 -> 只挪界面，定位先攒着，由 _fa_tuo_dingwei 按节拍交给队列；
          · 别的 -> 立刻交给队列（精确的）。
        松手那一下会补一发精确的（见 _tuo_bofangtou_wancheng）。

        yinpin 是老接口留下的：以前画面和声音分开，拖播放头时要单独把声音
        拦住。现在声音也归 mpv，一次 seek 画面和声音一起走，而且拖动时是
        暂停状态本来就不出声，所以这个参数收下不用。
        """
        if self.duqu is None:
            return int(self._bofang_ms)
        zhen_hao = int(zhen_hao)
        if jingque is not None:
            # 点名要精确（松手那一下 / 打轴）：立刻发
            self.duqu.tiaozheng(zhen_hao, dan_bu=True, jingque=jingque)
            self._shang_ci_tiaozheng_zhen = zhen_hao
        elif self._tuo_bt_zhong:
            # 鼠标正拖着：这儿不发，先攒着。等节拍到点由 _fa_tuo_dingwei 把
            # 最新那个交给队列（队列一次只放一发在飞，落地了才发下一个）
            self._tuo_dingwei_zhen = zhen_hao
        elif zhen_hao != self._shang_ci_tiaozheng_zhen:
            # 方向键 / 点一下定位：每次真跳，同帧不重复跳
            self.duqu.tiaozheng(zhen_hao, dan_bu=True, jingque=True)
            self._shang_ci_tiaozheng_zhen = zhen_hao
        # 记下"我要的是哪一帧"：等它回来之前，报上来的旧位置不许动播放头
        self._mubiao_zhen = (zhen_hao, time.perf_counter())
        # 界面（播放头 / 时间码 / 画面字幕）立刻跟到鼠标位置，不等画面
        return self._gen_xin_weizhi(zhen_hao)

    def _gen_xin_weizhi(self, zhen_hao):
        """光把界面（播放头 / 时间码 / 画面字幕）挪到这一帧，不给 mpv 发定位

        mpv 单帧步进那条路用：那一帧是 mpv 自己走的，位置它已经挪了，我们
        这边跟上就行。
        """
        zhen_hao = max(0, min(max(0, self.zong_zhen - 1), int(zhen_hao)))
        ms = int(round(zhen_hao / max(1e-6, self.fps) * 1000))
        self._bofang_ms = ms
        self.shijianzhou.shezhi_bofangtou(ms)
        self.jindu_tiao.shezhi_bofangtou(ms)
        self.shijian_dangqian.setText(_shijian_zhen_wenben(ms, zhen_hao))
        # hua=False：画面上那层字幕这次不换 —— 播放头/时间码立刻跟到鼠标位置，
        # 那层等 mpv 真出帧时由 paintGL 换，两边就同一刻出来（见 _shuaxin_zimu_tiao）
        self._shuaxin_zimu_tiao(hua=False)
        return ms

    def _shoudao_tiaozheng(self, ms):
        zhen = int(round(ms / 1000.0 * self.fps))
        zhen = max(0, min(max(0, self.zong_zhen - 1), zhen))
        # 鼠标正拖着播放头：界面立刻跟到鼠标位置（_tiaozheng_zhen 里做），
        # 给 mpv 的定位则压在 TUO_SEEK_JIAN_GE_MS 的节拍上，一次只发最新的
        # 那个位置；松手那一下才补一发精确的落到鼠标指着的那一帧。
        self._tiaozheng_zhen(zhen, yinpin=not self._tuo_bt_zhong)

    def _fa_tuo_dingwei(self):
        """拖动中的节拍：把攒着的最新位置交给定位队列

        指针一秒能挪上百个位置，一个个发 seek 会把 mpv 淹了。这里按节拍看一
        眼，每次只把最新那个交出去，中间滑过的位置一律扔掉。交出去之后由
        MpvBofang 排队 —— 上一发没落地就不发新的，只记最新位置，落地立刻
        补上（见 MpvBofang._pai_tiaozheng）。都是精确到帧的定位。
        """
        if self.duqu is None or not self._tuo_bt_zhong:
            return
        zhen_hao = self._tuo_dingwei_zhen
        self._tuo_dingwei_zhen = None
        if zhen_hao is None or zhen_hao == self._shang_ci_tiaozheng_zhen:
            return
        self.duqu.tiaozheng(zhen_hao, dan_bu=False, jingque=True)
        self._shang_ci_tiaozheng_zhen = zhen_hao

    # --------------------------------------------------------------
    # 鼠标拖播放头（时间轴刻度尺 / 视频下方进度条）
    #   照 Arctime：一按下去，正在播的就立刻停播、声音停；
    #   拖动过程只让画面跟着刷新（不出声）；松手停在原地，
    #   要接着看就再按一次播放 —— 不会自动续播。
    # --------------------------------------------------------------
    def _tuo_bofangtou_kaishi(self):
        """鼠标接管播放头：正在播放就立刻停下（画面改由拖动位置驱动）

        拖动期间主线程的活要尽量少：时间轴只画播放头那一小条、字幕列表的
        上色和滚动先停下 —— 不然鼠标一快，播放头就落在鼠标后头。
        """
        self._tuo_bt_zhong = True
        self._tuo_dingwei_zhen = None
        self.tuo_dingshi.start()
        self.shijianzhou.shezhi_kuaisu_hua(True)
        if self.zhengzai_bofang:
            self._zanting()

    def _tuo_bofangtou_wancheng(self):
        """手松了：把最后一发精确落位交给队列，再恢复正常重绘

        松手这一下要落到鼠标指着的那一帧上。队列要是还有一发在飞，就先排
        上（落地后自动补发，见 MpvBofang._pai_tiaozheng）—— 不硬插进去打断
        它，省得白解一次。字幕列表拖动期间一直没刷，也跟着补一次。
        """
        self._tuo_bt_zhong = False
        self.tuo_dingshi.stop()
        self._tuo_dingwei_zhen = None
        self.shijianzhou.shezhi_kuaisu_hua(False)
        if self.duqu is not None:
            zhen = int(round(self._bofang_ms / 1000.0 * max(1e-6, self.fps)))
            self._tiaozheng_zhen(zhen, jingque=True)
            self.shijianzhou.shezhi_bofangtou(self._bofang_ms, gensui=False)
        self.bianji_mianban.shezhi_bofang_ms(self._bofang_ms, False)
        self.zimu_mianban.shezhi_bofangtou(self._bofang_ms)

    def _qiehuan_beisu(self):
        b = self.beisu_kuang.currentData() or 1.0
        self.beisu = float(b)
        if self.duqu is not None:
            self.duqu.shezhi_beisu(self.beisu)
        if self.yinpin is not None:
            try:
                self.yinpin.setPlaybackRate(self.beisu)
            except Exception:  # noqa
                pass

    def _qiehuan_jingyin(self):
        if self.yinpin is None:
            return
        if self.yinpin.volume() > 0:
            self.yinliang_tiao.setValue(0)
        elif self.yinliang_tiao.value() == 0:
            self.yinliang_tiao.setValue(MO_REN_YINLIANG)
        self._shuaxin_jingyin_tubiao()

    def _qiehuan_yinliang(self, zhi):
        if self.yinpin is not None:
            self.yinpin.setVolume(max(0, min(100, int(zhi))))
        self._shuaxin_jingyin_tubiao()

    def _shuaxin_jingyin_tubiao(self):
        jing = self.yinliang_tiao.value() == 0
        _shezhi_tubiao(
            self.a_jingyin,
            QtWidgets.QStyle.SP_MediaVolumeMuted
            if jing
            else QtWidgets.QStyle.SP_MediaVolume,
        )

    def _suofang_qishi(self):
        """时间轴的初始放大倍数：上次关窗时多少就多少，头一回 SUOFANG_MOREN"""
        try:
            bei = float(
                self._weizhi_peizhi.value(JIEMIAN_SUOFANG_JIAN, SUOFANG_MOREN)
            )
        except (TypeError, ValueError):
            bei = SUOFANG_MOREN
        return max(1.0, min(float(SUOFANG_ZUIDA), bei))

    def _shijianzhou_suofang(self, beishu):
        """改缩放数值框（敲数字 / 上下箭头 / 滚轮）-> 改时间轴放大倍数"""
        self.shijianzhou.shezhi_beishu(float(beishu))

    def _shuaxin_shitu_tiao(self):
        """时间轴视图变了 -> 刷新缩放数值框（时间轴内部自己缩放时用）"""
        bei = int(round(self.shijianzhou.beishu()))
        if self.shijianzhou_suofang.value() != bei:
            self.shijianzhou_suofang.blockSignals(True)
            self.shijianzhou_suofang.setValue(bei)
            self.shijianzhou_suofang.blockSignals(False)

        you = bei > 1.001
        self.shijianzhou_tiao.setEnabled(you)
        self.shijianzhou_tiao.blockSignals(True)
        self.shijianzhou_tiao.setPageStep(max(1, int(round(1000.0 / bei))))
        self.shijianzhou_tiao.setValue(
            int(round(self.shijianzhou.shitu_bili() * 1000))
        )
        self.shijianzhou_tiao.blockSignals(False)

    def _shijianzhou_pingyi(self, zhi):
        """拖底下滚动条 -> 时间轴左右平移"""
        self.shijianzhou.shezhi_shitu_bili(zhi / 1000.0)

    # --------------------------------------------------------------
    # 帧回调
    # --------------------------------------------------------------
    def _shoudao_zhen(self, zhen_rgb, zhen_hao):
        """外面把整张图塞进来（全片扫描时那一帧）——直接贴上去，画面归它

        平时画面是 mpv 自己渲染的，走不到这儿。
        """
        tu = _zhen_tu_huan(zhen_rgb)
        self.huamian.shezhi_zhen(tu)
        ms = int(round(zhen_hao / max(1e-6, self.fps) * 1000))
        self._geng_xin_bofangtou(ms, zhen_hao)

    def _mpv_huan_zhen(self):
        """mpv 已渲染新帧：播放时同步播放头，随后同步画面字幕。"""
        if self.duqu is not None:
            ms = self.mpv.shijian_ms()
            if self.zhengzai_bofang and not self._tuo_bt_zhong:
                zhen = int(round(ms / 1000.0 * max(1e-6, self.fps)))
                self._geng_xin_bofangtou(ms, zhen, hua=False)
            self._shuaxin_huamian_zimu(ms)

    def _geng_xin_bofangtou(self, ms, zhen_hao, hua=True):
        """播放头那一圈（时间轴 / 进度条 / 时间码 / 字幕条）按这个位置刷一遍

        定位刚发出去那会儿报上来的还是旧位置：那会儿不许动播放头，不然
        播放头会在"新位置"和"旧位置"之间来回蹦，看着就是抖。
        """
        mu = self._mubiao_zhen
        if mu is not None:
            if abs(int(zhen_hao) - int(mu[0])) <= 1:
                self._mubiao_zhen = None          # 要的那一帧到了，守卫解除
            elif time.perf_counter() - mu[1] <= DINGWEI_SHOUWEI_MIAO:
                return                            # 还在等，这次不理会
            else:
                self._mubiao_zhen = None          # 等太久了（定位失败），别再拦着
        self._bofang_ms = ms
        self.shijianzhou.shezhi_bofangtou(ms)
        self.jindu_tiao.shezhi_bofangtou(ms)
        self.shijian_dangqian.setText(_shijian_zhen_wenben(ms, zhen_hao))
        # 拖播放头的时候不动列表（见 _shuaxin_zimu_tiao 里的说明）
        if not self._tuo_bt_zhong:
            self.zimu_mianban.shezhi_bofangtou(ms)
        self._shuaxin_zimu_tiao(hua=hua)
        # 「只播当前字幕块」：播到块尾就停在这儿（不在播放中就只是清掉残留）
        dao = self._bo_dao_ms
        if dao is not None and ms >= dao:
            self._bo_dao_ms = None
            if self.zhengzai_bofang:
                self._zanting()

    def _bofang_daowei(self):
        self._shezhi_zai_bofang(False)
        _shezhi_tubiao(self.a_bofang, QtWidgets.QStyle.SP_MediaPlay)
        self.mpv.zanting()
        logger.info("视频工作台：播放到结尾")

    def _dangqian_ms(self):
        if self.duqu is None:
            return 0
        return self.mpv.shijian_ms()

    def _shijian_wenben(self, ms):
        return _shijian_wenben(ms, self.fps)

    # --------------------------------------------------------------
    # 推理 / 硬字幕提取
    # --------------------------------------------------------------
    def _kaishi_yunsuan(self, paofa):
        """面板点了「开始」：先校验、建目录，再把模型加载丢后台"""
        if self._jiazai_xc is not None and self._jiazai_xc.isRunning():
            return                                  # 模型正在加载，别点两下
        if self.yunsuan is not None and self.yunsuan.isRunning():
            QtWidgets.QMessageBox.information(
                self, "提示", "上一次还在收尾，稍等一下再点"
            )
            return
        self.yunsuan = None
        if not self.lujing:
            QtWidgets.QMessageBox.warning(self, "提示", "先打开一个视频")
            return

        canyu = self.shezhi_mianban.canyu()
        shuchu = canyu.get("shuchu") or ""
        if not shuchu:
            return
        try:
            os.makedirs(shuchu, exist_ok=True)
        except OSError as cuowu:
            QtWidgets.QMessageBox.critical(
                self, "错误", f"建不了输出目录：\n{cuowu}"
            )
            return

        # 模型初始化要好几秒。放主线程会把界面整个冻住，
        # 所以丢后台加载，加载完（_moxing_jiazai_wan）才真正开跑。
        self._kaishi_paofa = paofa
        self._kaishi_canyu = canyu
        self._jiazai_quxiao = False
        self.shezhi_mianban._suoding(True)          # 按钮变「停止」，加载期间能取消
        self.shezhi_mianban.zhuangtai_shezhi("正在加载模型…")
        self._jiazai_xc = _JiazaiMoxingXiancheng(self, canyu, self)
        self._jiazai_xc.jiazai_wan.connect(self._moxing_jiazai_wan)
        self._jiazai_xc.start()

    def _moxing_jiazai_wan(self, moxing, ocr_moxing, leibie_id, leibie_ming,
                           cuowu):
        """模型加载完了（后台线程发回来的）——从这里开始才是真的跑"""
        if self._jiazai_quxiao:
            # 加载期间点了停止 / 关了窗口：东西丢掉，界面别再碰
            self._jiazai_quxiao = False
            return
        if cuowu:
            QtWidgets.QMessageBox.warning(self, "提示", cuowu)
            self.shezhi_mianban._suoding(False)
            self.shezhi_mianban.zhuangtai_shezhi(cuowu)
            return

        canyu = self._kaishi_canyu or {}
        paofa = self._kaishi_paofa
        canyu["moxing"] = moxing
        canyu["ocr_moxing"] = ocr_moxing
        canyu["leibie_id"] = leibie_id
        canyu["leibie_ming"] = leibie_ming

        self.zimu_mianban.qingkong()
        self._chexiao_qingkong()
        self.zimu_mianban.shezhi_shichang(self.shichang_ms)
        self.shijianzhou.qingkong_zimu()
        self._shezhi_bianji_zimu([])
        self.huamian.qingkong_kuang()
        # 字幕整个换了一批：上一份打开 / 另存过的文件不能再当 Ctrl+S 的目标
        self.zimu_wenjian = ""
        self.zimu_geshi = ""
        self._ass_yuan = None
        self._zidong_gai_dong = False
        self._zidong_shange = None
        self.chuliqi = ZhenChuli(canyu)
        self.yunsuan_moshi = paofa

        if paofa == "gensui":
            self.yunsuan = TuiliXiancheng(self.chuliqi, self)
            self.yunsuan.jieguo.connect(self._shoudao_tuili_jieguo)
            self.yunsuan.start()
            self._qidong_tuili_qu()
            self.shezhi_mianban.zhuangtai_shezhi("跟随播放中…")
            if not self.zhengzai_bofang:
                self._bofang()
        else:
            self._zanting()
            self.yunsuan = SaomiaoXiancheng(
                self.lujing, self.chuliqi, bool(canyu.get("ocr")),
                bool(canyu.get("huamian_budong")), self
            )
            self.yunsuan.jindu.connect(self.shezhi_mianban.jindu_gengxin)
            self.yunsuan.huamian.connect(self._shoudao_saomiao_huamian)
            self.yunsuan.jieguo.connect(self._shoudao_tuili_jieguo)
            self.yunsuan.jieshu.connect(self._saomiao_jieshu)
            self.yunsuan.start()
            self.shezhi_mianban.zhuangtai_shezhi("正在扫描…")

        logger.info(
            f"视频工作台：开始（{paofa}），输出 {canyu.get('shuchu')}"
        )

    def _jiazai_moxing(self, canyu):
        """加载检测 / OCR 模型

        这个方法跑在后台线程里，不许弹框，出错就把说明当返回值带出去。
        加载过的模型留着复用（跟主界面一样：除非换 YAML，不然不重新加载）。
        返回 (检测模型, OCR 模型, 类别 id, 类别名, 错误说明)，错误说明空串 = 成功
        """
        from anylabeling.views.labeling.utils.video import PPOCRv6Wrapper

        moxing = None
        leibie_id = []
        leibie_ming = {}

        if canyu.get("yong_yolo"):
            yuan = self._qu_moxing("jiance", canyu.get("moxing_lu") or "")
            if yuan is None:
                return None, None, [], {}, "检测模型加载失败，看看 YAML 选对没有"
            # 直接用模型本身：postprocess 能一次给出框、类别、分数，
            # 借 _OnnxYoloWrapper 的壳反而只剩"框的个数"
            moxing = yuan
            leibie = (yuan.config or {}).get("classes", [])
            if isinstance(leibie, dict):
                leibie = list(leibie.values())
            leibie_ming = {i: str(m) for i, m in enumerate(leibie)}
            yao = str(canyu.get("leibie") or "").strip()
            if yao:
                fan = {
                    str(ming).lower(): xu
                    for xu, ming in leibie_ming.items()
                }
                for yi in re.split(r"[,，\s]+", yao):
                    yi = yi.strip().lower()
                    if yi and yi in fan:
                        leibie_id.append(fan[yi])
                logger.info(f"视频工作台：只检测「{yao}」 -> id {leibie_id}")

        ocr_moxing = None
        if canyu.get("ocr"):
            yuan = self._qu_moxing("ocr", canyu.get("ocr_lu") or "")
            if yuan is None or not hasattr(yuan, "text_system"):
                return None, None, [], {}, "OCR 模型加载失败，看看 YAML 选对没有"
            ocr_moxing = PPOCRv6Wrapper(yuan.text_system)

        return moxing, ocr_moxing, leibie_id, leibie_ming, ""

    def _qu_moxing(self, yongtu, yaml_lu):
        """拿模型：加载过就复用，只有换了 YAML 才重新加载

        yongtu 是 "jiance" / "ocr"，同一个 YAML 当检测模型和当 OCR 模型是两回事，
        所以 key 里带上用途。加载失败不进缓存，下次还会重试。
        """
        from anylabeling.views.labeling.utils.video import (
            _load_model_from_yaml,
        )

        yao = (yongtu, osp.abspath(yaml_lu) if yaml_lu else "")
        if yao in self._moxing_huancun:
            logger.info(f"视频工作台：复用已加载的模型 {osp.basename(yaml_lu)}")
            return self._moxing_huancun[yao]
        yuan = _load_model_from_yaml(yaml_lu)
        if yuan is not None:
            self._moxing_huancun[yao] = yuan
        return yuan

    def _qidong_tuili_qu(self):
        """跟随播放做推理：另开一路解码专门喂模型

        画面那一路归 mpv，这一路不掺和：只读原图、不转色、不缩放、不留缓存，
        跟着 mpv 的播放时间往前跟，别跑到画面前面去。
        """
        self._tingzhi_tuili_qu()
        qu = _ZhenDuQu(self.lujing, self.fps, self)
        qu.zhi_tuili = True
        qu.zhen_hook = self._duqu_hook
        qu.pace_hook = self._tuili_pace_miao
        self.tuili_qu = qu
        qu.start()
        qu.tiaozheng(int(round(self._bofang_ms / 1000.0 * self.fps)))
        qu.jixu()

    def _tuili_pace_miao(self):
        """mpv 现在播到第几秒（给喂模型那一路上做节拍用）"""
        if self.mpv.h is None:
            return None
        return self.mpv.shijian_ms() / 1000.0

    def _tingzhi_tuili_qu(self):
        if self.tuili_qu is not None:
            self.tuili_qu.zhen_hook = None
            self.tuili_qu.tingzhi()
            self.tuili_qu = None

    def _duqu_hook(self, zhen_bgr, zhen_hao):
        """取帧线程每读一帧就调这里（跟随播放时把原图丢给推理线程）"""
        if self.yunsuan is None or self.yunsuan_moshi != "gensui":
            return
        self.yunsuan.fang_zhen(zhen_bgr, zhen_hao)

    def _shoudao_saomiao_huamian(self, zhen_rgb, zhen_hao, kuang):
        """全片扫描时把当前这一帧贴到画面上，框一起画"""
        self._shoudao_zhen(zhen_rgb, zhen_hao)
        self.huamian.shezhi_kuang(kuang)

    def _shoudao_tuili_jieguo(self, zhen_hao, kuang, xin_zimu):
        """一帧处理完：画框 + 记字幕段"""
        self.huamian.shezhi_kuang(kuang)
        if xin_zimu is not None:
            self._jia_zimu_kuai(*xin_zimu)
        if self.chuliqi is None:
            return
        if self.yunsuan_moshi == "gensui":
            self.shezhi_mianban.zhuangtai_shezhi(
                f"跟随播放中 · 本帧 {len(kuang or [])} 个目标 · "
                f"已存 {self.chuliqi.yi_cun} 张"
            )

    # ---- 字幕编辑页的列表：样式 / 说话人从打开时那份 ASS 里认 ----
    def _ass_qing_hang(self):
        """打开的那份 ASS 每行拆好留着：
        [（去标签的文字, 起ms, 止ms, 样式, 说话人, 原文, 字段表）]

        整份列表刷一次要认几百行，每行都重新去标签太费，拆一次存着用。
        "原文"是带行内标签（\\pos \\an \\fs …）的原样文字，画画面上的字幕要用；
        "字段表"是那一行的层 / 边距 / 特效 / 注释，编辑区上方那排要显示和改。
        """
        jiegou = self._ass_yuan
        if jiegou is None:
            return []
        jilu = getattr(self, "_ass_qing_jilu", None)
        if jilu is not None and jilu[0] is jiegou:
            return jilu[1]
        hang = [
            (
                _ass_chun_wen(h.get("Text")),
                _ass_hao_ms(h.get("Start")),
                _ass_hao_ms(h.get("End")),
                str(h.get("Style") or ""),
                str(h.get("Name") or ""),
                str(h.get("Text") or ""),
                _ass_hang_fu(h),
            )
            for h in (jiegou.get("hang") or [])
        ]
        self._ass_qing_jilu = (jiegou, hang)
        return hang

    def _zimu_pipei(self, zimu):
        """每条字幕配上原来那份 ASS 里的（样式, 说话人, 原文, 字段表）

        认法跟写 ASS 时一模一样：先按文字认原来那一行，文字改过就按原时间认；
        认不到（新加的行，或者根本没打开过 ASS）就是 Default / 空 / 空 / 空。
        """
        an_wen, an_shi = {}, {}
        for chun, qi, zhi, yang, shuo, yuan, fu in self._ass_qing_hang():
            an_wen.setdefault(chun, []).append((yang, shuo, yuan, fu))
            an_shi.setdefault((qi, zhi), []).append((yang, shuo, yuan, fu))
        jieguo = []
        for qi, zhi, wenben in zimu:
            chun = _ass_chun_wen(wenben)
            dai = an_wen.get(chun) or an_shi.get((int(qi), int(zhi)))
            jieguo.append(dai.pop(0) if dai else ("Default", "", "", {}))
        return jieguo

    def _zimu_fujia(self, zimu):
        """字幕编辑页列表的「样式 / 说话人」两列 + 编辑区上方那排要的字段

        原 ASS 里那一行是什么就带什么，编辑区上方改过的（存在 zimu_mianban 里）
        盖在上面，粘贴进来时外面直接指定的（_fu_wai）垫在底下。算出来的这份
        顺手缓存成 _zimu_pei_ji / _zimu_fu_ji，画画面和填那排工具都直接拿去用。
        """
        pei = self._zimu_pipei(zimu)
        self._zimu_pei_ji = pei
        wai = getattr(self, "_fu_wai", None) or {}
        fu_men = []
        for i, (yang, shuo, _yuan, fu) in enumerate(pei):
            ge = dict(fu or {})
            ge["yang"] = str(yang or "Default")
            ge["shuo"] = str(shuo or "")
            ge.update(wai.get(tuple(zimu[i])) or {})
            ge.update(self.zimu_mianban.hang_fu(i))
            fu_men.append(ge)
        self._zimu_fu_ji = fu_men
        return fu_men

    def _zimu_yangshi_ming(self):
        """打开的那份 ASS 里所有样式名（原大小写，给样式下拉当候选）"""
        jiegou = self._ass_yuan
        if jiegou is None:
            return []
        return _ass_yangshi_ming(jiegou.get("tou") or [])

    def _zimu_shuo_ming(self):
        """说话人的候选：原 ASS 里出现过的那些"""
        ming = []
        for _chun, _qi, _zhi, _yang, shuo, _yuan, _fu in self._ass_qing_hang():
            shuo = str(shuo or "").strip()
            if shuo and shuo not in ming:
                ming.append(shuo)
        return ming

    def _zimu_yangshi_ming_quan(self):
        """样式名候选：ASS 样式表里的 + 列表里实际用到过的（合并去重）"""
        ming = list(self._zimu_yangshi_ming())
        for yang, _shuo, _yuan, _fu in self._zimu_pipei(
            self.zimu_mianban.zimu_liebiao()
        ):
            yang = str(yang or "").strip()
            if yang and yang not in ming:
                ming.append(yang)
        if not ming:
            ming.append("Default")
        return ming

    def _ass_geshi_biao(self):
        """打开的那份 ASS 的基准分辨率 + 样式表（读一次存着）"""
        jiegou = self._ass_yuan
        if jiegou is None:
            return HUA_ZIMU_JIZHUN, {}
        ji = self._zimu_hua_ji
        if ji is not None and ji[0] is jiegou:
            return ji[1], ji[2]
        play, biao = _du_ass_yangshi(jiegou.get("tou") or [])
        play = play or HUA_ZIMU_JIZHUN
        self._zimu_hua_ji = (jiegou, play, biao)
        return play, biao

    def _huamian_ms(self):
        """画面上那层字幕按哪个时刻算：一律认 mpv 现在真正显示的那一帧

        拖动 / 左右跳帧的时候画面是 mpv 解出来的，比鼠标位置晚几百毫秒。
        字幕要是按鼠标位置画，就跑到画面前头去了（看着就是"字幕先切、画面
        后到"）。跟画面同一个时刻算，画面和字幕就一块儿出来。
        mpv 还没起来（没开视频）才退回界面上那个位置。
        """
        if self.duqu is not None:
            try:
                return int(self.mpv.shijian_ms())
            except Exception:  # noqa
                pass
        return int(self._bofang_ms)

    def _shuaxin_huamian_zimu(self, ms=None):
        """把播放头这一刻该显示的字幕塞给画面（样式照 ASS 来的）

        画面上该有的就是"播放头现在压着的这几条"，摞在一起的也都画。
        注释行不画；样式 / 边距都认编辑区上方改过的那份。
        ms 不给就按 mpv 现在显示的那一帧算（见 _huamian_ms）。
        """
        hua = getattr(self, "huamian", None)
        if hua is None:
            return
        zimu = self.zimu_mianban.zimu_liebiao()
        if not zimu:
            hua.shezhi_zimu([])
            return
        fu_men = self._zimu_fu_ji
        if len(fu_men) != len(zimu):
            fu_men = self._zimu_fujia(zimu)
        pei = self._zimu_pei_ji
        if len(pei) != len(zimu):
            pei = self._zimu_pipei(zimu)
        jizhun, biao = self._ass_geshi_biao()
        ms = int(self._huamian_ms() if ms is None else ms)
        xuan = []
        for i, (qi, zhi, wenben) in enumerate(zimu):
            if not (int(qi) <= ms < max(int(qi) + 1, int(zhi))):
                continue
            fu = fu_men[i] if i < len(fu_men) else {}
            if fu.get("zhushi"):
                continue                    # 注释行不画在画面上
            yang = str(fu.get("yang") or "Default")
            yuan = pei[i][2] if i < len(pei) else ""
            ge = dict(_ASS_MOREN_YANGSHI)
            zhao = biao.get(yang.strip().lower())
            if zhao:
                ge = dict(zhao)
            # 行边距：只有写了数才盖样式的边距（0 就是"照样式来"，跟 libass 一样）
            for jian, wei in (("ml", "zuo"), ("mr", "you"), ("mv", "shu")):
                zhi_shu = int(fu.get(wei) or 0)
                if zhi_shu > 0:
                    ge[jian] = zhi_shu
            # 整条（文字 + 行内标签）跟原文一模一样才照原文来；只要动过 ——
            # 改了字，或者上面那排加了 \1c \b \fn 这些标签 —— 就按列表里这份画。
            # （只比"纯文字"会把只加标签的改动当成没改，改完颜色画面不动）
            wen = (
                str(yuan or "")
                if yuan and _ass_yuan_wen(yuan) == _ass_yuan_wen(wenben)
                else str(wenben or "")
            )
            hang, gs = _jie_ass_tiao(wen, ge, biao)
            # qi / zhi / ms 是给 \fad 渐入渐出算这一刻透明度用的
            xuan.append(
                {
                    "hang": hang,
                    "gs": gs,
                    "qi": int(qi),
                    "zhi": int(zhi),
                    "ms": ms,
                    "ceng": int(fu.get("ceng") or 0),
                }
            )
            if len(xuan) >= 8:
                break
        # 层大的盖在上面（跟 libass 一个规矩）
        xuan.sort(key=lambda x: x["ceng"])
        hua.shezhi_zimu(xuan, jizhun)

    def _shezhi_bianji_zimu(self, zimu, xuan=None):
        """刷「字幕编辑」页的列表 + 编辑区上方那排

        条数、时间、文字、样式、说话人、字段都一起给过去；画面里那条的缓存
        也跟着作废（_zimu_fu_ji / _zimu_pei_ji 就是缓存本身，这里重算）。
        """
        fu_men = self._zimu_fujia(zimu)
        self._zimu_fu_ji = fu_men
        self.bianji_mianban.shezhi_yangshi_ming(self._zimu_yangshi_ming())
        self.bianji_mianban.shezhi_shuo_ming(self._zimu_shuo_ming())
        self.bianji_mianban.shezhi_zimu(zimu, xuan, fu_men)
        self._shuaxin_gongju(zimu, fu_men)
        self._shuaxin_kuai_yanse(zimu, fu_men)

    def _shuaxin_kuai_yanse(self, zimu=None, fu_men=None):
        """给时间轴上每块字幕算底色：说话人非空 -> 用它那个样式的主文字色

        说话人为空 / 样式名在样式表里查不到 -> None，时间轴那边用默认蓝。
        说话人和样式是一起套的（槽位），所以按说话人认就等于按样式认。
        """
        if zimu is None:
            zimu = self.zimu_mianban.zimu_liebiao()
        if fu_men is None:
            fu_men = self._zimu_fu_ji
            if len(fu_men) != len(zimu):
                fu_men = self._zimu_fujia(zimu)
        _jizhun, biao = self._ass_geshi_biao()
        men = []
        for i in range(len(zimu)):
            fu = fu_men[i] if i < len(fu_men) else {}
            men.append(self._kuai_yanse_yise(fu, biao))
        self.shijianzhou.shezhi_kuai_yanse(men)

    def _kuai_yanse_yise(self, fu, biao):
        """一条字幕该用什么底色：没说话人 / 样式查不到 -> None（用默认蓝）"""
        if not str(fu.get("shuo") or "").strip():
            return None
        ge = biao.get(str(fu.get("yang") or "Default").strip().lower())
        if not ge:
            return None
        se = str(ge.get("c1") or "").strip()
        if len(se) != 7 or not se.startswith("#"):
            return None
        yanse = QtGui.QColor(se)
        yanse.setAlpha(ZIMU_TOUMING_DU)
        return yanse

    def _shuaxin_gongju(self, zimu=None, fu_men=None):
        """把当前这条的字段 + 实际生效的样式填进编辑区上方那排"""
        if zimu is None:
            zimu = self.zimu_mianban.zimu_liebiao()
        if fu_men is None:
            fu_men = self._zimu_fu_ji
            if len(fu_men) != len(zimu):
                fu_men = self._zimu_fujia(zimu)
        xu = self.bianji_mianban.dangqian_xu()
        if not (0 <= xu < len(zimu)):
            self.bianji_mianban.shezhi_gongju({}, None)
            return
        fu = dict(fu_men[xu]) if xu < len(fu_men) else {}
        fu["qi"], fu["zhi"] = zimu[xu][0], zimu[xu][1]
        self.bianji_mianban.shezhi_gongju(
            fu, self._hang_gs(xu, zimu, fu_men), str(zimu[xu][2] or "")
        )

    def _hang_gs(self, xu, zimu=None, fu_men=None):
        """第 xu 条实际生效的格式（样式底子 + 行内标签）

        编辑区上方那排的 B/I/U/S 按钮和四个色块显示的就是这个。
        """
        if zimu is None:
            zimu = self.zimu_mianban.zimu_liebiao()
        if fu_men is None:
            fu_men = self._zimu_fu_ji
        if not (0 <= xu < len(zimu)):
            return {}
        _jizhun, biao = self._ass_geshi_biao()
        fu = fu_men[xu] if xu < len(fu_men) else {}
        yang = str(fu.get("yang") or "Default")
        ge = dict(biao.get(yang.strip().lower()) or _ASS_MOREN_YANGSHI)
        pei = self._zimu_pei_ji
        if len(pei) != len(zimu):
            pei = self._zimu_pipei(zimu)
        yuan = pei[xu][2] if xu < len(pei) else ""
        _qi, _zhi, wenben = zimu[xu]
        # 跟画面里一个规矩：整条（文字 + 行内标签）都没动过才照原文来
        wen = (
            str(yuan or "")
            if yuan and _ass_yuan_wen(yuan) == _ass_yuan_wen(wenben)
            else str(wenben or "")
        )
        _hang, gs = _jie_ass_tiao(wen, ge, biao)
        return gs

    def _jia_zimu_kuai(self, qi_ms, zhi_ms, wenben):
        """新建一条字幕：三处（硬字幕列表 / 底部时间轴 / 字幕编辑列表）一起加

        返回新的一条插在第几行 —— 是按时间插的，不一定是最后一行。
        """
        wei = self.zimu_mianban.tianjia_zimu(qi_ms, zhi_ms, wenben)
        self.shijianzhou.tianjia_zimu(qi_ms, zhi_ms, wenben)
        self._shezhi_bianji_zimu(self.zimu_mianban.zimu_liebiao())
        self._ji_chexiao("新建一条", "xinjian")
        return wei

    def _shoudao_xuan_zhong_zimu(self, xu):
        """字幕列表 ↔ 时间轴 ↔ 字幕编辑：选中同一段，几处高亮对齐，时间轴滚过去"""
        self.shijianzhou.shezhi_xuan_zhong(xu)
        self.zimu_mianban.shezhi_xuan_zhong(xu)
        self.bianji_mianban.shezhi_xuan_zhong(xu)
        self._shuaxin_gongju()
        self.shijianzhou.gundong_dao_zimu(xu)

    def _shoudao_xuan_zhong_duo(self, hang):
        """字幕列表里多选 / 全选：时间轴整批跟着高亮（不动播放头）"""
        if len(hang) < 2:
            return
        self.shijianzhou.shezhi_xuan_zhong_duo(hang)

    def _shoudao_kuang_xuan_kuai(self, hang):
        """时间轴上框选了一批字幕块：字幕列表整批跟着选中（跟列表里多选一回事）"""
        hang = [int(x) for x in (hang or [])]
        if not hang:
            return
        self.bianji_mianban.shezhi_xuan_zhong_duo(hang)

    # ---- 改字幕 ----
    def _shoudao_gongju(self, jian, zhi):
        """编辑区上方那排改了东西 -> 落到当前这一条字幕上

        行内标签（B/I/U/S/fn/四个颜色）是直接改进文字里那一对花括号；样式 /
        说话人 / 层 / 边距 / 特效 / 注释是那一行的字段；时间走改时间那条路。
        """
        if jian == "moshi":
            # 上面那排切了「时间 / 帧」：字幕列表的时间两列跟着换
            # （这是全局显示设置，跟有没有选中字幕无关，所以放在最前）
            self.bianji_mianban.shezhi_xianshi_moshi(
                str(zhi or "") == "zhen"
            )
            return
        if jian == "gen_tiao":
            # 「选中字幕时画面跟着跳」：全局开关，跟当前有没有选中哪条无关
            self._gen_tiao = bool(zhi)
            for ming in ("bianji_mianban", "zimu_mianban"):
                mian = getattr(self, ming, None)
                if mian is not None and hasattr(mian, "shezhi_zidong_tiao"):
                    mian.shezhi_zidong_tiao(self._gen_tiao)
            self._cun_kaiguan(JIEMIAN_GEN_TIAO, self._gen_tiao)
            return
        if jian == "shishi_gun":
            # 「时间轴实时滚动」：全局开关
            self._shishi_gun = bool(zhi)
            self.shijianzhou.shezhi_shishi_gun(self._shishi_gun)
            self._cun_kaiguan(JIEMIAN_SHISHI_GUN, self._shishi_gun)
            return
        xu = self.bianji_mianban.dangqian_xu()
        zimu = self.zimu_mianban.zimu_liebiao()
        if jian == "zhushi":
            # 列表里多选着的时候勾「注释」：一次把选中的这几条全设成注释
            # （Ctrl+A 全选再打勾 = 把整屏字幕全藏了）
            duo = self.bianji_mianban.xuan_zhong_hang()
            if len(duo) >= 2:
                self._piliang_zhushi(duo, bool(zhi))
                return
        if not (0 <= xu < len(zimu)):
            return
        if jian == "bianji":
            self._kai_yangshi_bianji()
            return
        if jian == "xiayihang":
            self._tiao_xiayitiao()
            return
        if jian == "tag":
            ming, can = zhi[0], zhi[1]
            xuan = zhi[2] if len(zhi) > 2 else None
            wen = str(zimu[xu][2] or "")
            if xuan and ming in ("fn", "b", "i", "u", "s"):
                # 编辑框里选中了一段：只给这一段插标签（照 AEG）
                # B/I/U/S 尾巴上用反向标签（{\b1}…{\b0}），这样就掀不掉前面
                # 已经打好的颜色；字体用 {\r} 截断。
                if ming == "fn":
                    xin = _wei_yiduan(wen, xuan, "{\\fn" + str(can) + "}", "{\\r}")
                else:
                    kai, fan = ("1", "0") if can else ("0", "1")
                    xin = _wei_yiduan(
                        wen, xuan,
                        "{\\%s%s}" % (ming, kai),
                        "{\\%s%s}" % (ming, fan),
                    )
            elif ming == "fn":
                xin = _jia_biaoqian(wen, f"\\fn{can}")
            elif can:
                xin = _jia_biaoqian(wen, f"\\{ming}1")
            else:
                xin = _qu_biaoqian(wen, ming)
            self._huan_zimu_wenben(xu, xin)
            return
        if jian == "yanse":
            wei, se = zhi[0], zhi[1]
            xuan = zhi[2] if len(zhi) > 2 else None
            # 主色写 \c（Aegisub 那四个色块就是这么写的），其它三个照 ASS 写
            bian = {"c1": "\\c", "c2": "\\2c", "c3": "\\3c", "c4": "\\4c"}
            tou = bian.get(wei) or bian["c1"]
            wen = str(zimu[xu][2] or "")
            if xuan:
                # 选中的那一段上色，尾巴 {\r} 截断 —— 后面的字还是原来的颜色
                xin = _wei_yiduan(wen, xuan, "{" + tou + str(se) + "}", "{\\r}")
            else:
                xin = _qu_biaoqian(wen, wei)
                xin = _jia_biaoqian(xin, tou + str(se))
            self._huan_zimu_wenben(xu, xin)
            return
        if jian in ("qi", "zhi"):
            qi, zhi_ms = int(zimu[xu][0]), int(zimu[xu][1])
            if jian == "qi":
                qi = max(0, int(zhi))
            else:
                zhi_ms = max(0, int(zhi))
            if zhi_ms < qi:
                qi, zhi_ms = zhi_ms, qi
            self._shoudao_zimu_tuo(xu, qi, zhi_ms)
            return
        # 剩下的都是那一行的字段
        fu = dict(self.bianji_mianban.hang_fu(xu))
        fu.update(self.zimu_mianban.hang_fu(xu))
        fu[jian] = zhi
        self.zimu_mianban.shezhi_hang_fu(xu, fu)
        self._shezhi_bianji_zimu(self.zimu_mianban.zimu_liebiao(), xu)
        self._zimu_tiao_wen = None
        self._shuaxin_huamian_zimu()
        xie = self._chong_xie_zimu_wenjian("改字段", "gongju")
        self.shezhi_mianban.zhuangtai_shezhi(
            "字幕已改" + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _piliang_zhushi(self, hang, kai):
        """把选中的这几条一起设成注释 / 取消注释（等于 AEG 全选后打勾）

        注释行存进 ASS 是 Comment 行，画面上不显示 —— 想整屏藏字幕就
        Ctrl+A 全选再打勾；只藏一条就单选后打勾（走原来那条路）。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        hang = [int(x) for x in (hang or []) if 0 <= int(x) < len(zimu)]
        if not hang:
            return
        for i in hang:
            fu = dict(self.bianji_mianban.hang_fu(i))
            fu.update(self.zimu_mianban.hang_fu(i))
            fu["zhushi"] = bool(kai)
            self.zimu_mianban.shezhi_hang_fu(i, fu)
        self._shezhi_bianji_zimu(zimu, self.bianji_mianban.dangqian_xu())
        # 整表重建会把多选冲掉，重新选回原来这一批
        self.bianji_mianban.shezhi_xuan_zhong_duo(hang)
        self._zimu_tiao_wen = None
        self._shuaxin_huamian_zimu()
        xie = self._chong_xie_zimu_wenjian("批量注释")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已把 {len(hang)} 条设成"
            + ("注释（画面上不显示）" if kai else "普通字幕（照常显示）")
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _tao_caowei(self, hao, hang=None):
        """套槽位：把选中的字幕的「说话人 + 样式」设成第 hao 个槽位

        F1~F12 和字幕列表右键菜单都走这里；槽位在样式编辑器的
        「自动化脚本 ▸ 设置说话人+样式 配置管理」里配。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        if hang is None:
            hang = self.bianji_mianban.xuan_zhong_hang()
        hang = [int(x) for x in (hang or []) if 0 <= int(x) < len(zimu)]
        if not hang:
            self.shezhi_mianban.zhuangtai_shezhi("先选中要套槽位的字幕")
            return
        cao = du_zidonghua_peizhi().get("caowei") or []
        if not (0 <= hao < len(cao)):
            return
        shuo = str(cao[hao].get("shuohua") or "")
        yang = str(cao[hao].get("yangshi") or "Default")
        if not shuo and yang == "Default":
            self.shezhi_mianban.zhuangtai_shezhi(
                f"槽位{hao + 1:02d} 还没配"
                "（样式编辑器 ▸ 自动化脚本 ▸ 设置说话人+样式）"
            )
            return
        for i in hang:
            fu = dict(self.bianji_mianban.hang_fu(i))
            fu.update(self.zimu_mianban.hang_fu(i))
            fu["shuo"] = shuo
            fu["yang"] = yang
            self.zimu_mianban.shezhi_hang_fu(i, fu)
        self._shezhi_bianji_zimu(zimu, self.bianji_mianban.dangqian_xu())
        # 整表重建会把多选冲掉，重新选回原来这一批
        self.bianji_mianban.shezhi_xuan_zhong_duo(hang)
        self._zimu_tiao_wen = None
        self._shuaxin_huamian_zimu()
        xie = self._chong_xie_zimu_wenjian("套槽位")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"槽位{hao + 1:02d} 套到 {len(hang)} 条："
            f"说话人「{shuo or '（空）'}」样式「{yang}」"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _zhixing_kuohao(self, hang):
        """【-批量添加方括号】：给选中字幕的文字套上「」

        照 AEG 那个 Lua 脚本的规矩，这三种跳过：
          说话人 / 样式里含排除关键词的
          文本本身已经是（…）/ (…) 的
          文本里已经有排除符号的（「」『』（）() …）
        开头的特效标签（花括号包起来那种）留在引号外面，标签原样不动。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        hang = [int(x) for x in (hang or []) if 0 <= int(x) < len(zimu)]
        if not hang:
            self.shezhi_mianban.zhuangtai_shezhi("先选中要加「」的字幕")
            return
        pei = du_zidonghua_peizhi()
        guan = [str(x) for x in (pei.get("paichu_ci") or []) if str(x)]
        fuhao = [str(x) for x in (pei.get("paichu_fuhao") or []) if str(x)]
        fu_men = self._zimu_fu_ji
        if len(fu_men) != len(zimu):
            fu_men = self._zimu_fujia(zimu)

        gai = 0
        for i in hang:
            fu = fu_men[i] if i < len(fu_men) else {}
            shuo = str(fu.get("shuo") or "")
            yang = str(fu.get("yang") or "")
            if any(k in shuo or k in yang for k in guan):
                continue        # 旁白之类的：不套
            jiu = str(zimu[i][2] or "")
            tou = re.match(r"^(\{[^}]*\})", jiu)
            biaoqian = tou.group(1) if tou else ""
            zhengwen = jiu[len(biaoqian):]
            chun = re.sub(r"\{[^}]*\}", "", zhengwen)
            if not chun.strip():
                continue        # 只有标签 / 空行
            if re.match(r"^（.+）$", chun) or re.match(r"^\(.+\)$", chun):
                continue        # 已经是括号了
            if any(h in chun for h in fuhao):
                continue        # 已经有那些符号了
            xin = biaoqian + "「" + zhengwen + "」"
            self.zimu_mianban.gai_zimu_wenben(i, xin)
            self.shijianzhou.gai_zimu_wenben(i, xin)
            gai += 1

        if not gai:
            self.shezhi_mianban.zhuangtai_shezhi(
                f"这 {len(hang)} 条都不用加（旁白 / 已是括号 / 已有那些符号）"
            )
            return
        self._shezhi_bianji_zimu(
            self.zimu_mianban.zimu_liebiao(), self.bianji_mianban.dangqian_xu()
        )
        # 整表重建会把多选冲掉，重新选回原来这一批
        self.bianji_mianban.shezhi_xuan_zhong_duo(hang)
        self._zimu_hang_ji = None
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        xie = self._chong_xie_zimu_wenjian("括号排除")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已给 {gai} 条套上「」"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _qiehuan_zhushi(self):
        """Alt+S：把当前这条（列表里多选就这一批）在「注释 / 普通」之间翻一下

        直接翻那个「注释」勾选框 —— 它的 toggled 早连着整条链路（单选改一条、
        多选整批改、刷列表和画面、重写 SRT / ASS），不用再走一遍。
        """
        self.bianji_mianban.gongjulan.gou_zhushi.toggle()

    def _huan_zimu_wenben(self, xu, xin):
        """点按钮改出来的文字（加了 / 去了行内标签）：几处一起同步再写回"""
        zimu = self.zimu_mianban.zimu_liebiao()
        if not (0 <= xu < len(zimu)) or str(zimu[xu][2]) == str(xin):
            return
        self.zimu_mianban.gai_zimu_wenben(xu, xin)
        self.shijianzhou.gai_zimu_wenben(xu, xin)
        self._shezhi_bianji_zimu(self.zimu_mianban.zimu_liebiao(), xu)
        # 文字 / 标签都动了：折行缓存作废，画面和时间轴上的块立刻重画
        self._zimu_hang_ji = None
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        xie = self._chong_xie_zimu_wenjian("改文字", "wenben")
        self.shezhi_mianban.zhuangtai_shezhi(
            "字幕已改" + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _shoudao_xiayitiao(self):
        """焦点在时间轴上按 Ins：换到下一条字幕（跟字幕列表回车一个走法）"""
        self.bianji_mianban.xia_yi_tiao()

    def _tiao_xiayitiao(self):
        """工具栏那个 ✓：跳到下一条；已经是最后一条就在它后面接一条新的"""
        zimu = self.zimu_mianban.zimu_liebiao()
        xu = self.bianji_mianban.dangqian_xu()
        if not (0 <= xu < len(zimu)):
            return
        if xu + 1 < len(zimu):
            self._shoudao_xuan_zhong_zimu(xu + 1)
            return
        qi = int(zimu[xu][1])
        chang = max(300, int(zimu[xu][1]) - int(zimu[xu][0]))
        xin = self._jia_zimu_kuai(qi, qi + chang, "")
        self._shoudao_xuan_zhong_zimu(xin)
        self.shezhi_mianban.zhuangtai_shezhi(
            f"末尾新加一条 · {self._shijian_wenben(qi)} → "
            f"{self._shijian_wenben(qi + chang)}"
        )

    def _kai_yangshi_bianji(self):
        """点了「编辑」：弹样式编辑器，改当前这条用的那个样式

        改的时候先只改内存里的样式表，画面立刻能看到效果；点「确定 / 应用」
        才写回 ASS。点「取消」就把这次改的全退回去，文件也跟着退回去。
        """
        xu = self.bianji_mianban.dangqian_xu()
        zimu = self.zimu_mianban.zimu_liebiao()
        if not (0 <= xu < len(zimu)):
            self.shezhi_mianban.zhuangtai_shezhi("先选一条字幕，再点编辑")
            return
        jiu_chuang = getattr(self, "_yangshi_chuang", None)
        if jiu_chuang is not None:
            jiu_chuang.showNormal()
            jiu_chuang.raise_()
            jiu_chuang.activateWindow()
            return
        fu_men = self._zimu_fu_ji
        if len(fu_men) != len(zimu):
            fu_men = self._zimu_fujia(zimu)
        fu = fu_men[xu] if xu < len(fu_men) else {}
        yang = str(fu.get("yang") or "Default")
        _jizhun, biao = self._ass_geshi_biao()
        ge = dict(biao.get(yang.strip().lower()) or _ASS_MOREN_YANGSHI)
        ge["name"] = yang
        jiu = dict(ge)
        dlg = YangshiBianjiDialog(
            ge, self, _ziti_ming_men(), _jizhun,
            # 预览里造字体跟画面里同一个函数：字号 / 粗档 / 字距全都一样
            lambda zs: _hua_ziti(zs, 1.0),
            # 下面「自动化脚本」里的槽位配置要拿样式名当候选
            yang_men=self._zimu_yangshi_ming_quan(),
        )
        self._yangshi_chuang = dlg
        dlg.gaile.connect(lambda z: self._yulan_yangshi(z, yang))
        dlg.queren.connect(lambda z: self._luo_yangshi(z, yang))

        def _guan_le(jieguo):
            # 非模态窗口：确定之外（取消 / 点 X 关掉）都把这次改的样式退回去
            self._yangshi_chuang = None
            if jieguo != QtWidgets.QDialog.Accepted:
                self._luo_yangshi(jiu, yang, tui=True)

        dlg.finished.connect(_guan_le)
        dlg.show()

    def _yulan_yangshi(self, zi, yang):
        """样式编辑器里改一下：先只改内存里的样式表，画面上立刻能看到"""
        if self._ass_yuan is None:
            return
        tou = list(self._ass_yuan.get("tou") or [])
        if not _hui_yangshi_tou(tou, yang, zi):
            return
        self._ass_yuan["tou"] = tou
        self._zimu_hua_ji = None
        self._shuaxin_huamian_zimu()
        self._shuaxin_kuai_yanse()   # 颜色实时改就实时换，时间轴上的块跟着变

    def _luo_yangshi(self, zi, yang, tui=False):
        """「确定 / 应用」或「取消」：写回 [V4+ Styles]，重画 + 重写文件"""
        if self._ass_yuan is None:
            return
        tou = list(self._ass_yuan.get("tou") or [])
        if not _hui_yangshi_tou(tou, yang, zi):
            self.shezhi_mianban.zhuangtai_shezhi(
                f"这份字幕里没找到样式「{yang}」，没写回去"
            )
            return
        self._ass_yuan["tou"] = tou
        self._zimu_hua_ji = None
        self._shuaxin_huamian_zimu()
        self._shuaxin_gongju()
        self._shuaxin_kuai_yanse()   # 样式颜色改了就换过去，时间轴上的块跟着变色
        xie = self._chong_xie_zimu_wenjian("改样式")
        self.shezhi_mianban.zhuangtai_shezhi(
            (f"样式「{yang}」已还原" if tui else f"样式「{yang}」已改")
            + ("，SRT / ASS 已重写" if xie else "")
        )

    def _shoudao_zimu_zhengzai_gai(self, xu, wenben):
        """字幕编辑区正在打字：把这一版文字落到选中的每一行，不重建列表、不写盘

        跟 AEG 一样：选了多行（或 Ctrl+A 全选）时，在编辑区里改这一版文字，
        选中的每一行都跟着改成同一版；把文字全删了，选中的行就一起清空。
        每敲一个字就刷整个列表 + 重写一遍 SRT / ASS 会卡得没法打字，所以这里
        只动内存里的那几行和它们画出来的样子；落盘留给编辑框失焦那次。
        """
        mu = self.bianji_mianban.xuan_zhong_hang()
        xu = int(xu)
        if len(mu) <= 1 or xu not in mu:
            mu = [xu]
        for i in mu:
            self.zimu_mianban.gai_zimu_wenben(i, wenben)
            self.shijianzhou.gai_zimu_wenben(i, wenben)
            self.bianji_mianban.gai_wenben_ji(i, wenben)
        self._zimu_hang_ji = None
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()

    def _shoudao_zimu_bianji_wancheng(self, xu, _wenben):
        """编辑框失焦：这一版文字改完了 —— 把 SRT / ASS 重写一遍

        编辑区里的改动是整批落到选中行上的，这里报一下这次改了几行。
        """
        mu = self.bianji_mianban.xuan_zhong_hang()
        ji = f"{len(mu)} 行" if len(mu) > 1 else f"第 {int(xu) + 1} 条"
        xie = self._chong_xie_zimu_wenjian("改文字", "wenben")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"字幕已改 · {ji}"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _shoudao_zimu_xiugai(self, xu, _wenben):
        """某条字幕的文字改了（弹框改 / 编辑区改完）：时间轴、列表一起刷，
        SRT / ASS 重写一遍"""
        self.shijianzhou.shezhi_zimu(self.zimu_mianban.zimu_liebiao())
        self.shijianzhou.shezhi_xuan_zhong(int(xu))
        self.zimu_mianban.shezhi_xuan_zhong(int(xu))
        self._shezhi_bianji_zimu(self.zimu_mianban.zimu_liebiao(), int(xu))
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        xie = self._chong_xie_zimu_wenjian("改文字", "wenben")
        self.shezhi_mianban.zhuangtai_shezhi(
            "字幕已改" + ("，SRT / ASS 已重写" if xie else "")
        )

    def _shoudao_zimu_tuo(self, xu, qi_ms, zhi_ms):
        """底部时间轴上把字幕块拖完了 -> 同步右侧列表 / 迷你轴，重写提取目录的字幕"""
        self.zimu_mianban.gai_zimu_shijian(xu, qi_ms, zhi_ms)
        self.shijianzhou.shezhi_zimu(self.zimu_mianban.zimu_liebiao())
        self.shijianzhou.shezhi_xuan_zhong(int(xu))
        self.zimu_mianban.shezhi_xuan_zhong(int(xu))
        self._shezhi_bianji_zimu(self.zimu_mianban.zimu_liebiao(), int(xu))
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        xie = self._chong_xie_zimu_wenjian("改时间", "tuo")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"字幕时间已改 · {self._shijian_wenben(qi_ms)} → "
            f"{self._shijian_wenben(zhi_ms)}"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _shoudao_zimu_nuo(self, gai):
        """方向键挪了选中的字幕块 -> 批量改内容，一次同步、一次重写文件

        跟上面拖块那条路不一样：拖块一次只动一条，这里是整组一起挪，
        逐条发信号会重写好几次文件，白白卡顿。
        """
        if not gai:
            return
        bao = self.shijianzhou.xuan_zhong_liebiao()
        for xu, qi_ms, zhi_ms in gai:
            self.zimu_mianban.gai_zimu_shijian(int(xu), int(qi_ms), int(zhi_ms))
        self.shijianzhou.shezhi_zimu(self.zimu_mianban.zimu_liebiao())
        if bao:
            self.zimu_mianban.shezhi_xuan_zhong(bao[0])
        self._shezhi_bianji_zimu(
            self.zimu_mianban.zimu_liebiao(), bao[0] if bao else -1
        )
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        xie = self._chong_xie_zimu_wenjian("改时间", "nuo")
        qi = min(x[1] for x in gai)
        zhi = max(x[2] for x in gai)
        self.shezhi_mianban.zhuangtai_shezhi(
            f"字幕时间已改 · {self._shijian_wenben(qi)} → "
            f"{self._shijian_wenben(zhi)}"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    # ---- 多选：删除 / 合并 ----
    def _shanchu_xuan_zhong_zimu(self):
        """按了 Delete：删掉时间轴上选中的字幕块

        焦点在输入框里时不抢 —— 那儿 Delete 就该删字符。
        """
        zhu = QtWidgets.QApplication.focusWidget()
        if isinstance(
            zhu,
            (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
             QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox),
        ):
            return
        xu = self.shijianzhou.xuan_zhong_liebiao()
        if xu:
            self._shoudao_zimu_shanchu(xu)

    def _hebing_xuan_zhong_zimu(self):
        """按了 Ctrl+M：把时间轴上选中的字幕块并成一条"""
        xu = self.shijianzhou.xuan_zhong_liebiao()
        if len(xu) >= 2:
            self._shoudao_zimu_hebing(xu)

    def _qiefen_zimu_kuai(self):
        """按了 S：把游标（播放头）停着的那条字幕块，从游标位置切成两块

        切出来两条文字一模一样，只是时间一分为二 —— 硬字幕提取把两句
        连成一条、时间没分开时，用它手动断开。焦点在输入框里时不抢。
        """
        zhu = QtWidgets.QApplication.focusWidget()
        if isinstance(
            zhu,
            (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
             QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox),
        ):
            return
        zimu = self.zimu_mianban.zimu_liebiao()
        if not zimu:
            return
        dao = int(self.shijianzhou.bofangtou_ms())
        xu = None
        for i, (qi, zhi, _wenben) in enumerate(zimu):
            if int(qi) <= dao <= int(zhi):
                xu = i
                break
        if xu is None:
            self.shezhi_mianban.zhuangtai_shezhi(
                "切分：游标不在任何字幕块上"
            )
            return
        qi, zhi, wenben = zimu[xu]
        qi, zhi = int(qi), int(zhi)
        if not (qi < dao < zhi):
            # 游标正好压在块头或块尾上：再切就会切出一条 0 长度的块
            self.shezhi_mianban.zhuangtai_shezhi(
                "切分：游标贴着字幕块的头 / 尾，把游标挪开一点再切"
            )
            return
        sheng = list(zimu)
        sheng[xu:xu + 1] = [(qi, dao, wenben), (dao, zhi, wenben)]
        # 切完就地留在切出来的前半段上（后半段不选）
        xie = self._ying_yong_zimu(sheng, xu, "切分", "qiefen")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已在 {self._shijian_wenben(dao)} 把这条字幕切成两块"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    # ---- 字幕列表右键：整行编辑（照 AEG 的网格右键菜单） ----
    def _hang_mu(self, mu=None):
        """这次动作处理哪几行：右键点过的用它，没有就用列表里当前选中的

        返回 (整份字幕, 排好序的行号表)。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        if mu is None:
            mu = getattr(self, "_youjian_mu", None)
        if mu is None:
            mu = self.bianji_mianban.xuan_zhong_hang()
        return zimu, sorted(
            {int(x) for x in (mu or []) if 0 <= int(x) < len(zimu)}
        )

    def _hang_paixu(self, zimu):
        """按时间重排一遍：整份列表是按时间排的，插完 / 粘完得摆回去"""
        return sorted(zimu, key=lambda z: (int(z[0]), int(z[1])))

    def _hang_shoudao(self, xin, xuan, shuo):
        """整行编辑类动作统一收尾：刷三处 + 重写 SRT / ASS + 状态栏说一句"""
        xie = self._ying_yong_zimu(
            xin, int(xuan), str(shuo).lstrip("已"), "hang"
        )
        self.shezhi_mianban.zhuangtai_shezhi(
            shuo
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _hang_mei_xuan(self, dongzuo):
        """没选中行时的统一提示"""
        self.shezhi_mianban.zhuangtai_shezhi(
            f"{dongzuo}：先在字幕列表里点一行"
        )

    def _hang_charu(self, mu=None, zai_qian=True):
        """插入(之前 / 之后)：在当前行前 / 后插一条空行（照 AEG）

        新行紧贴着当前行，长 XINJIAN_ZIMU_MS。旁边那条已经占满整段（硬字幕
        常常一段挨一段、中间一点空都没有）时，就让新行跟它叠一帧 —— 不能因为
        旁边挤就插不进去（时间轴能摞好几行，叠着也看得见）。

        但当前行本身已经顶到片头 / 片尾时，前面 / 后面是真没地方了：这一下
        什么都不做，也不弹提示。
        """
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("插入行")
            return
        i = mu[0]
        qi, zhi = int(zimu[i][0]), int(zimu[i][1])
        pian_chang = int(self.shichang_ms)
        # 让不开也得给一帧的长度，不然这条是零长、谁都看不见
        yi_zhen = max(1, int(round(1000.0 / self.fps))) if self.fps > 0 else 40
        if zai_qian:
            if qi <= 0:
                return              # 当前行就在片头第一帧，前面没地方：什么都不做
            z = qi
            q = max(0, z - XINJIAN_ZIMU_MS)
            if i - 1 >= 0:
                q = max(q, int(zimu[i - 1][1]))      # 前一条占到这里了
            wei = i
        else:
            if pian_chang > 0 and zhi >= pian_chang:
                return              # 当前行就在片尾最后一帧，后面没地方：什么都不做
            q = zhi
            z = q + XINJIAN_ZIMU_MS
            if i + 1 < len(zimu):
                z = min(z, int(zimu[i + 1][0]))      # 后一条从这儿开始
            wei = i + 1
        if z <= q:
            # 被旁边那条顶死了（不是片头 / 片尾）就叠一帧上去，照样插
            if zai_qian:
                q = max(0, qi - yi_zhen)
                z = qi
            else:
                q = zhi
                z = zhi + yi_zhen
                if pian_chang > 0:
                    z = min(z, pian_chang)
        tiao = (q, z, "")
        xin = list(zimu)
        xin.insert(wei, tiao)
        xin = self._hang_paixu(xin)
        xuan = xin.index(tiao) if tiao in xin else -1
        self._hang_shoudao(
            xin, xuan,
            f"已{'前' if zai_qian else '后'}插一条空字幕（第 {xuan + 1} 条）",
        )

    def _hang_charu_shipin(self, mu=None, zai_qian=True):
        """以视频时间插入(之前 / 之后)：新行从播放头那一帧开始（照 AEG）

        播放头已经贴在片头第一帧（前插）/ 片尾最后一帧（后插）时，前面 /
        后面真没地方：这一下什么都不做，也不弹提示。
        """
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("以视频时间插入")
            return
        chang = int(self.shichang_ms)
        if chang <= 0:
            self.shezhi_mianban.zhuangtai_shezhi("以视频时间插入：还没打开视频")
            return
        yi_zhen = max(1, int(round(1000.0 / self.fps))) if self.fps > 0 else 40
        i = mu[0]
        q = max(0, int(self.shijianzhou.bofangtou_ms()))
        if zai_qian:
            if q <= 0:
                return          # 播放头就在片头第一帧，前面没地方：什么都不做
        else:
            if q >= chang - yi_zhen:
                return          # 播放头就在片尾最后一帧，后面没地方：什么都不做
        z = min(q + XINJIAN_ZIMU_MS, chang)
        if z <= q:
            return
        wei = i if zai_qian else i + 1
        tiao = (q, z, "")
        xin = list(zimu)
        xin.insert(wei, tiao)
        # 不按时间重排：这两项的新行都是从播放头起、时间一模一样，重排一遍
        # 前/后就分不出来了 —— 点"之前"和点"之后"会是一个结果
        self._hang_shoudao(
            xin, wei,
            f"已从 {self._shijian_wenben(q)} 插一条空字幕"
            f"（第 {wei + 1} 条）",
        )

    def _hang_fuzhi(self, mu=None):
        """重复行：复制一份落在原块的**下面一行**（时间原样）

        我们时间轴能摞好几行字幕块；副本要是跟原件时间一样、还摆同一行，两条
        就完全压在一起，看着像没生效。所以副本沉到原件下面那一行。
        """
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("重复行")
            return
        self._xiayi_hang_tian(zimu, mu, [tuple(zimu[i]) for i in mu], "已重复")

    def _xiayi_hang_tian(self, zimu, mu, kuai, shuo):
        """把 kuai 这些块插在选中行后面，并让它们落到原块的下一行

        索引那边插完就完事；行号得单独再喊时间轴摆一次 —— 整份 shezhi_zimu
        是按"内容"认行号的，副本跟原件内容一样，认不到旧行号会掉回第一行。
        """
        mu = sorted({int(x) for x in (mu or []) if 0 <= int(x) < len(zimu)})
        kuai = [tuple(k) for k in (kuai or [])]
        if not mu or not kuai:
            return
        hang_jiu = [self.shijianzhou.zimu_hang(i) for i in mu]
        # 每条副本落在它那条原件的下一行；条数对不上（粘贴来的）都跟着最后那条走
        hang_xin = hang_jiu if len(kuai) == len(mu) else [hang_jiu[-1]] * len(kuai)
        wei = mu[-1] + 1
        xin = list(zimu)
        # 不重排：副本跟原件时间一样，插在它后面本来就在对的时间位置上；
        # 重排一遍的话副本索引会飘，行号对不上了（其它动作那边仍是重排的）。
        xin[wei:wei] = kuai
        # 选中还留在原件上（跟 AEG 一样：重复完不动选择）
        self._hang_shoudao(xin, mu[0], shuo + f" {len(kuai)} 条，落在原块的下一行")
        # 行号必须放在 shezhi_zimu 后面摆，不然会被挤回第一行
        self.shijianzhou.shezhi_hang(
            [
                (
                    wei + n,
                    min(
                        self.shijianzhou.ZIMU_HANG_ZUIDA - 1, hang_xin[n] + 1
                    ),
                )
                for n in range(len(kuai))
            ]
        )

    def _hang_qiege(self, mu=None, xiang_hou=False):
        """以当前帧前 / 后分割行：在播放头那一帧把一条切成两条（照 AEG）

        切出来两条文字一样、时间一分为二。播放头不在这条里就跳过这一条。
        """
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("分割行")
            return
        if self.fps <= 0:
            self.shezhi_mianban.zhuangtai_shezhi("分割行：还没读到视频帧率")
            return
        dao = max(0, int(self.shijianzhou.bofangtou_ms()))
        zhen = int(round(dao * self.fps / 1000.0))
        # 前分割 = 切在当前帧起点，后分割 = 切在下一帧起点（都落在帧边界上）
        dian = int(round((zhen + (1 if xiang_hou else 0)) * 1000.0 / self.fps))
        xin = list(zimu)
        tiao = None
        qie = 0
        for i in reversed(mu):
            qi, zhi, wenben = int(zimu[i][0]), int(zimu[i][1]), zimu[i][2]
            if not (qi < dian < zhi):
                continue
            xin[i:i + 1] = [(qi, dian, wenben), (dian, zhi, wenben)]
            tiao = xin[i]
            qie += 1
        if not qie:
            self.shezhi_mianban.zhuangtai_shezhi(
                "分割行：播放头不在选中的字幕行里"
            )
            return
        xin = self._hang_paixu(xin)
        xuan = xin.index(tiao) if tiao in xin else -1
        self._hang_shoudao(
            xin, xuan,
            f"已在 {self._shijian_wenben(dian)} 把 {qie} 条切成两段",
        )

    def _hang_huhuan(self, mu=None):
        """互换行：把选中的两条的内容对调（时间不动）

        AEG 那个是交换两行在网格里的先后位置；我们这份列表是按时间排的，
        位置由时间说了算，能对调的只有内容（点错行时用它换回来）。样式 /
        说话人那套字段跟着一起换。
        """
        zimu, mu = self._hang_mu(mu)
        if len(mu) != 2:
            self.shezhi_mianban.zhuangtai_shezhi("互换行：先正好选中两条")
            return
        a, b = mu
        xin = list(zimu)
        xin[a] = (zimu[a][0], zimu[a][1], zimu[b][2])
        xin[b] = (zimu[b][0], zimu[b][1], zimu[a][2])
        # 字段是按"内容"存着的，内容一换就对不上了，得手动搬过去
        fu_a = self.zimu_mianban.hang_fu(a)
        fu_b = self.zimu_mianban.hang_fu(b)
        self.zimu_mianban.shezhi_zimu(xin)
        if fu_b:
            self.zimu_mianban.shezhi_hang_fu(a, fu_b)
        if fu_a:
            self.zimu_mianban.shezhi_hang_fu(b, fu_a)
        self._hang_shoudao(xin, a, "已把这两条的内容对调")

    def _hang_shijian_lianxu(self, mu=None, gai_qi=True):
        """使时间连续：本行的起点接到上一行的终点（改开始时间），
        本行的终点接到下一行的起点（改结束时间）"""
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("使时间连续")
            return
        xin = list(zimu)
        gai = 0
        for i in mu:
            qi, zhi, wenben = (int(zimu[i][0]), int(zimu[i][1]), zimu[i][2])
            if gai_qi:
                if i - 1 < 0:
                    continue
                q = int(xin[i - 1][1])          # 前面那条刚改过就用新值
                if q >= zhi:
                    continue                    # 接上去会把行倒过来，跳过
                xin[i] = (q, zhi, wenben)
            else:
                if i + 1 >= len(xin):
                    continue
                z = int(xin[i + 1][0])
                if z <= qi:
                    continue
                xin[i] = (qi, z, wenben)
            gai += 1
        if not gai:
            self.shezhi_mianban.zhuangtai_shezhi("使时间连续：这几行没有可接的邻居")
            return
        self._hang_shoudao(
            xin, mu[0],
            f"已让 {gai} 条的{'开始' if gai_qi else '结束'}时间跟邻居接上",
        )

    def _hang_chongzu(self, mu=None):
        """重组行：把被拆开的相邻行按文本对回去（照 AEG 的 recombine）

        硬字幕提取常把一句话拆成相邻两条、文字互相咬着一截。这条命令把选中
        的行按时间捋一遍，文本完全相同、或一条是另一条的开头 / 结尾，就并
        成一条（时间取并集）；对不上的原样留着不动。
        """
        zimu, mu = self._hang_mu(mu)
        if len(mu) < 2:
            self.shezhi_mianban.zhuangtai_shezhi("重组行：先选中两条以上")
            return
        xuan = sorted(
            (zimu[i] for i in mu), key=lambda z: (int(z[0]), int(z[1]))
        )
        he = []
        dang = xuan[0]
        for hou in xuan[1:]:
            a = str(dang[2] or "").strip()
            b = str(hou[2] or "").strip()
            wen = None
            if a and b:
                if a == b or b.startswith(a) or b.endswith(a):
                    wen = b
                elif a.startswith(b) or a.endswith(b):
                    wen = a
            elif a:
                wen = a
            elif b:
                wen = b
            if wen is None:                     # 咬不上：各留各的
                he.append(dang)
                dang = hou
                continue
            dang = (
                min(int(dang[0]), int(hou[0])),
                max(int(dang[1]), int(hou[1])),
                wen,
            )
        he.append(dang)
        if len(he) == len(xuan):
            self.shezhi_mianban.zhuangtai_shezhi("重组行：这几条咬不上，没动")
            return
        pao = set(mu)
        xin = self._hang_paixu(
            [zimu[i] for i in range(len(zimu)) if i not in pao] + he
        )
        tiao = he[0]
        xuan_hang = xin.index(tiao) if tiao in xin else -1
        self._hang_shoudao(
            xin, xuan_hang,
            f"已把 {len(xuan)} 条重组回 {len(he)} 条",
        )

    def _ass_hang_wenben(self, zimu, fu_men, xu):
        """第 xu 条拼成一行 ASS 文本（AEG 网格里复制出来的就是这个格式）

        Dialogue: 层,起,止,样式,说话人,左边距,右边距,垂直边距,特效,文字

        Text 优先沿用原 ASS 里那一串（\\pos 这些行内标签原样留着）；文字真改
        过才用现在这份（换行换成 \\N）。认原来那一行的规矩跟写 ASS 时一样。
        """
        from anylabeling.views.labeling.utils.video import format_ass_time

        qi, zhi, wenben = zimu[xu]
        fu = dict(fu_men[xu]) if xu < len(fu_men) else {}
        qing = self._ass_qing_hang()
        chun = _ass_chun_wen(wenben)
        yuan = next((y for c, _q, _z, _y, _s, y, _f in qing if c == chun), "")
        if not yuan:
            yuan = next(
                (
                    y for _c, q, z, _y, _s, y, _f in qing
                    if q == int(qi) and z == int(zhi)
                ),
                "",
            )
        if yuan and _ass_yuan_wen(yuan) == _ass_yuan_wen(wenben):
            wen = str(yuan)
        else:
            wen = str(wenben or "").replace("\n", "\\N")
        return (
            ("Comment" if fu.get("zhushi") else "Dialogue")
            + ": "
            + ",".join(
                (
                    str(max(0, int(fu.get("ceng") or 0))),
                    format_ass_time(int(qi)),
                    format_ass_time(int(zhi)),
                    str(fu.get("yang") or "Default"),
                    str(fu.get("shuo") or ""),
                    f"{max(0, int(fu.get('zuo') or 0)):04d}",
                    f"{max(0, int(fu.get('you') or 0)):04d}",
                    f"{max(0, int(fu.get('shu') or 0)):04d}",
                    str(fu.get("texiao") or ""),
                    wen,
                )
            )
        )

    def _hang_jianqie(self, mu=None, shan=True):
        """剪切行 / 复制行：写成 ASS 的 Dialogue 行，放**系统**剪贴板

        AEG 网格里 Ctrl+C / Ctrl+X 出来的就是这一串（一段一行），所以粘到
        AEG、记事本、剪贴板记录工具里都是现成能用的。
        """
        zimu, mu = self._hang_mu(mu)
        if not mu:
            self._hang_mei_xuan("剪切行" if shan else "复制行")
            return
        fu_men = self._zimu_fujia(zimu)
        QtWidgets.QApplication.clipboard().setText(
            "\n".join(self._ass_hang_wenben(zimu, fu_men, i) for i in mu)
        )
        if shan:
            self._shoudao_zimu_shanchu(mu)
            return
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已复制 {len(mu)} 条到剪贴板（ASS 的 Dialogue 行）"
        )

    def _du_zhantie_hang(self, wen):
        """剪贴板文本 -> [(起ms, 止ms, 文字, 字段表), ...]

        只认 ASS 的 Dialogue / Comment 行（一段一行），别的内容跳过；一段都
        没认出来就返回空表。认出来的行把样式 / 说话人 / 层 / 边距 / 特效 /
        注释一起带上，粘进来不掉东西。
        """
        ming = list(_ASS_ZIDUAN)
        chu = []
        for hang in str(wen or "").splitlines():
            tiao = hang.strip()
            di = tiao.lower()
            if not di.startswith(("dialogue:", "comment:")):
                continue
            # Text 是最后一个字段、里面可以有逗号：只切前面那几个
            bu = tiao.split(":", 1)[1].strip().split(",", len(ming) - 1)
            bu += [""] * (len(ming) - len(bu))
            h = dict(zip(ming, bu))
            qi = _ass_hao_ms(h.get("Start"))
            zhi = _ass_hao_ms(h.get("End"))
            if qi is None or zhi is None or int(zhi) <= int(qi):
                continue
            h["_lei"] = "Comment" if di.startswith("comment:") else "Dialogue"
            fu = _ass_hang_fu(h)
            fu["yang"] = str(h.get("Style") or "Default")
            fu["shuo"] = str(h.get("Name") or "")
            chu.append(
                (
                    int(qi),
                    int(zhi),
                    str(h.get("Text") or "")
                    .replace("\\N", "\n")
                    .replace("\\n", "\n"),
                    fu,
                )
            )
        return chu

    def _hang_zhantie(self, mu=None):
        """粘贴行：认系统剪贴板里的 ASS 行，插到当前行后面，再按时间摆回列表

        AEG（或剪贴板记录工具）里复制出来的 Dialogue 行都能粘；粘进来的行
        样式 / 说话人 / 层 / 边距 / 特效 / 注释一起带过来。
        """
        tiao = self._du_zhantie_hang(
            QtWidgets.QApplication.clipboard().text()
        )
        if not tiao:
            self.shezhi_mianban.zhuangtai_shezhi(
                "粘贴行：剪贴板里没有 ASS 行（「Dialogue: …」那种）"
            )
            return
        zimu, mu = self._hang_mu(mu)
        wei = (mu[-1] + 1) if mu else len(zimu)
        xin = list(zimu)
        xin[wei:wei] = [t[:3] for t in tiao]
        xin = self._hang_paixu(xin)
        # 带过来的字段记在 _fu_wai 上：算「每条字段」时按行内容认领
        for qi, zhi, wb, fu in tiao:
            self._fu_wai[(qi, zhi, wb)] = dict(fu)
        xuan = xin.index(tiao[0][:3]) if tiao[0][:3] in xin else -1
        self._hang_shoudao(xin, xuan, f"已粘贴 {len(tiao)} 条字幕")

    def _shoudao_kuai_fuzhi(self):
        """时间轴上按了 Ctrl+C：把选中的块写成 ASS 行放**系统**剪贴板

        跟字幕列表里的 Ctrl+C 完全一条路 —— 剪贴板记录工具里能看到
        「Dialogue: …」，粘到 AEG / 记事本里也是现成能用的。
        """
        self._hang_jianqie(
            self.shijianzhou.xuan_zhong_liebiao() or None, shan=False
        )

    def _shoudao_kuai_zhantie(self):
        """时间轴上按了 Ctrl+V：在选中块的**下面一行**造一条重复块

        剪贴板里是 ASS 行就用它的时间（从别处复制来的也认）；剪贴板里没有
        ASS 行，就当"把选中的这块复制一份到下面"，等于重复行。
        """
        zimu, mu = self._hang_mu(self.shijianzhou.xuan_zhong_liebiao() or None)
        if not mu:
            self._hang_mei_xuan("粘贴到下一行")
            return
        tiao = self._du_zhantie_hang(QtWidgets.QApplication.clipboard().text())
        # 带过来的字段记在 _fu_wai 上：算「每条字段」时按行内容认领
        for qi, zhi, wb, fu in tiao:
            self._fu_wai[(qi, zhi, wb)] = dict(fu)
        self._xiayi_hang_tian(
            zimu,
            mu,
            [t[:3] for t in tiao] if tiao else [tuple(zimu[i]) for i in mu],
            "已粘贴" if tiao else "已复制",
        )

    # --------------------------------------------------------------
    # 查找 / 替换 / 选择（Ctrl+R / Ctrl+F / Ctrl+H 那三个窗口）
    # --------------------------------------------------------------
    def _kai_sousuo(self, na):
        """打开（或叫回）选择 / 查找 / 替换窗口

        na："xuan" 选择、"zhao" 查找、"huan" 替换。三个都是非模态窗口，
        开着照样能操作工作台，而且可以同时开着：各留一个实例，互不顶掉。
        收进最小化了再按一次快捷键就能叫回来。
        """
        ming = {"xuan": "_ss_xuan", "zhao": "_ss_zhao", "huan": "_ss_huan"}[na]
        chuang = getattr(self, ming, None)
        if chuang is None:
            if na == "xuan":
                chuang = XuanZeDialog(self)
            else:
                chuang = SuoSuoDialog(self, na == "huan")
            setattr(self, ming, chuang)
            # 头一回开：摆在窗口偏上居中的位置，三个依次往右下错开一点，
            # 不然同时开着会全叠在一块儿，还得自己一个个拖
            chuang.adjustSize()
            cuo = {"xuan": 0, "zhao": 26, "huan": 52}[na]
            wo = self.frameGeometry()
            chuang.move(
                wo.center().x() - chuang.width() // 2 + cuo,
                wo.top() + 70 + cuo,
            )
        if chuang.isMinimized():
            chuang.showNormal()          # 最小化收进去了：叫回来
        chuang.show()
        chuang.raise_()
        chuang.activateWindow()
        chuang.qing_jiaodian()

    def ss_hang_men(self):
        """给那三个窗口的整份字幕：一条一个字典（文字 / 样式 / 说话人 / 特效 / 注释）"""
        zimu = self.zimu_mianban.zimu_liebiao()
        fu_men = self._zimu_fujia(zimu)
        men = []
        for i, (_qi, _zhi, wen) in enumerate(zimu):
            fu = dict(fu_men[i]) if i < len(fu_men) else {}
            men.append({
                "wenben": str(wen or ""),
                "yang": str(fu.get("yang") or "Default"),
                "shuo": str(fu.get("shuo") or ""),
                "texiao": str(fu.get("texiao") or ""),
                "zhushi": bool(fu.get("zhushi")),
            })
        return men

    def ss_xuan_hang(self):
        """列表里现在选中了哪几行"""
        return [int(x) for x in self.bianji_mianban.xuan_zhong_hang()]

    def ss_dangqian(self):
        """现在编辑的是第几条（没选中给 -1）"""
        return int(self.bianji_mianban.dangqian_xu())

    def ss_she_xuan(self, hang):
        """把选中的设成这几行（列表 + 时间轴一起高亮，不动播放头）"""
        zimu = self.zimu_mianban.zimu_liebiao()
        hang = sorted({int(x) for x in (hang or []) if 0 <= int(x) < len(zimu)})
        if len(hang) == 1:
            self._shoudao_xuan_zhong_zimu(hang[0])
            return
        if not hang:
            self._shoudao_xuan_zhong_zimu(-1)
            return
        self.bianji_mianban.shezhi_xuan_zhong_duo(hang)
        self.shijianzhou.shezhi_xuan_zhong_duo(hang)

    def ss_dingwei(self, xu, qi=None, zhi=None):
        """定位到第 xu 条（列表 / 时间轴 / 编辑区一起跟上）

        文本栏给 qi / zhi 就在编辑框里把匹配的那一段圈出来。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        xu = int(xu)
        if not (0 <= xu < len(zimu)):
            return
        self._shoudao_xuan_zhong_zimu(xu)
        if qi is None or zhi is None:
            return
        kuang = getattr(self.bianji_mianban, "kuang", None)
        if kuang is None:
            return
        wen = str(zimu[xu][2] or "")
        qi = max(0, min(int(qi), len(wen)))
        zhi = max(qi, min(int(zhi), len(wen)))
        guang = QtGui.QTextCursor(kuang.document())
        guang.setPosition(qi)
        guang.setPosition(zhi, QtGui.QTextCursor.KeepAnchor)
        kuang.setTextCursor(guang)

    def ss_gai(self, jian, xiugai):
        """批量改某一栏：jian 见那三个窗口里的 LAN_JIAN，xiugai = [(行号, 新内容)]

        一次改完再刷列表 / 重写文件 —— 一条一条走全套流程会卡。
        """
        zimu = self.zimu_mianban.zimu_liebiao()
        if not xiugai or not zimu:
            return 0
        gai = 0
        if jian == "wenben":
            for xu, xin in xiugai:
                xu = int(xu)
                if not (0 <= xu < len(zimu)):
                    continue
                if str(zimu[xu][2]) == str(xin):
                    continue
                self.zimu_mianban.gai_zimu_wenben(xu, str(xin))
                self.shijianzhou.gai_zimu_wenben(xu, str(xin))
                gai += 1
        else:
            for xu, xin in xiugai:
                xu = int(xu)
                if not (0 <= xu < len(zimu)):
                    continue
                fu = dict(self.bianji_mianban.hang_fu(xu))
                fu.update(self.zimu_mianban.hang_fu(xu))
                if str(fu.get(jian) or "") == str(xin or ""):
                    continue
                fu[jian] = xin
                self.zimu_mianban.shezhi_hang_fu(xu, fu)
                gai += 1
        if not gai:
            return 0
        self._shezhi_bianji_zimu(
            self.zimu_mianban.zimu_liebiao(),
            self.bianji_mianban.dangqian_xu(),
        )
        self._zimu_hang_ji = None
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        self._shuaxin_huamian_zimu()
        xie = self._chong_xie_zimu_wenjian("批量改字幕")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"改了 {gai} 条字幕" + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )
        return gai

    def ss_zhuangtai(self, wen):
        """底部状态栏说一句"""
        self.shezhi_mianban.zhuangtai_shezhi(str(wen))

    def ss_tishi(self, biaoti, wen):
        """弹个提示框（只有"一条都没匹配上"这种才弹，照 AEG）"""
        QtWidgets.QMessageBox.information(self, str(biaoti), str(wen))

    def _shoudao_zimu_shanchu(self, xu_liebiao=None):
        """删掉这几条：右侧列表、时间轴、SRT / ASS 一起改"""
        zimu, shan = self._hang_mu(xu_liebiao)
        if not shan:
            self._hang_mei_xuan("删除行")
            return
        pao = set(shan)
        sheng = [z for i, z in enumerate(zimu) if i not in pao]
        xie = self._ying_yong_zimu(sheng, -1, f"删除 {len(shan)} 行", "shan")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已删除 {len(shan)} 段字幕"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _shoudao_zimu_hebing(self, xu_liebiao=None, liu_shou_hang=False):
        """并成一条：起 = 最早那条的起，止 = 最晚那条的止

        文字默认按顺序换行接上；liu_shou_hang=True（AEG 的"合并（保留首
        行）"）就只留第一条的文字。
        """
        zimu, bing = self._hang_mu(xu_liebiao)
        if len(bing) < 2:
            self.shezhi_mianban.zhuangtai_shezhi("合并：先选中两条以上")
            return
        pao = set(bing)
        xuan = [zimu[i] for i in bing]
        qi = min(int(x[0]) for x in xuan)
        zhi = max(int(x[1]) for x in xuan)
        if liu_shou_hang:
            wenben = str(xuan[0][2] or "").strip()
        else:
            wenben = "\n".join(t for t in (str(x[2]).strip() for x in xuan) if t)
        # 没选中的照旧 + 合并出来的那一条，按时间重新排一遍：
        # 隔着几段合并时，并出来的那条可能不在原来的位置
        dai = [(i, zimu[i]) for i in range(len(zimu)) if i not in pao]
        dai.append((-1, (qi, zhi, wenben)))
        dai.sort(key=lambda p: (int(p[1][0]), int(p[1][1])))
        xhao = next(i for i, (biao, _z) in enumerate(dai) if biao == -1)
        sheng = [z for _biao, z in dai]
        xie = self._ying_yong_zimu(sheng, xhao, "合并", "hebing")
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已合并 {len(bing)} 段字幕（{self._shijian_wenben(qi)} → "
            f"{self._shijian_wenben(zhi)}）"
            + ("，SRT / ASS 已重写" if xie else "")
            + "（Ctrl+S 保存字幕）"
        )

    def _ying_yong_zimu(self, zimu, xuan=-1, shuo=None, lei=None):
        """改完字幕统一收尾：右侧列表 / 迷你轴 / 底部时间轴一起刷，
        再重写提取目录里的 SRT / ASS。返回写出的文件列表（空 = 没写）。
        """
        self.zimu_mianban.shezhi_zimu(zimu)
        self.shijianzhou.shezhi_zimu(zimu)
        self.shijianzhou.shezhi_xuan_zhong(int(xuan))
        self.zimu_mianban.shezhi_xuan_zhong(int(xuan))
        self._shezhi_bianji_zimu(zimu, int(xuan))
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        return self._chong_xie_zimu_wenjian(shuo, lei)

    def _shuaxin_zimu_tiao(self, hua=True):
        """画面下方那条 + 画面里叠的那层字幕（都跟着播放头走）

        hua=False（点击 / 拖播放头那一路）时不去碰画面里叠的那层：那会儿 mpv
        的 time-pos 已经跳到目标了，可画面还得等解码 —— 跟着它换就是"字幕先
        切、画面后到"。那层由 paintGL 在"mpv 真出新视频帧"那一刻换。
        """
        if hua:
            self._shuaxin_huamian_zimu()
        # 字幕列表里"正播到"的那一行跟着上色 / 滚动（跟手动选中的行互不干扰）
        # 拖播放头的时候不刷它：几千条要扫一遍、还要滚动列表，一卡播放头就
        # 跟不上鼠标；松手那一下会补刷（见 _tuo_bofangtou_wancheng）。
        if not self._tuo_bt_zhong:
            # 正播着才让列表跟着播放行滚；暂停着只涂色不滚 —— 打注释 / 改字段
            # 会整表重建，紧接着这次刷新拿的是还没动的播放头位置，一滚列表就
            # "啪"跳回最顶上，等定位回来再滚下去，看着就是闪
            self.bianji_mianban.shezhi_bofang_ms(
                self._bofang_ms, self.zhengzai_bofang
            )
        zai = self.zimu_mianban.zimu_zai_ms(self._bofang_ms)
        wenben = _ass_chun_wen(zai[2]) if zai else ""
        if wenben == self._zimu_tiao_wen:
            return
        self._zimu_tiao_wen = wenben
        self.zimu_tiao.setToolTip(wenben)
        kuan = max(60, self.zimu_tiao.width() - 28)
        zi = QtGui.QFontMetrics(self.zimu_tiao.font()).elidedText(
            wenben, Qt.ElideRight, kuan
        )
        self.zimu_tiao.setText(zi)

    # ---- 自动保存 / 自动备份（照 AEG 那套） ----
    def _zimu_mulu(self):
        """字幕文件放在哪一层：存的哪个文件就在它旁边，还没存过就按输出目录 / 视频目录"""
        if self.zimu_wenjian:
            return osp.dirname(self.zimu_wenjian)
        shuchu = self.shezhi_mianban.shuchu_shuru.text().strip()
        if shuchu:
            return osp.dirname(shuchu)
        if self.lujing:
            return osp.dirname(self.lujing)
        return ""

    def _zidong_beifen(self, lu):
        """要动这份字幕之前，先把磁盘上的原件抄一份进「自动备份」

        照 AEG：只在"打开 / 第一次写"这个动作前做一次，同名覆盖 —— 一个文件
        永远只留一份原件。它救的是"越改越烂"：不管后面怎么改，打开时长什么样
        随时能拿回来。文件还不存在（头一回生成）就不用备。
        """
        if not lu or lu in self._zidong_beifen_guo or not osp.isfile(lu):
            return ""
        self._zidong_beifen_guo.add(lu)
        gen, kuo = osp.splitext(lu)
        shuo = osp.join(
            osp.dirname(lu),
            ZIDONG_BEIFEN_JIA,
            osp.basename(gen) + ".ORIGINAL" + kuo,
        )
        try:
            os.makedirs(osp.dirname(shuo), exist_ok=True)
            shutil.copy2(lu, shuo)
        except OSError as cuowu:
            logger.error(f"视频工作台：自动备份失败 {cuowu}")
            return ""
        logger.info(f"视频工作台：已自动备份 {shuo}")
        return shuo

    def _zidong_baocun(self):
        """到点了：字幕跟上次自动保存时不一样，就再存一份带时间戳的

        照 AEG：文件名是「原名(连扩展名).年月日-时分秒.AUTOSAVE.ass」，一次
        一份、从不覆盖，攒成一串历史版本。没改过就一份都不写。
        """
        if not self._zidong_gai_dong:
            return ""
        zimu = self.zimu_mianban.zimu_liebiao()
        mulu = self._zimu_mulu()
        if not zimu or not mulu or not osp.isdir(mulu):
            return ""
        xian = [(int(q), int(z), str(w)) for q, z, w in zimu]
        if xian == self._zidong_shange:
            self._zidong_gai_dong = False
            return ""
        ming = osp.basename(self.zimu_wenjian) or (
            osp.basename(self.lujing) or "未命名"
        )
        shuo = osp.join(
            mulu,
            ZIDONG_BAOCUN_JIA,
            ming + "." + time.strftime("%Y-%m-%d-%H-%M-%S") + ".AUTOSAVE.ass",
        )
        try:
            os.makedirs(osp.dirname(shuo), exist_ok=True)
            _xie_zimu_dao_wenjian(
                shuo, zimu, self._ass_yuan, self._zimu_fujia(zimu)
            )
        except OSError as cuowu:
            logger.error(f"视频工作台：自动保存失败 {cuowu}")
            return ""
        self._zidong_gai_dong = False
        self._zidong_shange = xian
        self.shezhi_mianban.zhuangtai_shezhi(f"已自动保存 · {osp.basename(shuo)}")
        logger.info(f"视频工作台：已自动保存 {shuo}")
        return shuo

    def _chong_xie_zimu_wenjian(self, shuo=None, lei=None):
        """改完字幕把 SRT / ASS 重写一遍（规矩跟跑完时一样，写在输出文件夹同级）

        shuo / lei：这一下算哪门子动作（撤销栈里显示的名字 / 合并用的类别）。
        改完刷完就顺手记一笔快照，Ctrl+Z 能退回来。
        """
        # 标记放最前：改没改过跟"这次有没有写出去"没关系。
        # 输出目录没设好时下面会直接返回，标记要是放在它后面就永远置不上，
        # 自动保存一次都不会触发。
        self._zidong_gai_dong = True    # 字幕变过了：下一轮到点就自动存一份
        self._ji_chexiao(shuo or "改字幕", lei)
        zimu = self.zimu_mianban.zimu_liebiao()
        shuchu = self.shezhi_mianban.shuchu_shuru.text().strip()
        if not zimu or not shuchu or not osp.isdir(shuchu):
            return []
        from anylabeling.views.labeling.utils.video import format_srt_time

        ming = osp.basename(shuchu)
        xie = []
        srt_lu = osp.join(osp.dirname(shuchu), f"{ming}.srt")
        self._zidong_beifen(srt_lu)     # 覆盖前先把原来那份抄进「自动备份」
        try:
            with open(srt_lu, "w", encoding="utf-8") as f:
                for xu, (qi, zhi, wenben) in enumerate(zimu, 1):
                    f.write(f"{xu}\n")
                    f.write(
                        f"{format_srt_time(qi)} --> {format_srt_time(zhi)}\n"
                    )
                    f.write(f"{_ass_chun_wen(wenben)}\n\n")
            xie.append(srt_lu)
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：重写 SRT 失败 {cuowu}")

        ass_lu = osp.join(osp.dirname(shuchu), f"{ming}.ass")
        self._zidong_beifen(ass_lu)     # 同上
        try:
            _xie_zimu_dao_wenjian(
                ass_lu, zimu, self._ass_yuan, self._zimu_fujia(zimu)
            )
            xie.append(ass_lu)
        except Exception as cuowu:  # noqa
            logger.error(f"视频工作台：重写 ASS 失败 {cuowu}")
        return xie

    # ---- 撤销 / 重做（照 AEG 的 SubsController：一步一整版快照） ----
    def _chexiao_kuazhao(self):
        """把当前这一版字幕整个拷一份（撤销 / 重做都拿它当存档）"""
        return (
            self.zimu_mianban.zimu_liebiao(),
            self.zimu_mianban.fujia_biao(),
            int(self.bianji_mianban.dangqian_xu()),
        )

    def _ji_chexiao(self, shuo, lei=None):
        """一个动作做完了：把现在这一版记一笔，供 Ctrl+Z 回退

        lei 是动作类别：同一类动作在 CHEXIAO_HEBING_MIAO 秒内连着做算一笔
        （长按方向键挪块、连着敲字不该刷出几十笔撤销）。跟上一笔一模一样
        也不重复记。
        """
        if self._chexiao_zhong:
            return
        kuai = self._chexiao_kuazhao()
        shike = time.monotonic()
        if self._chexiao_zhan:
            jshuo, jkuai, jlei, jshike = self._chexiao_zhan[-1]
            if jkuai == kuai:
                return
            if lei and lei == jlei and shike - jshike <= CHEXIAO_HEBING_MIAO:
                self._chexiao_zhan[-1] = (jshuo, kuai, lei, shike)
                self._chongzuo_zhan.clear()
                return
        self._chexiao_zhan.append((str(shuo), kuai, lei, shike))
        while len(self._chexiao_zhan) > CHEXIAO_ZUIDA:
            self._chexiao_zhan.pop(0)
        self._chongzuo_zhan.clear()

    def _chexiao_qingkong(self, shuo="载入字幕"):
        """换了一份字幕：撤销栈从头来（照 AEG：打开文件就把撤销栈清空）"""
        self._chexiao_zhan = []
        self._chongzuo_zhan = []
        self._chexiao_zhong = False
        if self.zimu_mianban.zimu_liebiao():
            self._ji_chexiao(shuo, "zairu")

    def _chexiao_huifu(self, kuai):
        """把存档里的那一版字幕整个摆回来（列表 / 时间轴 / 文件一起回去）"""
        zimu, fu_biao, xuan = kuai
        self._chexiao_zhong = True
        try:
            # 字段表得先摆好：_ying_yong_zimu 一路会把「样式 / 说话人」重算出来
            self.zimu_mianban.shezhi_fujia_biao(fu_biao)
            self._ying_yong_zimu(
                [(int(q), int(z), str(w)) for q, z, w in zimu], int(xuan)
            )
        finally:
            self._chexiao_zhong = False

    def _shuru_chexiao(self, dong):
        """这一下 Ctrl+Z / Ctrl+Y 该不该交给本窗口的输入框自己办

        只有焦点落在**本窗口内**的输入框里、且那框里确实有得撤（重做）时
        才让它办，返回 True 表示已经办完了。焦点在别处 —— 时间轴上、或者
        查找 / 替换 / 选择那三个窗口里 —— 一律走字幕的撤销栈。
        """
        zhu = QtWidgets.QApplication.focusWidget()
        if zhu is None or not self.isAncestorOf(zhu):
            return False
        if not isinstance(
            zhu,
            (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
             QtWidgets.QPlainTextEdit),
        ):
            return False
        you = (zhu.isRedoAvailable() if dong == "redo"
               else zhu.isUndoAvailable())
        if not you:
            return False
        zhu.redo() if dong == "redo" else zhu.undo()
        return True

    def _chexiao(self):
        """Ctrl+Z：退回到上一版

        焦点在本窗口的输入框里、那框里又有字可撤时，这一次撤那一行文字
        （输入框自己的撤销）；其余情况一律撤字幕 —— 跟 AEG 一个规矩，
        撤销栈是整个字幕的，不分窗口。
        """
        if self._shuru_chexiao("undo"):
            return
        if len(self._chexiao_zhan) <= 1:
            self.ss_zhuangtai("没有可撤销的了")
            return
        # 栈里每笔存的是「那个动作做完之后的整版字幕」，所以栈顶就是现在这一版。
        # 撤销 = 把栈顶（现在这版）挪进重做栈，再退回新的栈顶（上一版）。
        shuo, kuai, _lei, _shike = self._chexiao_zhan.pop()
        self._chongzuo_zhan.append((shuo, kuai, "", 0.0))
        self._chexiao_huifu(self._chexiao_zhan[-1][1])
        self.ss_zhuangtai(f"已撤销：{shuo}")

    def _chongzuo(self):
        """Ctrl+Y：把撤销掉的那一步再做回来（输入框里有得重做就是重做那行文字）"""
        if self._shuru_chexiao("redo"):
            return
        if not self._chongzuo_zhan:
            self.ss_zhuangtai("没有可重做的了")
            return
        # 重做栈里那笔就是要回去的那一版：放回撤销栈顶（栈顶=现在这一版），
        # 再把它摆回来。
        shuo, kuai, _lei, _shike = self._chongzuo_zhan.pop()
        self._chexiao_zhan.append((shuo, kuai, "", 0.0))
        self._chexiao_huifu(kuai)
        self.ss_zhuangtai(f"已重做：{shuo}")

    # ---- 拖放：把 .srt / .ass 拖进工作台就直接载入 ----
    def _tuo_de_zimu(self, event):
        """拖进来的东西里第一个字幕文件；没有就返回空串"""
        if not event.mimeData().hasUrls():
            return ""
        for i in event.mimeData().urls():
            lu = i.toLocalFile()
            if lu and lu.lower().endswith(ZIMU_HOUZHUI) and osp.isfile(lu):
                return lu
        return ""

    def dragEnterEvent(self, event):
        if self._tuo_de_zimu(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        lu = self._tuo_de_zimu(event)
        if not lu:
            event.ignore()
            return
        event.acceptProposedAction()
        self._jiazai_zimu_lu(lu)

    # ---- 字幕文件的打开 / 保存 ----
    def _dakai_zimu_wenjian(self):
        """（弹框选文件）载入一份 SRT / ASS 到时间轴上"""
        if not self.lujing:
            QtWidgets.QMessageBox.warning(self, "提示", "先打开一个视频")
            return
        qi_lu = self.zimu_wenjian or osp.dirname(self.lujing)
        lu, _guolv = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开字幕",
            qi_lu,
            "字幕文件 (*.srt *.ass *.ssa);;SRT 字幕 (*.srt);;"
            "ASS 字幕 (*.ass);;所有文件 (*)",
        )
        if not lu:
            return
        self._jiazai_zimu_lu(lu)

    def _jiazai_zimu_lu(self, lu):
        """按路径载入字幕：弹框选的、拖进来的、跟视频一块拖进来的都走这儿"""
        if not lu or not osp.isfile(lu):
            return False
        try:
            zimu = _du_zimu_wenjian(lu)
        except OSError as cuowu:
            QtWidgets.QMessageBox.critical(
                self, "错误", f"读不了这个字幕：\n{cuowu}"
            )
            return False
        if not zimu:
            QtWidgets.QMessageBox.warning(
                self, "提示", "这份字幕里没读出字幕行"
            )
            return False

        geshi = osp.splitext(lu)[1].lower().lstrip(".")
        # 打开的是 ASS / SSA：把原结构留着，保存时按它写（样式之类不丢）。
        # 得赶在刷列表之前拆好 —— 列表里的「样式 / 说话人」两列就是从这儿认的。
        self._ass_yuan = (
            _chai_ass(_du_wenben(lu)) if geshi in ("ass", "ssa") else None
        )
        self._ass_qing_jilu = None

        jiu = self.zimu_mianban.zimu_shu()
        self.zimu_mianban.shezhi_zimu(zimu)
        self.zimu_mianban.shezhi_shichang(self.shichang_ms)
        self.zimu_mianban.shezhi_xuan_zhong(-1)
        self.shijianzhou.shezhi_zimu(zimu)
        self.shijianzhou.shezhi_xuan_zhong(-1)
        self._shezhi_bianji_zimu(zimu, -1)
        self.shijianzhou.gundong_dao_ms(0, 0)
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()
        self.zimu_wenjian = lu
        self.zimu_geshi = geshi
        self._zidong_beifen(lu)     # 打开就先把它抄一份进「自动备份」（没改过之前）
        self._zidong_gai_dong = False
        self._zidong_shange = [(int(q), int(z), str(w)) for q, z, w in zimu]
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已载入字幕 {len(zimu)} 条 · {osp.basename(lu)}"
            + (f"（替换原有 {jiu} 条）" if jiu else "")
        )
        logger.info(f"视频工作台：载入字幕 {lu}，{len(zimu)} 条")
        self._chexiao_qingkong(f"载入 {osp.basename(lu)}")
        return True

    def _cun_zimu(self):
        """保存字幕（按钮 / Ctrl+S）：有目标文件就直接存回去，没有就弹另存为"""
        zimu = self.zimu_mianban.zimu_liebiao()
        if not zimu:
            self.shezhi_mianban.zhuangtai_shezhi("还没有字幕可保存")
            return
        if self.zimu_wenjian:
            self._xie_zimu_dao(self.zimu_wenjian, zimu)
            return
        self._ling_cun_zimu()

    def _ling_cun_zimu(self):
        """另存为：让用户选路径和格式"""
        zimu = self.zimu_mianban.zimu_liebiao()
        if not zimu:
            self.shezhi_mianban.zhuangtai_shezhi("还没有字幕可保存")
            return
        mo = osp.splitext(osp.basename(self.lujing or "字幕"))[0] or "字幕"
        qian = self.zimu_wenjian or osp.join(
            osp.dirname(self.lujing or ""), f"{mo}.srt"
        )
        lu, guolv = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存字幕",
            qian,
            "SRT 字幕 (*.srt);;ASS 字幕 (*.ass)",
        )
        if not lu:
            return
        if not osp.splitext(lu)[1]:
            lu += ".ass" if "ASS" in (guolv or "") else ".srt"
        self._xie_zimu_dao(lu, zimu)

    def _xie_zimu_dao(self, lu, zimu):
        """把字幕写到指定文件，并把它记成之后 Ctrl+S 的目标"""
        self._zidong_beifen(lu)     # 覆盖前先把原来那份抄进「自动备份」
        try:
            _xie_zimu_dao_wenjian(
                lu, zimu, self._ass_yuan, self._zimu_fujia(zimu)
            )
        except OSError as cuowu:
            QtWidgets.QMessageBox.critical(
                self, "错误", f"写不了这个文件：\n{cuowu}"
            )
            return False
        self.zimu_wenjian = lu
        self.zimu_geshi = osp.splitext(lu)[1].lower().lstrip(".")
        self._zidong_gai_dong = False
        self._zidong_shange = [(int(q), int(z), str(w)) for q, z, w in zimu]
        self.shezhi_mianban.zhuangtai_shezhi(
            f"字幕已保存 · {osp.basename(lu)}（{len(zimu)} 条）"
        )
        logger.info(f"视频工作台：字幕已保存 {lu}，{len(zimu)} 条")
        return True

    def _tingzhi_yunsuan(self):
        """面板点了「停止」"""
        if self.yunsuan is None and self.chuliqi is None:
            # 模型还在后台加载：标记取消，加载完那边会自己丢掉
            if self._jiazai_xc is not None and self._jiazai_xc.isRunning():
                self._jiazai_quxiao = True
                self.shezhi_mianban.zhuangtai_shezhi("已取消（模型加载中）")
            self.shezhi_mianban._suoding(False)
            return

        if self.yunsuan is not None:
            self.yunsuan.tingzhi()
            # 线程真停了才丢引用：还在跑就被回收，Qt 会直接崩
            if not self.yunsuan.isRunning():
                self.yunsuan = None
        self._tingzhi_tuili_qu()
        # 扫描时画面是一帧帧贴上去的（老路）；停了交还给 mpv
        self.huamian.qingkong()

        # 跟随播放这一路的最后一段要在停止时补收尾
        if self.chuliqi is not None and self.yunsuan_moshi == "gensui":
            try:
                xin = self.chuliqi.jie_shu()
                if xin is not None:
                    self._jia_zimu_kuai(*xin)
                if self.chuliqi.yong_ocr and self.chuliqi.shuchu_mulu:
                    self.chuliqi.xie_zimu_wenjian(self.chuliqi.shuchu_mulu)
            except Exception as cuowu:  # noqa
                logger.error(f"视频工作台：收尾失败 {cuowu}")

        yi_cun = self.chuliqi.yi_cun if self.chuliqi is not None else 0
        self.chuliqi = None
        self.yunsuan_moshi = ""
        self.shezhi_mianban._suoding(False)
        self.shezhi_mianban.zhuangtai_shezhi(
            f"已停止 · 已存 {yi_cun} 张 · 字幕 {self.zimu_mianban.zimu_shu()} 条"
        )
        logger.info(f"视频工作台：已停止，存图 {yi_cun} 张")

    def _shoudao_huamian_jingzhi(self, jingzhi):
        """面板改了「扫描时画面静止」

        扫描线程每帧都会读这个开关，所以扫描途中也能随时改：勾上马上不刷
        画面，取消马上恢复跟着刷，不用停下重来。
        """
        jingzhi = bool(jingzhi)
        if isinstance(self.yunsuan, SaomiaoXiancheng):
            self.yunsuan.hua_mian_bu_dong = jingzhi
            logger.info(f"视频工作台：扫描时画面静止 = {jingzhi}")

    def _saomiao_jieshu(self, chenggong, shuoming, wenjian):
        """全片扫描跑完了"""
        self.shezhi_mianban._suoding(False)
        self._tingzhi_tuili_qu()
        # 扫描时画面是一帧帧贴上去的（老路）；扫完交还给 mpv
        self.huamian.qingkong()
        yi_cun = self.chuliqi.yi_cun if self.chuliqi is not None else 0
        zimu = self.zimu_mianban.zimu_shu()
        chu = f"{shuoming} · 已存 {yi_cun} 张 · 字幕 {zimu} 条"
        if wenjian:
            chu += f" · 已写出 {len(wenjian)} 个字幕文件"
        self.shezhi_mianban.zhuangtai_shezhi(chu)
        if not chenggong:
            QtWidgets.QMessageBox.warning(self, "提示", shuoming)
        logger.info(f"视频工作台：{chu}")

    # --------------------------------------------------------------
    # 其它
    # --------------------------------------------------------------
    def _xuan_video(self):
        lujing, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频文件", "", SHIPIN_FILTER
        )
        if lujing:
            self.dakai(lujing)

    def _shuaxin_anniu(self):
        you = bool(self.lujing)
        for a in (
            self.a_bofang,
            self.a_tui1miao,
            self.a_jin1miao,
            self.a_shang_yizhen,
            self.a_xia_yizhen,
            self.a_jingyin,
        ):
            a.setEnabled(you)
        self.yinliang_tiao.setEnabled(you)
        self.beisu_kuang.setEnabled(you)

    def _xinjian_zimu_kongbai(self):
        """没选中字幕块时按回车：以播放头为起点，建一条 XINJIAN_ZIMU_MS 的空白字幕块

        新块不选中 —— 接着按回车就是接着往后建，不用先取消选择。
        """
        if self.shichang_ms <= 0:
            return False
        qi = int(self.shijianzhou.bofangtou_ms())
        zhi = min(qi + XINJIAN_ZIMU_MS, int(self.shichang_ms))
        if zhi <= qi:
            return False
        self._jia_zimu_kuai(qi, zhi, "")
        return True

    def resizeEvent(self, event):
        """窗口尺寸变了，字幕条那行省略号得按新宽度重算"""
        super().resizeEvent(event)
        if getattr(self, "zimu_mianban", None) is None:
            return
        self._zimu_tiao_wen = None
        self._shuaxin_zimu_tiao()

    def _biping_zhujiemian_kuangjie(self):
        """把主界面那批"应用程序级"快捷键临时降成"窗口级"

        主界面菜单 action 是在 utils.qt.new_action 里统一建的，那里把上下文写死成
        ApplicationShortcut —— 焦点落在视频工作台（主窗口的子窗口）上也照样触发。
        结果 Q/W/E/R 这种单键跟主界面撞车，Qt 打歧义警告。降成窗口级之后，
        只有主窗口自己是活动窗口时才响应，视频工作台里就归本窗口独占了。

        还有一类不是 QAction 而是 QShortcut：主界面按钮的快捷键
        （label_widget._set_button_application_shortcut，比如"取消"按钮 = Alt+S）
        也是 ApplicationShortcut，一样全局抢键，上面那遍 QAction 收不到它。
        不一起降下来，本窗口的 Alt+S 就跟它撞成歧义，Qt 两个都不响应 ——
        也就是按了没反应。所以这里连 QShortcut 一起收。

        扫描范围：从直接父级（标注面板）一路往上扫到主窗口，两级都收 ——
        menu 那批 action 挂在主窗口上、按钮那批 QShortcut 挂在标注面板上，
        只扫一级会漏掉另一级。同一个对象被扫到两遍也没事，按 id 去重。

        往上走用 parentWidget() 而不是 parent()：标注面板（LabelingWrapper）
        自己存了个叫 parent 的属性，把 QObject.parent() 顶掉了，调它直接报错。
        """
        gen = []
        shang = QtWidgets.QWidget.parentWidget(self)
        while shang is not None:
            gen.append(shang)
            shang = QtWidgets.QWidget.parentWidget(shang)
        yikan = set()
        for yi in gen:
            for dongzuo in yi.findChildren(QtWidgets.QAction):
                if dongzuo.shortcutContext() != Qt.ApplicationShortcut:
                    continue
                if id(dongzuo) in yikan:
                    continue
                yikan.add(id(dongzuo))
                self._beiping_kuangjie.append(dongzuo)
                dongzuo.setShortcutContext(Qt.WindowShortcut)
            for jian in yi.findChildren(QtWidgets.QShortcut):
                if jian.context() != Qt.ApplicationShortcut:
                    continue
                # 本窗口自己那批别动（它们挂在 self 底下，也在扫描范围里）
                fu = jian.parent()
                if fu is self or (
                    isinstance(fu, QtWidgets.QWidget)
                    and self.isAncestorOf(fu)
                ):
                    continue
                if id(jian) in yikan:
                    continue
                yikan.add(id(jian))
                self._beiping_kuangjie.append(jian)
                jian.setContext(Qt.WindowShortcut)

    def _huanyuan_zhujiemian_kuangjie(self):
        """关窗口时把主界面快捷键的上下文还回去"""
        for dongzuo in self._beiping_kuangjie:
            if isinstance(dongzuo, QtWidgets.QShortcut):
                dongzuo.setContext(Qt.ApplicationShortcut)
            else:
                dongzuo.setShortcutContext(Qt.ApplicationShortcut)
        self._beiping_kuangjie = []

    def eventFilter(self, duixiang, shijian):
        """视频工作台的全局按键：Tab / ` / ~

        这两个键在视频工作台里哪儿按都好使（焦点在字幕编辑框里照样），而且
        不许落进输入框变成 Tab 缩进 / 波浪号。

        不走 QShortcut 的原因：焦点在 QPlainTextEdit（字幕编辑框）里时，它会
        把 Tab、波浪号当成自己要输入的字符，还会抢先声明"这键归我"，快捷键
        就轮不上了。在应用级把按键拦下来最稳：是这两个键就直接办功能，
        事件吃掉、不再往下送。
        """
        if shijian.type() != QtCore.QEvent.KeyPress or not self.isVisible():
            return super().eventFilter(duixiang, shijian)
        # 只拦本窗口里的按键：别去抢主界面或别处的 Tab / 波浪号。
        # 查找 / 替换 / 选择那几个小窗口是独立的顶层窗口（它们自己要用 Tab
        # 切焦点、要能正常打字），所以按"顶层窗口是不是自己"来判，
        # 不能按血缘关系 —— 它们挂在工作台底下，血缘上一路都是自己人。
        if isinstance(duixiang, QtWidgets.QWidget):
            if duixiang is not self and duixiang.window() is not self:
                return super().eventFilter(duixiang, shijian)
        if shijian.modifiers() & (
            Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier
        ):
            return super().eventFilter(duixiang, shijian)
        jian = shijian.key()
        if jian == Qt.Key_Tab:
            self._tab_an_xia()
            return True
        if jian in (Qt.Key_QuoteLeft, Qt.Key_AsciiTilde):
            self._bofang_dangqian_kuai()
            return True
        return super().eventFilter(duixiang, shijian)

    def closeEvent(self, event):
        # 查找 / 替换 / 选择那几个小窗口跟着一起收掉（样式编辑器连同它上面
        # 打开的那两个自动化配置窗口一并收掉）
        for na in ("_ss_zhao", "_ss_huan", "_ss_xuan", "_yangshi_chuang"):
            chuang = getattr(self, na, None)
            if chuang is not None:
                chuang.close()
                setattr(self, na, None)
        if self._yingyong_lan_jian is not None:
            self._yingyong_lan_jian.removeEventFilter(self)
            self._yingyong_lan_jian = None
        self._zidong_jishi.stop()
        self._cun_jiemian_buju()          # 记下窗口位置 / 标签页 / 分栏 / 缩放
        self._huanyuan_zhujiemian_kuangjie()
        self._guanbi_video()
        # 先松开 mpv 的渲染上下文，再把内核整个关掉（顺序反了会崩）
        self.huamian.shifang()
        self.mpv.guan()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        jian = event.key()
        if jian in (Qt.Key_Return, Qt.Key_Enter):
            # 没选中字幕块时，回车 = 在播放头位置建一条空白字幕块
            if not self.shijianzhou.xuan_zhong_liebiao():
                if self._xinjian_zimu_kongbai():
                    return
        if jian == Qt.Key_Space:
            self._qiehuan_bofang()
            return
        if jian == Qt.Key_Left:
            self._fangxiangjian_tiaobu(-1)
            return
        if jian == Qt.Key_Right:
            self._fangxiangjian_tiaobu(1)
            return
        if jian == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


def dakai_video_gongzuotai(parent=None, video_lujing=None, zimu_lujing=None):
    """入口：没给路径就弹框选；打开视频工作台窗口并返回它

    zimu_lujing 给了就顺手把这份字幕载进来（主界面同时拖视频 + 字幕时走这条）。
    """
    if not video_lujing or not osp.isfile(video_lujing):
        video_lujing, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent, "打开视频文件", "", SHIPIN_FILTER
        )
        if not video_lujing or not osp.isfile(video_lujing):
            return None

    chuang = VideoWorkDialog(parent)
    chuang.setAttribute(Qt.WA_DeleteOnClose, True)
    chuang.show()
    chuang.raise_()
    chuang.activateWindow()
    if not chuang.dakai(video_lujing):
        chuang.close()
        return None
    if zimu_lujing and osp.isfile(zimu_lujing):
        chuang._jiazai_zimu_lu(zimu_lujing)
    # 记进"最近打开的视频"，下次直接从菜单点开，不用再翻目录
    if parent is not None and hasattr(parent, "add_recent_video"):
        parent.add_recent_video(video_lujing)
    return chuang
