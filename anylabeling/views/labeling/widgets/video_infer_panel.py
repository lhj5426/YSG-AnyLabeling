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

import json
import math
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
from anylabeling.views.labeling.widgets.video_work_styles import (
    du_ku,
    ku_zai,
    lie_ku,
    shan_ku,
    xie_ku,
    xin_ku,
)


# =====================================================================
# 用户可调参数（只改这一段）
# =====================================================================
MULU_HOUZHUI = "_字幕提取"      # 输出文件夹：建在视频旁边，名字 = 视频名 + 这个后缀
YANSE_KUANG = "#00E5FF"         # 检测框颜色
YANSE_KUANG_ZI = "#FF453A"      # 检测框上的类别文字颜色
YANSE_ZIMU = "#2F7FE0"          # 字幕块颜色（时间轴上的那种蓝，所有块都一样）
YANSE_ZIMU_ZI = "#FFFFFF"       # 字幕块上的文字颜色（块底色偏暗时用这个）
YANSE_ZIMU_ZI_AN = "#111111"    # 字幕块底色偏亮（比如套了白样式）时，块上的文字改用这个
YANSE_ZIMU_LIANG = 140          # 块底色亮过这个数就用黑字（0-255，越大越容易用白字）
YANSE_ZIMU_XUAN = "#22C55E"     # 选中的字幕块：只沿块内部描一圈绿框，底色文字都不动
YANSE_ZIMU_ZAI = "#FF5252"      # 播放头正压着的那块：描边换这个红（只是"正播到这块"的提示，不算选中）
YANSE_ZHUSHI_HANG = "#F05F27"   # 注释行在列表里的整行底色（橘红：一眼看出这条被藏起来了）
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
ZIMU_BIANJI_ZIHAO_MOREN = 13    # 「字幕编辑」编辑框的字号（px），没存过就用这个
ZIMU_BIANJI_ZIHAO_ZUI_XIAO = 10  # Ctrl+滚轮 能缩到多小
ZIMU_BIANJI_ZIHAO_ZUIDA = 100    # Ctrl+滚轮 能放到多大
ZIMU_BIANJI_ZIHAO_PEIZHI = "jiemian/bianji_zihao"   # 记住编辑框字号用的配置项名
ZIMU_HUANHANG_SE = "#33C33C"    # 编辑框里换行标签 \N 的字色（绿，一眼看出断在哪儿）
BIANJI_ZITI = "新兰圆-B"        # 编辑框用的字体（照 ASS 里那个名字写）；系统没装就退回默认字体
_BIANJI_ZITI_JI = None          # 查过的结果（None = 还没查）
ZIMU_LIEBIAO_ZIHAO_MOREN = 13    # 「字幕列表」那张表的字号（px），没存过就用这个
ZIMU_LIEBIAO_ZIHAO_ZUI_XIAO = 10  # Ctrl+滚轮 在字幕列表上能缩到多小
ZIMU_LIEBIAO_ZIHAO_ZUIDA = 100    # Ctrl+滚轮 在字幕列表上能放到多大
ZIMU_LIEBIAO_ZIHAO_PEIZHI = "jiemian/liebiao_zihao"   # 记住字幕列表字号用的配置项名
ZIMU_LIEBIAO_ZITI = "新兰圆-B"   # 字幕列表字体（跟编辑框一个脸）；没装就按链子退回
ZIMU_LIEBIAO_DISE = "#595151"    # 字幕列表底色（照 AEG）
ZIMU_LIEBIAO_BIAN = "#ffffff"    # 字幕列表的框线和格线（照 AEG，白的）
ZIMU_LIEBIAO_ZI = "#ffffff"      # 字幕列表的字色（底色是固定的深色，字就纯白）
ZIMU_LIEBIAO_XU_DISE = "#000000"  # 序号列底色（照 AEG，黑的）
ZIMU_LIEBIAO_XU_ZI = "#ffffff"   # 序号列字色（照 AEG，白的）
ZIMU_LIEBIAO_XUAN = "#219B17"    # 列表里选中那几行的底色（绿）
ZIMU_LIEBIAO_BOFANG = "#1234EE"  # 列表里正播到的那一行的底色（蓝）
_LIEBIAO_ZITI_JI = None         # 查到的字体名（空 = 还没查到，每次都会重查一次）
_LIEBIAO_ZITI_MEI_BAO = False   # "没找到"这句话报过没有（免得反复刷屏）
ZIDONGHUA_PEIZHI_MING = "zimu_zidonghua.json"   # 老版本的自动化脚本配置（现已并进 gongzuotai.ini，这个名字只用于搬家）
BUJU_PEIZHI_MING = "gongzuotai.ini"    # 视频工作台的全部记忆 / 配置都存这一个文件，放软件根目录
CAOWEI_SHU = 12                 # 「说话人 + 样式」槽位个数（对 F1 ~ F12）
KUOHAO_PAI_CHU_ZI = "旁白"      # 批量加「」：说话人 / 样式里含这几个词的，不套「」
KUOHAO_PAI_CHU_FU = "「」『』（）()"   # 批量加「」：文本里已经有这些符号的，不再套「」
# =====================================================================


# ---------------------------------------------------------------- 界面记忆配置文件
def buju_peizhi_lu():
    """界面记忆的 ini（软件根目录下，跟自动化脚本配置放一块）

    本文件在 <软件根>/anylabeling/views/labeling/widgets/ 下面，
    往上走五层就是软件根目录。
    """
    gen = osp.dirname(
        osp.dirname(
            osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        )
    )
    return osp.join(gen, BUJU_PEIZHI_MING)


_JIU_GAO_JI_LU = osp.join(osp.expanduser("~"), ".ysg_video_zimu_gao.json")   # 老版本存"字幕块行高 / 波形倍数"的地方

# 老版本写在注册表里的项名 -> 现在 ini 里的项名
_JIU_JIAN_MAP = {
    "video_work_chuangkou_weizhi": "jiemian/chuangkou_weizhi",
    "video_work_fenlan_bili": "jiemian/fenlan_bili",
    "video_work_tab_ye": "jiemian/tab_ye",
    "video_work_shijianzhou_suofang": "jiemian/shijianzhou_suofang",
    "video_work_liebiao_zai_xia": "jiemian/liebiao_zai_xia",
    "video_work_shangxia_tuoguo": "jiemian/shangxia_tuoguo",
    "video_work_bianji_zihao": "jiemian/bianji_zihao",
}


def _qian_yi_jiu_peizhi():
    """把老版本散落在各处的东西搬进 gongzuotai.ini（只在 ini 还不存在时搬一次）

    老版本有三个地方：注册表（窗口位置 / 分栏 / 标签页 / 缩放 / 字号）、
    C 盘用户目录的 .ysg_video_zimu_gao.json（字幕块行高 / 波形倍数）、
    软件根目录的 zimu_zidonghua.json（自动化脚本配置）。搬完就统一了，
    之后这些老地方不再读写。
    """
    ini = buju_peizhi_lu()
    if osp.exists(ini):
        return
    try:
        xin = QtCore.QSettings(ini, QtCore.QSettings.IniFormat)
        xin.setIniCodec("UTF-8")
        # 1) 注册表：除了认识的那几个，其它 video_work_ 开头的也一并搬过去
        try:
            jiu = QtCore.QSettings("anylabeling", "anylabeling")
            for jian_ming in jiu.allKeys():
                if not jian_ming.startswith("video_work_"):
                    continue
                xin.setValue(
                    _JIU_JIAN_MAP.get(jian_ming, jian_ming), jiu.value(jian_ming)
                )
        except Exception:  # noqa
            pass
        # 2) C 盘用户目录那份 json：字幕块每行高、波形振幅倍数
        try:
            with open(_JIU_GAO_JI_LU, "r", encoding="utf-8") as wj:
                ji = json.load(wj)
            if isinstance(ji, dict):
                if ji.get("meihang_gao"):
                    xin.setValue("jiemian/meihang_gao", ji["meihang_gao"])
                if ji.get("bo_fangda"):
                    xin.setValue("jiemian/bo_fangda", ji["bo_fangda"])
        except Exception:  # noqa
            pass
        # 3) 根目录那份 json：自动化脚本配置（拆成一行一项搬过去）
        try:
            jiu_lu = osp.join(osp.dirname(ini), ZIDONGHUA_PEIZHI_MING)
            with open(jiu_lu, "r", encoding="utf-8") as wj:
                jiu_pei = json.load(wj)
            for i, x in enumerate((jiu_pei.get("caowei") or [])[:CAOWEI_SHU]):
                xin.setValue(
                    f"zidonghua/{i + 1}_shuohua", str((x or {}).get("shuohua") or "")
                )
                xin.setValue(
                    f"zidonghua/{i + 1}_yangshi",
                    str((x or {}).get("yangshi") or "Default"),
                )
            for xiang in ("paichu_ci", "paichu_fuhao"):
                if jiu_pei.get(xiang):
                    xin.setValue(
                        f"zidonghua/{xiang}",
                        "|".join(str(x) for x in jiu_pei[xiang]),
                    )
        except Exception:  # noqa
            pass
        xin.sync()
    except Exception:  # noqa
        pass


def _kai_buju_ini():
    """开一个读写 gongzuotai.ini 的 QSettings"""
    pei = QtCore.QSettings(buju_peizhi_lu(), QtCore.QSettings.IniFormat)
    pei.setIniCodec("UTF-8")   # 中文原样写进文件，别转义成 \x5f20 那样
    return pei


def buju_qsettings():
    """视频工作台的界面记忆：软件根目录下的 gongzuotai.ini"""
    _qian_yi_jiu_peizhi()
    return _kai_buju_ini()


# ---------------------------------------------------------------- 编辑框字号
def du_bianji_zihao():
    """上次 Ctrl+滚轮 调出来的编辑框字号；没存过 / 存坏了就用默认

    跟「谁在下面」存在同一个 ini 里，下次开软件还是上次那个大小。
    """
    try:
        zhi = buju_qsettings().value(
            ZIMU_BIANJI_ZIHAO_PEIZHI, None
        )
        if zhi is None:
            return ZIMU_BIANJI_ZIHAO_MOREN
        return max(
            ZIMU_BIANJI_ZIHAO_ZUI_XIAO,
            min(ZIMU_BIANJI_ZIHAO_ZUIDA, int(zhi)),
        )
    except (TypeError, ValueError):
        return ZIMU_BIANJI_ZIHAO_MOREN


def xie_bianji_zihao(zihao):
    """把编辑框字号记下来"""
    try:
        q = buju_qsettings()
        q.setValue(ZIMU_BIANJI_ZIHAO_PEIZHI, int(zihao))
        q.sync()
    except Exception:  # noqa
        pass


# ---------------------------------------------------------------- 字幕列表字号
def du_liebiao_zihao():
    """上次 Ctrl+滚轮 调出来的字幕列表字号；没存过 / 存坏了就用默认"""
    try:
        zhi = buju_qsettings().value(ZIMU_LIEBIAO_ZIHAO_PEIZHI, None)
        if zhi is None:
            return ZIMU_LIEBIAO_ZIHAO_MOREN
        return max(
            ZIMU_LIEBIAO_ZIHAO_ZUI_XIAO,
            min(ZIMU_LIEBIAO_ZIHAO_ZUIDA, int(zhi)),
        )
    except (TypeError, ValueError):
        return ZIMU_LIEBIAO_ZIHAO_MOREN


def xie_liebiao_zihao(zihao):
    """把字幕列表字号记下来"""
    try:
        q = buju_qsettings()
        q.setValue(ZIMU_LIEBIAO_ZIHAO_PEIZHI, int(zihao))
        q.sync()
    except Exception:  # noqa
        pass


# ---------------------------------------------------------------- 自动化脚本配置
def _zidonghua_mo_ren():
    """没配过的时候的默认值"""
    return {
        "caowei": [
            {"shuohua": "", "yangshi": "Default"} for _ in range(CAOWEI_SHU)
        ],
        "paichu_ci": ["旁白", "narration", "Narration"],
        "paichu_fuhao": ["「", "」", "『", "』", "（", "）", "(", ")"],
    }


def _caowei_jian(hao, xiang):
    """槽位项名：zidonghua/1_shuohua 这种（hao 从 1 数起）"""
    return f"zidonghua/{hao}_{xiang}"


def _chai_yi_hang(zhi, mo):
    """ini 里「排除词 / 符号」是一行用 | 串起来的；拆回成列表"""
    if zhi is None:
        return list(mo)
    return [x for x in str(zhi).split("|") if x != ""]


def du_zidonghua_peizhi():
    """读配置（跟界面记忆一起存在 gongzuotai.ini 里）；没配过 / 读坏了都退回默认值"""
    mo = _zidonghua_mo_ren()
    try:
        pei = buju_qsettings()
        if not pei.contains("zidonghua/1_shuohua") and not pei.contains(
            "zidonghua/1_yangshi"
        ):
            return mo
        cao = [
            {
                "shuohua": str(pei.value(_caowei_jian(i + 1, "shuohua"), "") or ""),
                "yangshi": str(pei.value(_caowei_jian(i + 1, "yangshi"), "") or "Default"),
            }
            for i in range(CAOWEI_SHU)
        ]
        return {
            "caowei": cao,
            "paichu_ci": _chai_yi_hang(pei.value("zidonghua/paichu_ci", None), mo["paichu_ci"]),
            "paichu_fuhao": _chai_yi_hang(pei.value("zidonghua/paichu_fuhao", None), mo["paichu_fuhao"]),
        }
    except Exception:  # noqa
        return mo


def xie_zidonghua_peizhi(pei):
    """写配置；写不进去返回 False（外面提示一声就行，不崩）"""
    try:
        q = buju_qsettings()
        for i, x in enumerate((pei.get("caowei") or [])[:CAOWEI_SHU]):
            q.setValue(_caowei_jian(i + 1, "shuohua"), str((x or {}).get("shuohua") or ""))
            q.setValue(
                _caowei_jian(i + 1, "yangshi"),
                str((x or {}).get("yangshi") or "Default"),
            )
        q.setValue(
            "zidonghua/paichu_ci",
            "|".join(str(x) for x in (pei.get("paichu_ci") or [])),
        )
        q.setValue(
            "zidonghua/paichu_fuhao",
            "|".join(str(x) for x in (pei.get("paichu_fuhao") or [])),
        )
        q.sync()
    except Exception as cuowu:  # noqa
        logger.error(f"自动化脚本配置写不进去：{cuowu}")
        return False
    return True


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


def _ys_biao_ti():
    """配置弹窗里那种小字：列标题 / 说明 / 提示"""
    c = _ys()
    return f"QLabel {{ color: {c['wenzi_ci']}; font-size: 12px; }}"


def _bianji_ziti_ming():
    """编辑框想用的那个字体在系统里叫什么；没装就返回 \"\"

    系统里新兰圆是拆成 -B / -M / -R 三个家族的（这三个内部又是
    XinLanYuan 一族），所以先按全名找，找不到再按「新兰圆」/ xinlanyuan
    松一点找。都没找到就当没装，样式里不写 font-family，Qt 用默认字体。
    """
    global _BIANJI_ZITI_JI
    if _BIANJI_ZITI_JI is not None:
        return _BIANJI_ZITI_JI
    zhao = ""
    try:
        men = QtGui.QFontDatabase().families()
        for m in men:
            if m == BIANJI_ZITI:
                zhao = m
                break
        if not zhao:
            for m in men:
                if "新兰圆" in m or "xinlanyuan" in m.lower():
                    zhao = m
                    break
    except Exception:  # noqa
        zhao = ""
    _BIANJI_ZITI_JI = zhao
    return zhao


def _liebiao_ziti_ming():
    """字幕列表想用的那个字体在系统里叫什么；没装就返回 ""

    先按全名「思源黑体 CN Medium」找，找不到就按别的写法挨个试（思源黑体在
    系统里可能叫 Source Han Sans CN Medium）；再找不到就松一点按「思源黑体」/
    source han sans 找。都没有就试一下 Noto Sans SC —— 那是思源黑体的免费
    发布版，同一个字，装上哪个都一样。全没有就当没装，样式里不写
    font-family，Qt 用默认字体（不会出方块）。
    """
    global _LIEBIAO_ZITI_JI, _LIEBIAO_ZITI_MEI_BAO
    if _LIEBIAO_ZITI_JI:
        return _LIEBIAO_ZITI_JI
    zhao = ""
    try:
        men = list(QtGui.QFontDatabase().families())
        huan = (
            ZIMU_LIEBIAO_ZITI,
            "新兰圆-M",
            "新兰圆-R",
            "Source Han Sans CN Medium",
            "思源黑体 CN",
            "Source Han Sans CN",
            "Noto Sans SC Medium",
            "Noto Sans SC",
        )
        for xiang in huan:
            for m in men:
                if m == xiang:
                    zhao = m
                    break
            if zhao:
                break
        if not zhao:
            for m in men:
                di = m.lower()
                if (
                    "新兰圆" in m
                    or "思源黑体" in m
                    or "source han sans" in di
                ):
                    zhao = m
                    break
    except Exception:  # noqa
        zhao = ""
    if zhao:
        _LIEBIAO_ZITI_JI = zhao
    elif not _LIEBIAO_ZITI_MEI_BAO:
        # 没找到就别记住（记住空的会导致以后永远认不出来）；报一次给日志
        _LIEBIAO_ZITI_MEI_BAO = True
        logger.info("字幕列表字体：没找到思源黑体，用系统默认字体")
    return zhao


def _ys_zihao_ti():
    """Ctrl+滚轮 报字号的那个小浮标：深底浅字，跟主题反过来更显眼"""
    c = _ys()
    return f"""
    QLabel {{
        background-color: {c['wenzi']};
        color: {c['beijing2']};
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 13px;
        font-family: "Microsoft YaHei";
    }}
    """


def _ys_bianji_kuang(zihao=None):
    """编辑框样式；zihao 是字号（px），不给就用默认的"""
    if zihao is None:
        zihao = ZIMU_BIANJI_ZIHAO_MOREN
    c = _ys()
    ziti = _bianji_ziti_ming()
    zi = f'font-family: "{ziti}";\n        ' if ziti else ""
    return f"""
    QPlainTextEdit {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 4px 8px;
        font-size: {int(zihao)}px;
        {zi}font-weight: bold;
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
        padding: 3px 4px;
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


def _ys_zimu_biao(zihao=None):
    """字幕列表（ASS 字段表）：带网格、整行选中

    底色 / 格线照 AEG 来：深底 #595151 + 白线；字号由 Ctrl+滚轮 调
    （不调就按记住的那个），字体优先「思源黑体 CN Medium」。
    """
    if zihao is None:
        zihao = ZIMU_LIEBIAO_ZIHAO_MOREN
    ziti = _liebiao_ziti_ming()
    zi = f'font-family: "{ziti}";\n        ' if ziti else ""
    return f"""
    QTableWidget {{
        background-color: {ZIMU_LIEBIAO_DISE};
        color: {ZIMU_LIEBIAO_ZI};
        border: 1px solid {ZIMU_LIEBIAO_BIAN};
        border-radius: 6px;
        font-size: {int(zihao)}px;
        font-weight: bold;
        {zi}gridline-color: {ZIMU_LIEBIAO_BIAN};
        outline: none;
    }}
    QTableWidget::item {{ padding: 2px 4px; }}
    QTableWidget::item:selected {{
        background-color: {ZIMU_LIEBIAO_XUAN};
        color: #ffffff;
    }}
    QHeaderView::section {{
        background-color: {ZIMU_LIEBIAO_DISE};
        color: {ZIMU_LIEBIAO_ZI};
        border: none;
        border-right: 1px solid {ZIMU_LIEBIAO_BIAN};
        border-bottom: 1px solid {ZIMU_LIEBIAO_BIAN};
        padding: 3px 4px;
        font-size: {int(zihao)}px;
        font-weight: bold;
        {zi}}}
    QTableCornerButton::section {{
        background-color: {ZIMU_LIEBIAO_DISE};
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


def wen_shi_fou(parent, biao_ti, wen, bei_zhu="", shi="是", fou="否",
                qu_xiao=None, moren_shi=True):
    """「是 / 否」两个中文按钮的询问框

    Qt 自带的 Yes / No 在中文界面上就是两个英文单词，太难看了，这儿自己
    挂两个中文按钮。给了 qu_xiao 就多一个「取消」，那会儿返回 None。
    """
    kuang = QtWidgets.QMessageBox(parent)
    kuang.setWindowTitle(str(biao_ti))
    kuang.setIcon(QtWidgets.QMessageBox.Question)
    kuang.setText(str(wen))
    if bei_zhu:
        kuang.setInformativeText(str(bei_zhu))
    an_shi = kuang.addButton(str(shi), QtWidgets.QMessageBox.YesRole)
    an_fou = kuang.addButton(str(fou), QtWidgets.QMessageBox.NoRole)
    an_qu = None
    if qu_xiao is not None:
        an_qu = kuang.addButton(str(qu_xiao), QtWidgets.QMessageBox.RejectRole)
    kuang.setDefaultButton(an_shi if moren_shi else an_fou)
    kuang.exec_()
    dian = kuang.clickedButton()
    if an_qu is not None and dian is an_qu:
        return None
    return dian is an_shi


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


def zhen_tou_ms(zhen, fps, wei=False):
    """第 zhen 帧当块头 / 块尾写进字幕的毫秒（照抄 AEG 的 TimeAtFrame）

    源码：Aegisub-main/libaegisub/common/vfr.cpp，Framerate::TimeAtFrame
    （第 225 行起）。帧率固定时 AEG 的 timecodes[n] 是

        timecodes[n] = n * 1000 / fps      （整数乘除，向下截断，第 144 行）

    写进字幕的时间不是"这一帧的起点"，而是这一帧与邻帧的中点：

        START(n) = timecodes[n-1] + (timecodes[n]   - timecodes[n-1] + 1) / 2
        END(n)   = timecodes[n]   + (timecodes[n+1] - timecodes[n]   + 1) / 2

    好处：① 块尾跟下一块的块头是同一个值，接连的块严丝合缝；② 画面字幕是
    按 [开始, 结束) 半开区间判的（libass 就是），压在帧起点上的话这一帧正好
    落在区间外、不显示 —— 看着就是"少一帧"；③ ASS 只存厘秒，值贴在中点上
    存一次读回来也还是同一帧。

    wei=False 块头（START）/ wei=True 块尾（END）。帧率不对给 None。
    """
    try:
        fps = float(fps or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        return None
    zhen = max(0, int(zhen))
    ben = int(zhen * 1000.0 / fps)                  # timecodes[zhen]
    if wei:
        xia = int((zhen + 1) * 1000.0 / fps)        # timecodes[zhen + 1]
        # 那句 +1 是 AEG 的原话："两帧只差 1 毫秒时要往上进位"
        return max(0, ben + (xia - ben + 1) // 2)
    shang = int((zhen - 1) * 1000.0 / fps)          # timecodes[zhen - 1]
    return max(0, shang + (ben - shang + 1) // 2)


def ms_tou_zhen(ms, fps, wei=False):
    """毫秒 -> 帧号（照抄 AEG 的 FrameAtTime）

    源码：Aegisub-main/libaegisub/common/vfr.cpp，Framerate::FrameAtTime
    （第 197 行起）。AEG 认帧号是**查表**，不是四舍五入：

        EXACT(t) = 最大的 n 使 timecodes[n] <= t       （第 222 行）
        END(t)   = EXACT(t - 1)                        （第 214 行）
        START(t) = END(t) + 1                          （第 212 行）

    把 timecodes[n] = n * 1000 / fps（向下截断）代进去，EXACT 就等价于

        EXACT(t) = ceil(t * fps / 1000) - 1

    （因为 floor(n * 1000 / fps) <= t 等价于 n * 1000 / fps < t + 1）

    于是：
        wei=False 开始帧（START）：ceil(ms * fps / 1000)
        wei=True  结束帧（END）  ：ceil(ms * fps / 1000) - 1

    同一个毫秒值当块尾认到的帧号，比当块头时小 1 —— AEG 就是这样：块尾和
    下一块的块头本来就是同一个值，两头各认各的才对得上。帧率不对给 0。
    """
    try:
        fps = float(fps or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        return 0
    k = int(math.ceil(max(0, int(ms or 0)) * fps / 1000.0))
    return max(0, k - 1 if wei else k)


def _zhen_wenben(ms, fps, wei=None):
    """毫秒 -> 帧号（按帧看时间的时候显示用）；不知道帧率就给 0

    wei 不给 = 时长那种"差多少帧"，直接四舍五入；wei=False 开始帧、
    wei=True 结束帧，照 AEG 认（见 ms_tou_zhen）。
    """
    try:
        fps = float(fps or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        return "0"
    ms = max(0, int(ms or 0))
    if wei is None:
        return str(int(round(ms / 1000.0 * fps)))
    return str(ms_tou_zhen(ms, fps, bool(wei)))


def _zhen_to_ms(zhen, fps, wei=None):
    """帧号 -> 毫秒；帧率不对或者帧号不是数字给 None

    wei 不给 = 时长那种"几帧"，直接乘帧长；wei=False 开始帧、wei=True
    结束帧，照 AEG 算（见 zhen_tou_ms）—— 写回去的值还得能认回同一个帧号。
    """
    try:
        zhen = int(str(zhen).strip())
    except (TypeError, ValueError):
        return None
    try:
        fps = float(fps or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0:
        return None
    if wei is None:
        return int(round(max(0, zhen) / fps * 1000.0))
    return zhen_tou_ms(zhen, fps, bool(wei))


def _ass_shi_jian_wenben(ms):
    """毫秒 -> 0:00:00.00（ASS 里那种时间写法，表格里显示用）"""
    ms = max(0, int(ms or 0))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    x = (ms % 1000) // 10
    return f"{h}:{m:02d}:{s:02d}.{x:02d}"


def zimu_charu_weizhi(liebiao, qi_ms, zhi_ms):
    """新字幕该插在第几行：按开始时间找，开始时间一样就按结束时间

    字幕块是在播放头那儿建的，可能建在片子开头、也可能建在中间，所以不能一律
    堆到列表最后一行 —— 按它的时间插到该在的位置（时间轴、字幕列表都照这个来）。
    """
    wo = (int(qi_ms), int(zhi_ms))
    for i, tiao in enumerate(liebiao or []):
        if wo < (int(tiao[0]), int(tiao[1])):
            return i
    return len(liebiao or [])


def _wenben_to_ms(wenben):
    """把 0:00:05.65 / 00:00:05.650 / 1:05 这种写法读成毫秒；读不出来给 None"""
    tiao = re.sub(r"\s", "", str(wenben or "")).replace("：", ":")
    if not tiao:
        return None
    bu = tiao.split(":")
    if len(bu) not in (2, 3):
        return None
    try:
        if len(bu) == 2:
            shi, fen, miao = 0, int(bu[0]), bu[1].replace("．", ".")
        else:
            shi, fen, miao = int(bu[0]), int(bu[1]), bu[2].replace("．", ".")
        if "." in miao:
            miao, hao = miao.split(".", 1)
        else:
            miao, hao = miao, "0"
        hao = (hao + "000")[:3]
        return ((shi * 60 + fen) * 60 + int(miao)) * 1000 + int(hao)
    except (TypeError, ValueError):
        return None


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

    tiaozheng = pyqtSignal(int)     # 点了某处 -> 请求跳到这个 ms（任意位置）
    # 点了某个字幕块 -> 请求跳到这一段的开头：这是"块头"，认帧得按 AEG 的
    # START 认（见 ms_tou_zhen），跟上面"任意位置"不是一回事，别混着走
    zimu_qi_tiao = pyqtSignal(int)
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
                self.zimu_qi_tiao.emit(int(self._zimu[xu][0]))
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

    tiaozheng = pyqtSignal(int)     # 请求跳到某个 ms（任意位置）
    # 请求跳到某条字幕的开头（块头，认帧按 AEG 的 START 认）
    zimu_qi_tiao = pyqtSignal(int)
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
        self.zhou.zimu_qi_tiao.connect(self.zimu_qi_tiao.emit)
        bu.addWidget(self.zhou, 0)

        self.liebiao = QtWidgets.QListWidget()
        self.liebiao.setStyleSheet(_ys_liebiao())
        self.liebiao.setAlternatingRowColors(False)
        self.liebiao.setToolTip("单击 = 选中这一段；双击 = 改这段的文字")
        self.liebiao.itemDoubleClicked.connect(self._shuang_ji)
        self.liebiao.currentRowChanged.connect(self._xuan_zhong_bian)
        bu.addWidget(self.liebiao, 1)

        self._zimu = []
        # 每条的字段（样式 / 说话人 / 层 / 边距 / 特效 / 注释）：键是那条文字时间
        # 三元组。整份列表换来换去（排序 / 删除 / 合并）时按内容跟着搬，见 _fu_ban。
        self._fu = {}
        self.zidong_tiao = True  # 列表里点一条 -> 画面跟不跟着跳（外面那个开关定）

    # ---- 外部接口 ----
    def shezhi_zidong_tiao(self, kai):
        """「选中字幕时画面跟着跳」开关（照 AEG）：关了 = 点列表只选中，画面不动"""
        self.zidong_tiao = bool(kai)

    def shezhi_shichang(self, ms):
        self.zhou.shezhi_shichang(ms)

    def shezhi_bofangtou(self, ms):
        self.zhou.shezhi_bofangtou(ms)

    def qingkong(self):
        self._zimu = []
        self._fu = {}
        self.liebiao.clear()
        self.zhou.qingkong()
        self.shu_wenben.setText("0 条")

    def tianjia_zimu(self, qi_ms, zhi_ms, wenben):
        """新建一条：按时间插到该在的位置（不是一律堆在最后一行）

        返回插在第几行，外面好把时间轴、字幕编辑列表那几处对齐。
        原来选中的那条按"内容"重新认一遍，插在前面也不会串行。
        """
        hang = self.liebiao.currentRow()
        jiu = self._zimu[hang] if 0 <= hang < len(self._zimu) else None
        tiao = (int(qi_ms), int(zhi_ms), str(wenben or ""))
        wei = zimu_charu_weizhi(self._zimu, tiao[0], tiao[1])
        self._zimu.insert(wei, tiao)
        self._shuaxin()
        if jiu is not None:
            try:
                self.shezhi_xuan_zhong(self._zimu.index(jiu))
            except ValueError:      # 原来那条已经不在表里了
                pass
        return wei

    def shezhi_zimu(self, zimu):
        xin = [tuple(x) for x in (zimu or [])]
        self._fu_ban(xin)
        self._zimu = xin
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

    def zimu_xu_zai_ms(self, ms):
        """这个时刻落在第几条上（带序号的 zimu_zai_ms）；没落在任何一条上给 -1

        「播放头选中」用：播放头压着的到底是哪一条。
        """
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            return -1
        for i, (qi, zhi, _wenben) in enumerate(self._zimu):
            if int(qi) <= ms <= int(zhi):
                return i
        return -1

    def gai_zimu(self, xu, wenben):
        """就地改第 xu 条的文字（列表 + OCR 小时间轴一起刷）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        qi, zhi, _jiu = self._zimu[xu]
        jiu = self._zimu[xu]
        self._zimu[xu] = (qi, zhi, str(wenben or ""))
        self._fu_ban_dan(jiu, self._zimu[xu])
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
        lao = self._zimu[xu]
        self._zimu[xu] = (qi, zhi, wenben)
        self._fu_ban_dan(lao, self._zimu[xu])
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
        lao = self._zimu[xu]
        self._zimu[xu] = (qi_ms, zhi_ms, wenben)
        self._fu_ban_dan(lao, self._zimu[xu])
        self._shuaxin()

    # ---- 每条的字段（样式 / 说话人 / 层 / 边距 / 特效 / 注释）----
    def hang_fu(self, xu):
        """第 xu 条的字段；没设过给空表（外面拿它当"照原样"用）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return {}
        return dict(self._fu.get(tuple(self._zimu[xu])) or {})

    def shezhi_hang_fu(self, xu, fu):
        """给第 xu 条设字段（外面工具栏改的）"""
        xu = int(xu)
        if not (0 <= xu < len(self._zimu)):
            return
        self._fu[tuple(self._zimu[xu])] = dict(fu or {})

    def hang_fu_liebiao(self):
        """整份字段表，跟 zimu_liebiao 一一对应（写 ASS 时要用）"""
        return [dict(self._fu.get(tuple(z)) or {}) for z in self._zimu]

    def fujia_biao(self):
        """整份字段表（键是那一行的内容）：撤销 / 重做存档用，拷一份走"""
        return {tuple(k): dict(v) for k, v in self._fu.items()}

    def shezhi_fujia_biao(self, biao):
        """把整份字段表换回去（撤销 / 重做用）"""
        self._fu = {tuple(k): dict(v) for k, v in (biao or {}).items()}

    def _fu_ban_dan(self, lao, xin):
        """一条的键变了（改文字 / 改时间）：字段跟着挪过去"""
        if tuple(lao) == tuple(xin):
            return
        fu = self._fu.pop(tuple(lao), None)
        if fu:
            self._fu[tuple(xin)] = fu

    def _fu_ban(self, xin):
        """整份列表换掉：按内容把字段搬到新表上（排序 / 删除 / 合并都走这儿）"""
        jiu = {}
        for z in self._zimu:
            fu = self._fu.get(tuple(z))
            if fu:
                jiu[tuple(z)] = dict(fu)
        self._fu = {}
        for z in xin:
            if tuple(z) in jiu:
                self._fu[tuple(z)] = dict(jiu[tuple(z)])

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
        if self.zidong_tiao:
            # 「选中字幕时画面跟着跳」关掉时：只选中，播放头和画面都不动
            self.zimu_qi_tiao.emit(int(self._zimu[hang][0]))

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
class _AssGaoliang(QtGui.QSyntaxHighlighter):
    """编辑区里的代码分色（照 AEG 的 SubsTextEditCtrl 那套）

    AEG 把花括号、反斜杠、标签名、参数各画一种颜色，一眼就能看出哪段是标签、
    标签叫什么名字。这里照抄那份分法：

        {   }       括号       灰
        (   )   ,   小括号逗号  灰
        \\           反斜杠     蓝
        pos c fn …  标签名     品红
        75,705 …    参数       蓝
        \\N          换行标签   绿（这个单独一色，一眼看出这行断在哪儿）

    正文不设色，用编辑框自己的字色。颜色是按我们的深色底调过的，AEG 那套是
    浅色底的配色，直接抄过来会看不清。
    """

    HUI = QtGui.QColor("#9AA0A6")       # 括号 / 小括号 / 逗号
    LAN = QtGui.QColor("#6BB6FF")       # 反斜杠 / 参数
    HONG = QtGui.QColor("#FF7AD9")      # 标签名
    LU = QtGui.QColor(ZIMU_HUANHANG_SE)  # 换行标签 \N

    def highlightBlock(self, wen):
        wen = str(wen or "")
        chang = len(wen)
        i = 0
        while i < chang:
            if wen[i] != "{":
                i += 1
                continue
            jie = wen.find("}", i + 1)          # 这一对花括号到哪儿结束
            if jie < 0:
                jie = chang
            self.setFormat(i, 1, self.HUI)
            k = i + 1
            while k < jie:
                ch = wen[k]
                if ch == "\\":
                    self.setFormat(k, 1, self.LAN)
                    k += 1
                elif ch in "(),":
                    self.setFormat(k, 1, self.HUI)
                    k += 1
                else:
                    ming = re.match(r"\d?[A-Za-z]+", wen[k:jie])
                    if ming:
                        # 标签名：\1c \fn \fscx 这些（前面最多带一个数字）
                        self.setFormat(k, ming.end(), self.HONG)
                        k += ming.end()
                    else:
                        # 剩下的就是参数：75,705 / &H05F0A1& / 30 …
                        can = re.match(r"[^\\,)]+", wen[k:jie])
                        bu = can.end() if can else 1
                        self.setFormat(k, bu, self.LAN)
                        k += bu
            if jie < chang:
                self.setFormat(jie, 1, self.HUI)
            i = jie + 1
        # 换行标签 \N 单独画绿（摆最后，盖掉上面花括号里那层配色）
        for zhao in re.finditer(r"\\N", wen):
            self.setFormat(zhao.start(), 2, self.LU)


def _kuang_jin(wen):
    """字幕文本 -> 编辑框里显示的样子

    字幕文本里换行写的是标签 \\N，框里每个 \\N 后头再跟一个真换行 —— 看着就是
    "这行断在这儿"，同时 \\N 本身还在，能用退格删掉。老数据里万一夹了真换行，
    先归成 \\N 再展开。
    """
    return str(wen or "").replace("\n", "\\N").replace("\\N", "\\N\n")


def _kuang_chu(wen):
    """编辑框里显示的样子 -> 字幕文本

    框里的真换行一律丢掉：换行只认 \\N 这个标签。所以把某个 \\N 删掉就等于取消了
    那一处换行 —— 剩下的换行符跟着没了，两行并回一行。
    """
    return str(wen or "").replace("\n", "")


class _ZimuWenbenKuang(QtWidgets.QPlainTextEdit):
    """字幕编辑框：一离开焦点就说一声「这一条改完了」（好去重写字幕文件）

    Ctrl + 滚轮 = 改框里的字号（跟 Aegisub 那排字号一个意思），不带 Ctrl 的
    滚轮还是正常上下滚。字号写在样式里，不然压不住外面套的那套样式。

    回车 = 换到下一条字幕（列表往下走一行）；Shift + 回车 = 在这条字幕里换行
    （落一个 \\N 标签 + 真换行）。

    框里存的不是字幕原文，是「带 \\N 标记的写法」：每个换行标签后头跟一个真换行，
    \\N 是真字符，能圈能删 —— 删掉就是取消那一处换行。进出用 shezhi_wenben /
    wenben 换算（见 _kuang_jin / _kuang_chu），别直接拿 toPlainText 当字幕文本。

    框里的行内标签（{...}）和 \\N 是分色画的，见 _AssGaoliang。
    """

    likai = pyqtSignal()
    xiayitiao = pyqtSignal()        # 回车：换到下一条字幕

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zihao = du_bianji_zihao()
        self.setStyleSheet(_ys_bianji_kuang(self._zihao))
        self._gaoliang = _AssGaoliang(self.document())

        # Ctrl+滚轮 调字号时，框右上角闪一个小浮标报当前数值，免得瞎调
        self._zihao_ti = QtWidgets.QLabel(self)
        self._zihao_ti.setStyleSheet(_ys_zihao_ti())
        self._zihao_ti.hide()
        self._zihao_ti_ji = QtCore.QTimer(self)
        self._zihao_ti_ji.setSingleShot(True)
        self._zihao_ti_ji.timeout.connect(self._zihao_ti.hide)

    def _tan_zihao_ti(self):
        """滚完字号在框左上角闪一下：「字号 20」「字号 100（最大）」"""
        shuo = f"字号 {self._zihao}"
        if self._zihao >= ZIMU_BIANJI_ZIHAO_ZUIDA:
            shuo += "（最大）"
        elif self._zihao <= ZIMU_BIANJI_ZIHAO_ZUI_XIAO:
            shuo += "（最小）"
        self._zihao_ti.setText(shuo)
        self._zihao_ti.adjustSize()
        self._zihao_ti.move(14, 10)
        self._zihao_ti.show()
        self._zihao_ti.raise_()
        self._zihao_ti_ji.start(1200)

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
            xie_bianji_zihao(zihao)      # 记住，下次开软件还是这个大小
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
        self._tan_zihao_ti()
        event.accept()

    def keyPressEvent(self, event):
        """回车 = 换到下一条字幕；Shift + 回车 = 在字幕里换行

        跟 Aegisub 一个手感：改完这一条敲回车就往下走；本条里要换行（ASS 的
        \\N）就按 Shift + 回车 —— 落下来的是「\\N + 真换行」，那个 \\N 就摆在
        框里，能圈能删。
        """
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                self.textCursor().insertText("\\N\n")
            else:
                self.xiayitiao.emit()
            event.accept()
            return
        super().keyPressEvent(event)
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            # 刚删过东西：哪一行行尾的 \N 没了，那一处的换行也一并去掉
            self._zhengli_huanhang()

    def _zhengli_huanhang(self):
        """行尾的 \\N 被删掉：那一处的换行跟着没了，两行当场并回一行

        规矩：每行行尾挂着 \\N，最后一行没有。所以某行行尾缺 \\N（后面还有行）
        就说明那处的标签被删了 —— 把那个换行符也删掉，删 \\N 就等于取消换行。
        只删换行符本身，撤销栈还是按你操作的那一下算。
        """
        blk = self.document().firstBlock()
        sha = []
        while blk.isValid() and blk.next().isValid():
            if not blk.text().endswith("\\N"):
                sha.append(blk.position() + blk.length() - 1)
            blk = blk.next()
        if not sha:
            return
        ze = QtGui.QTextCursor(self.document())
        ze.beginEditBlock()
        for wei in reversed(sha):
            ze.setPosition(wei)
            ze.deleteChar()
        ze.endEditBlock()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.likai.emit()

    # ---- 框里的显示写法 <-> 字幕文本 ----
    def shezhi_wenben(self, wen):
        """把一条字幕的文字填进框（框里是带 \\N 标记的显示写法）"""
        self.setPlainText(_kuang_jin(wen))

    def wenben(self):
        """取框里的文字，按字幕文本给出去（换行认 \\N，没有真换行）"""
        return _kuang_chu(self.toPlainText())

    def _kuang_wei(self, wei):
        """字幕文本里的位置 -> 框里的位置"""
        wen = self.wenben()
        wei = max(0, min(int(wei), len(wen)))
        return wei + wen[:wei].count("\\N")

    def shuju_wei(self, wei):
        """框里的位置 -> 字幕文本里的位置（工具栏往选中那段加标签要用）"""
        wen = self.toPlainText()
        wei = max(0, min(int(wei), len(wen)))
        return wei - wen[:wei].count("\n")

    def dingwei_shuju(self, qi, zhi):
        """按字幕文本的位置在框里圈出这一段（搜索定位用）"""
        ze = self.textCursor()
        ze.setPosition(self._kuang_wei(qi))
        ze.setPosition(self._kuang_wei(zhi), QtGui.QTextCursor.KeepAnchor)
        self.setTextCursor(ze)

    def insertFromMimeData(self, yuan):
        """粘进来的多行文字：真换行按「\\N + 换行」落进来，不会粘成一坨"""
        wen = str(yuan.text() or "")
        if "\n" in wen:
            self.textCursor().insertText(wen.replace("\n", "\\N\n"))
            return
        super().insertFromMimeData(yuan)


def _yanse_to_ass(yan):
    """QColor -> ASS 写的 &HBBGGRR&（ASS 是蓝绿红倒着写的）"""
    return f"&H{yan.blue():02X}{yan.green():02X}{yan.red():02X}&"


def _zuiduo_zishu(wenben):
    """一条字幕最长那一行有几个字（行内标签不算，AEG 的字符数框就是这个）"""
    chun = _chun_wen(wenben)
    hang = chun.replace("\\N", "\n").replace("\\n", "\n").split("\n")
    return max([len(x.strip()) for x in hang] or [0])


class _ZimuGongjulan(QtWidgets.QWidget):
    """字幕编辑区上方那排（照 AEG 的 SubsEditBox 抄）

    第 1 排：注释 / 样式 + 编辑 / 说话人 / 最长一行字数
    第 2 排：层 / 开始 / 结束 / 时长
    第 3 排：B I U S fn / 四色 / 改完跳下一条
    窗口够宽时第 3 排自动并到第 2 排后面（AEG 的 OnSize 就是这么干的）。

    这排只管"把当前这条的值显示出来"和"把用户改的往外发"，至于怎么落库、怎么
    重画、怎么写回 ASS，全是外面的事。往外发的统一是 gongju_gaile(哪个字段, 值)：
    yang 样式 / shuo 说话人 / ceng 层 / zhushi 注释 / qi zhi 时间 /
    tag 行内标签 / yanse 颜色 / bianji 打开样式编辑器 / xiayihang 新建下一条。
    """

    gongju_gaile = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tian = False      # 正往里填值：这会儿控件自己发的信号不算用户改
        self._zi_anniu = {}
        self._yanse_anniu = {}
        self._qi_ms = 0
        self._zhi_ms = 0
        self._fps = 0.0         # 视频帧率：按帧看时间用（外面 shezhi_fps 给）
        self._zhen = False      # 现在显示的是帧号还是时间
        self._bingpai = False   # 第 3 排现在是不是并到第 2 排后面了
        self._yao_kuan = 0      # 并排需要多宽（量一次记着，见 _pai_yang）
        self._mid = []          # 第 2 排的控件
        self._bot = []          # 第 3 排的控件
        self._ziti = QtGui.QFont()
        self._yanse = {"c1": "#FFFFFF", "c2": "#FF0000",
                       "c3": "#000000", "c4": "#000000"}
        self._wenben_kuang = None   # 编辑框（外面给）：有选中的字就只对那段动手

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(0, 0, 0, 0)
        wai.setSpacing(4)

        # ---- 第一行：注释 / 样式 / 说话人 / 最长一行字数 ----
        y1 = QtWidgets.QHBoxLayout()
        y1.setSpacing(6)

        self.gou_zhushi = QtWidgets.QCheckBox("注释")
        self.gou_zhushi.setToolTip(
            "勾上 = 这条是注释：不画在画面上，存进 ASS 是 Comment 行"
        )
        self.gou_zhushi.setStyleSheet(_ys_xuanxiang())
        y1.addWidget(self.gou_zhushi)

        self.xia_yang = QtWidgets.QComboBox()
        # 宽度不写死：照 AEG 的 top_sizer，样式和说话人各占 2 份富余宽度，
        # 窗口拉宽它俩跟着长、拉窄跟着缩（份数在下面 addWidget 的第二个参数）。
        # 底下这个最小值只保证再窄也不挤成一条线。
        self.xia_yang.setMinimumWidth(110)
        self.xia_yang.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.xia_yang.setMinimumContentsLength(4)
        self.xia_yang.setToolTip("这一条用的样式")
        # 弹出列表往"大而长"上做：行更高、比输入框宽、一次列 20 条。
        # Qt 默认那条又窄又挤，样式名长一点就看不全，翻起来也累。
        bang = self.xia_yang.view()
        bang.setMinimumWidth(240)
        bang.setStyleSheet(
            "QAbstractItemView::item { min-height: 26px; padding-left: 4px; }"
        )
        self.xia_yang.setMaxVisibleItems(20)
        # 滚轮落在它上面 = 上一条 / 下一条换样式（照 AEG），见 eventFilter
        self.xia_yang.installEventFilter(self)
        y1.addWidget(self.xia_yang, 2)

        self.an_yang = QtWidgets.QPushButton("编辑")
        self.an_yang.setToolTip("打开样式编辑器")
        self.an_yang.setStyleSheet(_ys_ci_anniu())
        y1.addWidget(self.an_yang)

        self.xia_shuo = QtWidgets.QComboBox()
        # 跟样式下拉一样按份数伸缩（照 AEG 的 actor_box，那份也是 2）
        self.xia_shuo.setMinimumWidth(100)
        self.xia_shuo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.xia_shuo.setMinimumContentsLength(4)
        self.xia_shuo.setToolTip("说话人（只作标记，不影响画面）")
        y1.addWidget(self.xia_shuo, 2)

        # 最长一行字数：我们没有特效框，它就直接跟在说话人后面，也拿框装起来
        self.lian_zishu = QtWidgets.QLineEdit("0")
        self.lian_zishu.setReadOnly(True)
        self.lian_zishu.setAlignment(Qt.AlignCenter)
        self.lian_zishu.setFixedWidth(46)
        self.lian_zishu.setToolTip("这条字幕最长一行的字数")
        self.lian_zishu.setStyleSheet(_ys_shuru())
        y1.addWidget(self.lian_zishu)

        # 两个界面开关（照 AEG / ARC）：只作开关，不挂快捷键
        self.gou_gen_tiao = QtWidgets.QCheckBox("画面跟随")
        self.gou_gen_tiao.setToolTip(
            "选中字幕时画面跟着跳\n"
            "照 AEG：勾上 = 在字幕列表里点一条，画面跟着跳到它的开头；\n"
            "不勾 = 只选中那一条，播放头和画面都不跟过去"
        )
        self.gou_gen_tiao.setStyleSheet(_ys_xuanxiang())
        y1.addWidget(self.gou_gen_tiao)

        self.gou_shishi_gun = QtWidgets.QCheckBox("时间轴跟随")
        self.gou_shishi_gun.setToolTip(
            "时间轴实时滚动\n"
            "照 ARC：勾上 = 播放时播放头定在视口中间，时间轴往左滚；\n"
            "不勾 = 播放头在时间轴上往前走，走出视野才挪一屏"
        )
        self.gou_shishi_gun.setStyleSheet(_ys_xuanxiang())
        y1.addWidget(self.gou_shishi_gun)

        self.gou_bofangtou_xuanzhong = QtWidgets.QCheckBox("播放头选中")
        self.gou_bofangtou_xuanzhong.setToolTip(
            "播放头压到哪条字幕就自动选中它\n"
            "勾上 = 播放头落在某条字幕的区间里，那条自动被选中（时间轴绿框、\n"
            "字幕列表跟着选、编辑区也切过去）；\n"
            "播放头走到没字幕的地方就保持上一次的选中，不清空"
        )
        self.gou_bofangtou_xuanzhong.setStyleSheet(_ys_xuanxiang())
        y1.addWidget(self.gou_bofangtou_xuanzhong)

        y1.addStretch(1)

        wai.addLayout(y1)

        # ---- 第 2 排：层 / 开始 / 结束 / 时长 ----
        self.pai2 = QtWidgets.QWidget()
        self.h2 = QtWidgets.QHBoxLayout(self.pai2)
        self.h2.setContentsMargins(0, 0, 0, 0)
        self.h2.setSpacing(6)
        self.h2.addStretch(1)   # 末尾留个撑开的空档：控件都挤在左边

        # ---- 第 3 排：B I U S fn / 四色 / 改完跳下一条 ----
        self.pai3 = QtWidgets.QWidget()
        self.h3 = QtWidgets.QHBoxLayout(self.pai3)
        self.h3.setContentsMargins(0, 0, 0, 0)
        self.h3.setSpacing(6)
        self.h3.addStretch(1)

        def _mid(kuang):
            self._mid.append(kuang)
            self.h2.insertWidget(self.h2.count() - 1, kuang)   # 排在空档前
            return kuang

        def _bot(kuang):
            self._bot.append(kuang)
            self.h3.insertWidget(self.h3.count() - 1, kuang)
            return kuang

        _mid(_zici_wenben("层"))
        self.shuzi_ceng = _mid(self._shuzi(0, 99999, 54, "层：数大的盖在上面"))

        _mid(_zici_wenben("开始"))
        self.shuru_kaishi = _mid(self._shijian_shuru("这一条的开始时间"))

        _mid(_zici_wenben("结束"))
        self.shuru_jieshu = _mid(self._shijian_shuru("这一条的结束时间"))

        # 时长：能改 —— 改它就等于改结束时间（开始时间不动），照 AEG
        self.lian_shichang = _mid(
            self._shijian_shuru("这一条的时长（改了 = 结束时间跟着变）")
        )

        for jian, zi, tip in (
            ("b", "B", "粗体"),
            ("i", "I", "斜体"),
            ("u", "U", "下划线"),
            ("s", "S", "删除线"),
        ):
            an = QtWidgets.QPushButton(zi)
            an.setCheckable(True)
            an.setFixedSize(26, 24)
            an.setToolTip(f"{tip}：往这条字幕里加 / 去 \\{jian}1")
            an.setStyleSheet(self._an_yangshi())
            _bot(an)
            self._zi_anniu[jian] = an

        self.an_fn = QtWidgets.QPushButton("fn")
        self.an_fn.setFixedSize(34, 24)
        self.an_fn.setToolTip("换字体：往这条字幕里加 \\fn")
        self.an_fn.setStyleSheet(_ys_ci_anniu())
        _bot(self.an_fn)

        for wei, tip in (
            ("c1", "主要颜色"),
            ("c2", "次要颜色"),
            ("c3", "边框颜色"),
            ("c4", "阴影颜色"),
        ):
            an = QtWidgets.QPushButton()
            an.setFixedSize(28, 24)
            an.setToolTip(f"{tip}（点一下改）")
            an.setStyleSheet(_ys_ci_anniu())
            _bot(an)
            self._yanse_anniu[wei] = an

        self.an_xiayihang = QtWidgets.QPushButton("✓")
        self.an_xiayihang.setFixedSize(26, 24)
        self.an_xiayihang.setToolTip("这一条改完，跳到下一条（到底了就新建一条）")
        self.an_xiayihang.setStyleSheet(_ys_ci_anniu())
        _bot(self.an_xiayihang)

        # 时间 / 帧：切字幕列表和上面时间框按哪种看（照 AEG 的 Time / Frames）
        self._moshi_zu = QtWidgets.QButtonGroup(self)
        for zi, tishi, zhen in (
            ("时间", "开始/结束看时间", False),
            ("帧", "开始/结束看帧号", True),
        ):
            an = QtWidgets.QRadioButton(zi)
            an.setChecked(not zhen)
            an.setToolTip(tishi)
            an.setStyleSheet(_ys_xuanxiang())
            self._moshi_zu.addButton(an)
            _bot(an)
            if zhen:
                self.an_zhen = an
            else:
                self.an_shijian = an

        # 计数：[当前条/总条数/剩余条数]，贴在这一排最右（"时间 / 帧"右边那片空白）
        self.ji_shu = QtWidgets.QLabel("[0/0/0]")
        self.ji_shu.setObjectName("YsgHint")
        self.ji_shu.setToolTip("当前条 / 总条数 / 剩余条数")
        self.h3.addWidget(self.ji_shu)

        wai.addWidget(self.pai2)
        wai.addWidget(self.pai3)

        # ---- 往外发 ----
        self.gou_zhushi.toggled.connect(
            lambda kai: self._fa("zhushi", bool(kai))
        )
        self.gou_gen_tiao.toggled.connect(
            lambda kai: self._fa("gen_tiao", bool(kai))
        )
        self.gou_shishi_gun.toggled.connect(
            lambda kai: self._fa("shishi_gun", bool(kai))
        )
        self.gou_bofangtou_xuanzhong.toggled.connect(
            lambda kai: self._fa("bofangtou_xuanzhong", bool(kai))
        )
        self.xia_yang.currentTextChanged.connect(
            lambda ming: self._fa("yang", str(ming or "").strip())
        )
        self.xia_shuo.currentTextChanged.connect(
            lambda ming: self._fa("shuo", str(ming or "").strip())
        )
        self.an_yang.clicked.connect(lambda: self._fa("bianji", "yang"))
        self.an_xiayihang.clicked.connect(lambda: self._fa("xiayihang", True))
        self.shuzi_ceng.valueChanged.connect(lambda v: self._fa("ceng", int(v)))
        for jian, an in self._zi_anniu.items():
            an.toggled.connect(
                lambda kai, j=jian: self._fa(
                    "tag", (j, bool(kai), self._xuan_qu())
                )
            )
        self.an_fn.clicked.connect(self._xuan_ziti)
        for wei, an in self._yanse_anniu.items():
            an.clicked.connect(lambda _=False, w=wei: self._xuan_yanse(w))
        self.shuru_kaishi.editingFinished.connect(self._kaishi_wangou)
        self.shuru_jieshu.editingFinished.connect(self._jieshu_wangou)
        self.lian_shichang.editingFinished.connect(self._shichang_wangou)
        self.an_shijian.toggled.connect(
            lambda kai: kai and self._huan_moshi(False)
        )
        self.an_zhen.toggled.connect(
            lambda kai: kai and self._huan_moshi(True)
        )

    # ---- 小零件 ----
    def _shuzi(self, zui_xiao, zui_da, kuan, tishi):
        kuang = QtWidgets.QSpinBox()
        kuang.setRange(zui_xiao, zui_da)
        kuang.setMinimumWidth(kuan)
        kuang.setMaximumWidth(kuan + 26)
        kuang.setToolTip(tishi)
        kuang.setStyleSheet(_ys_shuzi())
        return kuang

    def _shijian_shuru(self, tishi):
        kuang = QtWidgets.QLineEdit()
        kuang.setMinimumWidth(78)
        kuang.setMaximumWidth(120)
        kuang.setAlignment(Qt.AlignCenter)     # 时间 / 帧号都居中，照 AEG 那样
        kuang.setToolTip(f"{tishi}（0:00:05.65 这种写法）")
        kuang.setStyleSheet(_ys_shuru())
        return kuang

    def _an_yangshi(self):
        c = _ys()
        return _ys_ci_anniu() + f"""
        QPushButton:checked {{
            background-color: {c['zhuse']};
            color: #ffffff;
            border-color: {c['zhuse']};
        }}
        """

    # ---- 宽了就并排（照 AEG 的 OnSize） ----
    def resizeEvent(self, shi):
        super().resizeEvent(shi)
        self._pai_yang()

    def _pai_yang(self):
        """第 3 排要不要并到第 2 排后面：够宽就并，不够就分两排"""
        if not self._bot:
            return
        if self._yao_kuan <= 0:
            # 两排各自排下来要多宽：量一次记着（并排以后就量不准了）
            self._yao_kuan = (
                self.h2.sizeHint().width()
                + self.h3.sizeHint().width()
                + 30
            )
        he = self.width() >= self._yao_kuan
        if he == self._bingpai:
            return
        self._bingpai = he
        if he:
            for kuang in list(self._bot):
                self.h2.insertWidget(self.h2.count() - 1, kuang)
            self.pai3.hide()
        else:
            for kuang in list(self._bot):
                self.h3.insertWidget(self.h3.count() - 1, kuang)
            self.pai3.show()

    def _fa(self, jian, zhi):
        if self._tian:
            return
        self.gongju_gaile.emit(jian, zhi)

    def eventFilter(self, duixiang, shijian):
        """样式下拉上滚轮：往上滚一格 = 上一个样式，往下滚 = 下一个

        换了之后走的是 currentTextChanged 那条老路，跟鼠标点选完全一样。
        事件吃掉，别让滚轮冒出去把外面的列表 / 时间轴滚走。
        """
        if duixiang is self.xia_yang and shijian.type() == QtCore.QEvent.Wheel:
            bu = shijian.angleDelta().y()
            ge = self.xia_yang.count()
            if bu and ge:
                xin = self.xia_yang.currentIndex() + (-1 if bu > 0 else 1)
                self.xia_yang.setCurrentIndex(max(0, min(ge - 1, xin)))
            return True
        return super().eventFilter(duixiang, shijian)

    def _kaishi_wangou(self):
        ms = self._du_shijian(self.shuru_kaishi)
        if ms is None:
            self._tian_shijian()
            return
        self._fa("qi", int(ms))

    def _jieshu_wangou(self):
        ms = self._du_shijian(self.shuru_jieshu)
        if ms is None:
            self._tian_shijian()
            return
        self._fa("zhi", int(ms))

    def _shichang_wangou(self):
        """改了时长：结束时间 = 开始时间 + 这个时长（开始时间不动，照 AEG）

        读不出来、或者算出来不比开始晚（时长不是正数），就把三个框还原。
        """
        chang = self._du_shijian(self.lian_shichang)
        if chang is None or chang <= 0:
            self._tian_shijian()
            return
        self._fa("zhi", int(self._qi_ms + chang))

    def _du_shijian(self, kuang):
        """读开始 / 结束框：看帧的时候框里是帧号，看时间的时候是时间。读不出来给 None"""
        t = str(kuang.text() or "").strip()
        if self._zhen:
            if not re.match(r"^\d+$", t):
                return None
            if kuang is self.lian_shichang:
                return _zhen_to_ms(int(t), self._fps)       # 时长：几帧
            # 开始框按"块头"认、结束框按"块尾"认（照 AEG，见 ms_tou_zhen）
            return _zhen_to_ms(
                int(t), self._fps, wei=(kuang is self.shuru_jieshu)
            )
        return _wenben_to_ms(t)

    def _huan_moshi(self, zhen):
        """切「时间 / 帧」：框里换成对应的写法，再告诉外面列表也跟着换"""
        zhen = bool(zhen)
        if zhen == self._zhen:
            return
        self._zhen = zhen
        self._tian_shijian()
        if self._tian:
            return
        self.gongju_gaile.emit("moshi", "zhen" if zhen else "shijian")

    def shezhi_jishu(self, xu, zong):
        """右下角那个 [当前/总数/剩余]

        xu 是当前选中的第几条（0 起算）—— 字幕列表选中的那条、或者时间轴上
        选中的那个块，单选才有；多选 / 没选中给 -1，这时候"当前"算 0。
        """
        try:
            zong = max(0, int(zong or 0))
        except (TypeError, ValueError):
            zong = 0
        try:
            xu = int(xu)
        except (TypeError, ValueError):
            xu = -1
        dang = xu + 1 if 0 <= xu < zong else 0
        self.ji_shu.setText(f"[{dang}/{zong}/{zong - dang}]")

    def shezhi_kaiguan(self, gen_tiao=True, shishi_gun=False,
                       bofangtou_xuanzhong=False):
        """外面读 ini 之后把三个开关的初始状态摆上（摆的时候不往外发信号）"""
        jiu = self._tian
        self._tian = True
        try:
            self.gou_gen_tiao.setChecked(bool(gen_tiao))
            self.gou_shishi_gun.setChecked(bool(shishi_gun))
            self.gou_bofangtou_xuanzhong.setChecked(
                bool(bofangtou_xuanzhong)
            )
        finally:
            self._tian = jiu

    def shezhi_moshi(self, zhen):
        """外面切了模式（列表那边）：radio 和时间框跟着换"""
        zhen = bool(zhen)
        if zhen != self._zhen:
            self._zhen = zhen
            if zhen:
                self.an_zhen.setChecked(True)
            else:
                self.an_shijian.setChecked(True)
        self._tian_shijian()

    def _tian_shijian(self):
        """把开始 / 结束 / 时长三个框按当前模式填上"""
        jiu = self._tian      # 外面可能正在填值，别给它提前解了
        self._tian = True
        try:
            if self._zhen:
                self.shuru_kaishi.setToolTip("这一条的开始帧号（直接写数字）")
                self.shuru_jieshu.setToolTip("这一条的结束帧号（直接写数字）")
                self.lian_shichang.setToolTip(
                    "这一条的时长，算帧数（改了 = 结束帧跟着变）"
                )
                self.shuru_kaishi.setText(
                    _zhen_wenben(self._qi_ms, self._fps, wei=False)
                )
                self.shuru_jieshu.setText(
                    _zhen_wenben(self._zhi_ms, self._fps, wei=True)
                )
                self.lian_shichang.setText(
                    str(
                        max(
                            0,
                            ms_tou_zhen(self._zhi_ms, self._fps, wei=True)
                            - ms_tou_zhen(self._qi_ms, self._fps, wei=False)
                            + 1,
                        )
                    )
                )
            else:
                self.shuru_kaishi.setToolTip(
                    "这一条的开始时间（0:00:05.65 这种写法）"
                )
                self.shuru_jieshu.setToolTip(
                    "这一条的结束时间（0:00:05.65 这种写法）"
                )
                self.lian_shichang.setToolTip(
                    "这一条的时长（改了 = 结束时间跟着变；0:00:02.34 这种写法）"
                )
                self.shuru_kaishi.setText(
                    _ass_shi_jian_wenben(self._qi_ms)
                )
                self.shuru_jieshu.setText(_ass_shi_jian_wenben(self._zhi_ms))
                self.lian_shichang.setText(
                    _ass_shi_jian_wenben(max(0, self._zhi_ms - self._qi_ms))
                )
        finally:
            self._tian = jiu

    def shezhi_fps(self, fps):
        """视频帧率给过来（按帧看时间要用）；不知道帧率就不让切到帧"""
        try:
            fps = float(fps or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        self._fps = max(0.0, fps)
        self.an_zhen.setEnabled(self._fps > 0)
        if self._fps <= 0 and self._zhen:
            self.an_shijian.setChecked(True)      # 会自己走 _huan_moshi
        elif self._zhen:
            self._tian_shijian()                  # 帧率换了，帧号得重算

    def shezhi_wenben_kuang(self, kuang):
        """外面把编辑框给进来：框里选中了一段字，按钮就只对那段动手（照 AEG）"""
        self._wenben_kuang = kuang

    def _xuan_qu(self):
        """编辑框里选中的那一段（起, 止）；没选东西给 None

        注意要在弹对话框之前问，弹完回来光标可能就没了。

        框里是带 \\N 标记的显示写法（比字幕文本多着真换行），位置得换算回字幕
        文本的坐标 —— 外面拿这个位置往字幕文本里插标签。
        """
        kuang = self._wenben_kuang
        if kuang is None:
            return None
        cur = kuang.textCursor()
        if not cur.hasSelection():
            return None
        qi, zhi = int(cur.selectionStart()), int(cur.selectionEnd())
        huan = getattr(kuang, "shuju_wei", None)
        if huan is not None:
            qi, zhi = huan(qi), huan(zhi)
        return (qi, zhi) if zhi > qi else None

    def _xuan_ziti(self):
        """fn 按钮：挑一个字体，往这条字幕里写 \\fn字体名"""
        if self._tian:
            return
        xuan = self._xuan_qu()
        zi, cheng = QtWidgets.QFontDialog.getFont(self._ziti, self, "选字体")
        if not cheng:
            return
        self._fa("tag", ("fn", zi.family(), xuan))

    def _xuan_yanse(self, wei):
        """四个颜色按钮：挑一个颜色，往这条字幕里写 \\c / \\2c / \\3c / \\4c"""
        if self._tian:
            return
        xuan = self._xuan_qu()
        qi = QtGui.QColor(str(self._yanse.get(wei) or "#FFFFFF"))
        yan = QtWidgets.QColorDialog.getColor(qi, self, "选颜色")
        if not yan.isValid():
            return
        self._fa("yanse", (wei, _yanse_to_ass(yan), xuan))

    # ---- 往控件里填值 / 换候选 ----
    def shezhi_yangshi_ming(self, ming_liebiao):
        """样式下拉的候选（打开字幕 / 样式改完时给一次）"""
        jiu = self.xia_yang.currentText()
        self._tian = True
        self.xia_yang.clear()
        self.xia_yang.addItems([str(x) for x in (ming_liebiao or [])])
        if jiu:
            self.xia_yang.setCurrentText(jiu)
        self._tian = False

    def shezhi_shuo_ming(self, ming_liebiao):
        """说话人下拉的候选"""
        jiu = self.xia_shuo.currentText()
        self._tian = True
        self.xia_shuo.clear()
        self.xia_shuo.addItems([""] + [str(x) for x in (ming_liebiao or [])])
        if jiu:
            self.xia_shuo.setCurrentText(jiu)
        self._tian = False

    def shezhi_gongju(self, fu, gs=None, wenben=None):
        """把当前这一条填进来；fu 空 = 没选中（整排灰掉）"""
        fu = dict(fu or {})
        gs = dict(gs or {})
        self._tian = True
        try:
            you = bool(fu)
            for kuang in (
                self.gou_zhushi, self.xia_yang, self.an_yang, self.xia_shuo,
                self.shuzi_ceng, self.shuru_kaishi, self.shuru_jieshu,
                self.lian_shichang, self.an_fn, self.an_xiayihang,
            ):
                kuang.setEnabled(you)
            for an in list(self._zi_anniu.values()) + list(
                self._yanse_anniu.values()
            ):
                an.setEnabled(you)
            if not you:
                self.lian_zishu.setText("0")
                self.lian_shichang.setText(_ass_shi_jian_wenben(0))
                return
            self._tian_ming(self.xia_yang, fu.get("yang"), "Default")
            self._tian_ming(self.xia_shuo, fu.get("shuo"), "")
            self.gou_zhushi.setChecked(bool(fu.get("zhushi")))
            self.lian_zishu.setText(str(_zuiduo_zishu(wenben)))
            self.shuzi_ceng.setValue(int(fu.get("ceng") or 0))
            self._qi_ms = int(fu.get("qi") or 0)
            self._zhi_ms = int(fu.get("zhi") or 0)
            self._tian_shijian()
            for jian, an in self._zi_anniu.items():
                an.setChecked(bool(gs.get(jian)))
            self._tian_ziti(gs)
            self._tian_yanse(gs)
        finally:
            self._tian = False

    def _tian_ming(self, kuang, ming, moren):
        """下拉里挑中这个名字；表里没有就先塞进去（免得显示成空白）"""
        ming = str(ming or "").strip() or str(moren or "")
        if ming and kuang.findText(ming) < 0:
            kuang.addItem(ming)
        kuang.setCurrentText(ming)

    def _tian_ziti(self, gs):
        zi = QtGui.QFont()
        ming = str(gs.get("font") or "").strip()
        if ming.startswith("@"):
            ming = ming[1:]
        if ming:
            zi.setFamily(ming)
        zi.setPixelSize(100)
        self._ziti = zi
        # AEG 的 fn 按钮就是两个字，不显示字体名（字体名放提示里）
        self.an_fn.setText("fn")
        self.an_fn.setToolTip(
            f"换字体（现在是 {ming}）：往这条字幕里加 \\fn" if ming
            else "换字体：往这条字幕里加 \\fn"
        )

    def _tian_yanse(self, gs):
        for wei, an in self._yanse_anniu.items():
            ming = str(gs.get(wei) or self._yanse.get(wei) or "#FFFFFF")
            if not re.match(r"^#[0-9A-Fa-f]{6}$", ming):
                ming = self._yanse.get(wei) or "#FFFFFF"
            self._yanse[wei] = ming
            an.setStyleSheet(
                _ys_ci_anniu()
                + f"\nQPushButton {{ background-color: {ming}; }}"
            )


def _yanse_qt(ming, tou=255):
    """'#RRGGBB' + 不透明度 -> QColor（认不出来给白色）"""
    yan = QtGui.QColor(str(ming or "#FFFFFF"))
    if not yan.isValid():
        yan = QtGui.QColor("#FFFFFF")
    try:
        a = int(round(float(tou)))
    except (TypeError, ValueError):
        a = 255
    yan.setAlpha(max(0, min(255, a)))
    return yan


# 预览框里的字至少画这么高（像素）：预览框是整幅画面的缩略，40 号字按缩略比例
# 只剩几像素高、看不清。只改这一行就能调预览里字的大小。
YULAN_ZI_GAO = 30.0

_SHENG_SHU_GONGJU = None


def _shu_gongju():
    """画面那边的竖排字工具（字体名带 @ 的那一支躺倒要用）

    画面在 video_work_dialog 里，这边用到时才去拿 —— 文件头上导会绕成环形导入。
    """
    global _SHENG_SHU_GONGJU
    if _SHENG_SHU_GONGJU is None:
        from anylabeling.views.labeling.widgets.video_work_dialog import (
            _shi_shu, _shu_kuan, _shu_lu,
        )
        _SHENG_SHU_GONGJU = (_shi_shu, _shu_kuan, _shu_lu)
    return _SHENG_SHU_GONGJU


class _YangshiYulan(QtWidgets.QWidget):
    """样式预览：按当前样式把那几个字画出来（描边、阴影、缩放、旋转都照做）

    预览框当成整幅画面的缩略：对齐 1~9 加左/右/垂直边距照画面的比例摆（改哪个
    数字这儿都跟着动），字按缩略比例太小看不清，所以单独放大到看得清为止。
    """

    def __init__(self, parent=None, ziti_gongchang=None, jizhun=None):
        super().__init__(parent)
        self._zi = {}
        self._wen = "字体测试内容"
        self._bei = QtGui.QColor("#8E8E8E")
        self._jizhun_kuan = self._kan_jizhun_kuan(jizhun)
        # 造字体的函数（外面把画面里那个给进来）：字号、粗斜体、字距都一样
        self._ziti_gongchang = ziti_gongchang
        self.setMinimumHeight(104)

    @staticmethod
    def _kan_jizhun_kuan(jizhun):
        """画面基准宽（PlayResX）：预览按它算水平方向的比例"""
        try:
            return max(1.0, float(int((jizhun or (1280, 720))[0])))
        except (TypeError, ValueError, IndexError):
            return 1280.0

    def shezhi_jizhun(self, jizhun):
        self._jizhun_kuan = self._kan_jizhun_kuan(jizhun)
        self.update()

    def shezhi(self, zi, wen=None):
        self._zi = dict(zi or {})
        if wen is not None:
            self._wen = str(wen)
        self.update()

    def _na_ziti(self, zi):
        """当前样式对应的字体：跟画面里同一个造法"""
        if callable(self._ziti_gongchang):
            try:
                ziti = self._ziti_gongchang(zi)
                if isinstance(ziti, QtGui.QFont):
                    return ziti
            except Exception:  # noqa
                pass
        # 外面没给（单独试这个控件的时候）：按老规矩自己拼一个
        ming = str(zi.get("font") or "").strip().lstrip("@")
        ziti = QtGui.QFont()
        if ming:
            ziti.setFamily(ming)
        ziti.setPixelSize(max(1, int(round(max(1.0, float(zi.get("fs") or 40))))))
        ziti.setBold(bool(zi.get("b")))
        ziti.setItalic(bool(zi.get("i")))
        ziti.setUnderline(bool(zi.get("u")))
        ziti.setStrikeOut(bool(zi.get("s")))
        return ziti

    def paintEvent(self, shi):
        hua = QtGui.QPainter(self)
        hua.setRenderHint(QtGui.QPainter.Antialiasing, True)
        hua.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        hua.fillRect(self.rect(), self._bei)
        zi = self._zi

        def _shu(jian, moren):
            try:
                return float(str(zi.get(jian)).strip())
            except (TypeError, ValueError):
                return moren

        # 照 Aegisub 的样式预览来（src/subs_preview.cpp 的 SetStyle：alignment 强制
        # 成 5、三个边距清零）—— 预览框自己就是一整幅画面，所以字永远落在正中；改
        # 对齐 / 边距它不动。要看落位看画面那边（那边是跟着对齐边距走的）。
        jx = max(1.0, self._jizhun_kuan)
        s = max(1, self.width()) / jx            # 基准像素 -> 预览像素
        jy = max(1, self.height()) / s           # 预览这块高合多少基准像素
        ziti = self._na_ziti(zi)
        fm = QtGui.QFontMetricsF(ziti)
        wen = self._wen
        ming_zi = str(zi.get("font") or "").strip()
        _shi_shu, _shu_kuan, _shu_lu = _shu_gongju()
        shu_pai = _shi_shu(ming_zi)     # 字体名带 @ = 竖排那一支，字要躺倒
        fscx = max(0.01, _shu("fscx", 100.0) / 100.0)
        ml = mr = mv = 0.0               # AEG 把三个边距清零
        an = 5                           # AEG 把对齐钉在正中
        lie = (an - 1) % 3               # 0 左 / 1 中 / 2 右
        pai = (an - 1) // 3              # 0 下 / 1 中 / 2 上
        # 按缩略比例算，40 号字只剩几个像素高、看不清 —— 所以字单独放大到看得清
        # 为止（摆位还是同一套基准坐标，一点不动，只是字画大些；描边、阴影跟着
        # 一起放大，粗细看着才跟画面上一样）。
        z = 1.0
        if fm.height() > 0.1:
            z = max(1.0, YULAN_ZI_GAO / (fm.height() * s))
        if lie == 0:
            kong = self.width() - ml * s
        elif lie == 1:
            kong = self.width() - (ml + mr) * s
        else:
            kong = self.width() - mr * s
        kong = max(1.0, kong)
        kuan_hua_zi = (
            _shu_kuan(ziti, ming_zi, wen) if shu_pai
            else fm.horizontalAdvance(wen)
        ) * fscx
        if kuan_hua_zi > 0.1:
            z = min(z, max(1.0, kong * 0.98 / (kuan_hua_zi * s)))
        if abs(z - 1.0) > 0.01:
            ziti = QtGui.QFont(ziti)
            px = ziti.pixelSize()
            if px > 0:
                ziti.setPixelSize(max(1, int(round(px * z))))
            else:
                pt = ziti.pointSizeF()
                if pt > 0:
                    ziti.setPointSizeF(pt * z)
            if ziti.letterSpacingType() == QtGui.QFont.AbsoluteSpacing:
                ziti.setLetterSpacing(
                    QtGui.QFont.AbsoluteSpacing, ziti.letterSpacing() * z
                )
            fm = QtGui.QFontMetricsF(ziti)
        kuan_zi = (
            _shu_kuan(ziti, ming_zi, wen) if shu_pai
            else fm.horizontalAdvance(wen)
        )
        if lie == 0:
            ax = ml
        elif lie == 1:
            # 居中：在左右边距之间居中，左/右边距各推一半 —— 跟画面（libass）同一
            # 套算法，不是死钉在正中。
            ax = (jx + ml - mr) / 2.0
        else:
            ax = jx - mr
        if pai == 0:
            ay = jy - mv
        elif pai == 1:
            ay = jy / 2.0
        else:
            ay = mv
        kuan_hua = kuan_zi * fscx
        if lie == 0:
            x0 = ax
        elif lie == 1:
            x0 = ax - kuan_hua / 2.0
        else:
            x0 = ax - kuan_hua
        zong_gao = fm.ascent() + fm.descent()
        if pai == 0:
            ding = ay - zong_gao
        elif pai == 1:
            ding = ay - zong_gao / 2.0
        else:
            ding = ay
        zhong_x = x0 + kuan_hua / 2.0
        di = ding + fm.ascent()

        if shu_pai:
            lu = _shu_lu(ziti, ming_zi, wen, 0.0, 0.0)   # 每个字躺倒，横着排
        else:
            lu = QtGui.QPainterPath()
            lu.addText(0.0, 0.0, ziti, wen)
        # 挪到正中；\fscx 绕着字块中点拉、\frz 绕着中点那条基线转 —— 跟画面里
        # 一模一样（画面那边也是这么画的）
        hua.save()
        hua.scale(s, s)
        hua.translate(zhong_x, di)
        if abs(fscx - 1.0) > 0.001:
            hua.scale(fscx, 1.0)
        if abs(_shu("frz", 0.0)) > 0.01:
            hua.rotate(_shu("frz", 0.0))
        hua.translate(-kuan_zi / 2.0, 0.0)

        shad = max(0.0, _shu("shad", 0.0)) * z
        if shad > 0.1:
            hua.save()
            hua.translate(shad, shad)
            hua.fillPath(lu, _yanse_qt(zi.get("c4"), zi.get("a4", 140)))
            hua.restore()
        bord = max(0.0, _shu("bord", 0.0)) * z
        if bord > 0.1:
            bi_bi = QtGui.QPen(_yanse_qt(zi.get("c3"), zi.get("a3", 255)))
            bi_bi.setWidthF(bord * 2.0)
            bi_bi.setJoinStyle(QtCore.Qt.RoundJoin)
            bi_bi.setCapStyle(QtCore.Qt.RoundCap)
            hua.strokePath(lu, bi_bi)
        hua.fillPath(lu, _yanse_qt(zi.get("c1"), zi.get("a1", 255)))
        hua.restore()


class _ZitiXia(QtWidgets.QComboBox):
    """样式编辑器里的字体下拉：点开那一刻才把系统里几百个字体塞进去

    几百项一次塞好，Qt 每次量窗口尺寸都要把它们从头上到尾遍历一遍，点一下
    「编辑」就得多等一百多毫秒。所以先只放当前这一支，真要挑字体、把下拉
    点开的那一刻再补全（补全只要几毫秒，感觉不出来）。
    """

    def __init__(self, ziti_men, xian, parent=None):
        super().__init__(parent)
        self._ziti_men = [str(x) for x in (ziti_men or [])]
        self._xian = str(xian or "")
        self._man = False
        self.addItem(self._xian)

    def showPopup(self):
        if not self._man:
            self._man = True
            jiu = self.currentText()
            self.blockSignals(True)
            self.clear()
            self.addItems(self._ziti_men)
            if jiu and jiu not in self._ziti_men:
                self.insertItem(0, jiu)
            self.setCurrentText(jiu)
            self.blockSignals(False)
        super().showPopup()

    def shezhi_dangqian(self, ming):
        """把当前这支字体整个换掉（新建 / 复制样式时用）"""
        ming = str(ming or "")
        self._xian = ming
        self._man = False       # 下回点开下拉重新铺一遍
        self.blockSignals(True)
        self.clear()
        self.addItem(ming)
        self.blockSignals(False)


class ShuohuaCaoweiDialog(QtWidgets.QDialog):
    """【设置说话人+样式】配置管理（照 AEG 那个 Lua 脚本的配置窗口抄）

    12 个槽位，每个 = 说话人（能打字的输入框）+ 样式（下拉，候选是这份字幕
    里真有的样式名）。点「确定」才存进配置文件，之后按 F1~F12 或在字幕列表
    右键就能把选中的字幕套上对应槽位。
    """

    baocun = pyqtSignal(object)     # 点「确定」：把这份槽位配置递给外面写文件

    def __init__(self, pei, yang_men=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("【设置说话人+样式GUI设置】/ 配置管理")
        self.setModal(False)        # 非模态：开着这个窗口照样能操作工作台
        # 能最小化，不挡路；关掉就销毁，下次打开重新读一遍配置
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._cao = [dict(x) for x in (pei or [])]
        while len(self._cao) < CAOWEI_SHU:
            self._cao.append({"shuohua": "", "yangshi": "Default"})

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(10, 10, 10, 10)
        wai.setSpacing(6)

        wang = QtWidgets.QGridLayout()
        wang.setHorizontalSpacing(8)
        wang.setVerticalSpacing(3)
        for lie, zi in enumerate(("槽位", "说话人", "样式")):
            biao = QtWidgets.QLabel(zi)
            biao.setStyleSheet(_ys_biao_ti())
            wang.addWidget(biao, 0, lie)
        wang.setColumnStretch(1, 3)
        wang.setColumnStretch(2, 3)

        ming = [str(x) for x in (yang_men or []) if str(x) != ""]
        if "Default" not in ming:
            ming.insert(0, "Default")

        self.shuru = []
        self.xia = []
        for i in range(CAOWEI_SHU):
            biao = QtWidgets.QLabel(f"槽位{i + 1:02d}")
            biao.setStyleSheet(_ys_biao_ti())
            wang.addWidget(biao, i + 1, 0)

            shu = QtWidgets.QLineEdit(str(self._cao[i].get("shuohua") or ""))
            shu.setStyleSheet(_ys_shuru())
            shu.setPlaceholderText("说话人名")
            wang.addWidget(shu, i + 1, 1)

            xia = QtWidgets.QComboBox()
            xia.setEditable(True)       # 样式表里没列出来的名字也能手打
            xia.addItems(ming)
            xian = str(self._cao[i].get("yangshi") or "Default")
            if xia.findText(xian) < 0:
                xia.addItem(xian)
            xia.setCurrentText(xian)
            xia.setMaxVisibleItems(20)
            wang.addWidget(xia, i + 1, 2)

            self.shuru.append(shu)
            self.xia.append(xia)
        wai.addLayout(wang, 1)

        tishi = QtWidgets.QLabel("修改后点确定保存，取消则不保存")
        tishi.setStyleSheet(_ys_biao_ti())
        wai.addWidget(tishi)

        an = QtWidgets.QHBoxLayout()
        an.addStretch(1)
        quxiao = QtWidgets.QPushButton("取消")
        quxiao.setStyleSheet(_ys_ci_anniu())
        queding = QtWidgets.QPushButton("确定")
        queding.setStyleSheet(_ys_zhu_anniu())
        for x in (quxiao, queding):
            x.setFixedHeight(26)
            x.setMinimumWidth(72)
            an.addWidget(x)
        wai.addLayout(an)

        quxiao.clicked.connect(self.reject)
        queding.clicked.connect(self._queding)

    def _queding(self):
        """非模态窗口没有 exec_ 的返回值：这份配置用信号递给外面去写文件"""
        self.baocun.emit(self.caowei())
        self.accept()

    def caowei(self):
        """现在这份槽位配置（外面拿去写文件）"""
        return [
            {
                "shuohua": self.shuru[i].text().strip(),
                "yangshi": self.xia[i].currentText().strip() or "Default",
            }
            for i in range(CAOWEI_SHU)
        ]


class KuohaoPaichuDialog(QtWidgets.QDialog):
    """【批量添加方括号】排除配置（照 AEG 那个 Lua 脚本的配置窗口抄）

    左列 = 排除关键词（说话人 / 样式里含它的不套），右列 = 排除符号（文本里
    已经有的不再套）。行数照原脚本：比现有条目多 5 行、至少 15 行，填满最后
    一行保存，下次打开就再多 5 行。
    """

    baocun = pyqtSignal(object)     # 点「确定」：把这份排除配置递给外面写文件

    def __init__(self, pei, parent=None):
        super().__init__(parent)
        self.setWindowTitle("【-批量添加方括号GUI】/ 配置")
        self.setModal(False)        # 非模态：开着这个窗口照样能操作工作台
        # 能最小化，不挡路；关掉就销毁，下次打开重新读一遍配置
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        ci = [str(x) for x in ((pei or {}).get("paichu_ci") or [])]
        fu = [str(x) for x in ((pei or {}).get("paichu_fuhao") or [])]

        hang = max(len(ci), len(fu)) + 5
        if hang < 15:
            hang = 15

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(10, 10, 10, 10)
        wai.setSpacing(6)

        shuo = QtWidgets.QLabel("排除配置（填满最后一行后保存，下次会自动增加更多行）")
        shuo.setStyleSheet(_ys_biao_ti())
        wai.addWidget(shuo)

        wang = QtWidgets.QGridLayout()
        wang.setHorizontalSpacing(6)
        wang.setVerticalSpacing(3)
        for lie, zi in enumerate(("排除关键词（说话人/样式）", "排除符号（文本内容）")):
            biao = QtWidgets.QLabel(zi)
            biao.setStyleSheet(_ys_biao_ti())
            wang.addWidget(biao, 0, lie * 2, 1, 2)
        wang.setColumnStretch(1, 2)
        wang.setColumnStretch(3, 2)

        self.shuru_ci = []
        self.shuru_fu = []
        for i in range(hang):
            wang.addWidget(QtWidgets.QLabel(f"{i + 1}:"), i + 1, 0)
            kuang = QtWidgets.QLineEdit(ci[i] if i < len(ci) else "")
            kuang.setStyleSheet(_ys_shuru())
            wang.addWidget(kuang, i + 1, 1)
            self.shuru_ci.append(kuang)

            wang.addWidget(QtWidgets.QLabel(f"{i + 1}:"), i + 1, 2)
            kuang = QtWidgets.QLineEdit(fu[i] if i < len(fu) else "")
            kuang.setStyleSheet(_ys_shuru())
            wang.addWidget(kuang, i + 1, 3)
            self.shuru_fu.append(kuang)
        wai.addLayout(wang, 1)

        tishi = QtWidgets.QLabel("提示：需要更多行？填满后保存，下次打开会自动增加5行")
        tishi.setStyleSheet(_ys_biao_ti())
        wai.addWidget(tishi)

        an = QtWidgets.QHBoxLayout()
        an.addStretch(1)
        quxiao = QtWidgets.QPushButton("取消")
        quxiao.setStyleSheet(_ys_ci_anniu())
        queding = QtWidgets.QPushButton("确定")
        queding.setStyleSheet(_ys_zhu_anniu())
        for x in (quxiao, queding):
            x.setFixedHeight(26)
            x.setMinimumWidth(72)
            an.addWidget(x)
        wai.addLayout(an)

        quxiao.clicked.connect(self.reject)
        queding.clicked.connect(self._queding)

    def _queding(self):
        """非模态窗口没有 exec_ 的返回值：这份配置用信号递给外面去写文件"""
        self.baocun.emit(self.paichu())
        self.accept()

    def paichu(self):
        """现在这份排除配置（外面拿去写文件）"""
        return {
            "paichu_ci": [
                k.text().strip() for k in self.shuru_ci if k.text().strip()
            ],
            "paichu_fuhao": [
                k.text().strip() for k in self.shuru_fu if k.text().strip()
            ],
        }


# 「新建样式」的默认字段（照 AEG 的 AssStyle 默认：Arial / 48 / 白字红次色 / 黑边黑影）
_YS_XIN_YANGSHI = {
    "name": "Default",
    "font": "Arial",
    "fs": 48.0,
    "c1": "#FFFFFF", "c2": "#FF0000", "c3": "#000000", "c4": "#000000",
    "a1": 255, "a2": 255, "a3": 255, "a4": 255,
    "b": False, "i": False, "u": False, "s": False,
    "fscx": 100.0, "fscy": 100.0, "fsp": 0.0, "frz": 0.0,
    "bord": 2.0, "shad": 2.0, "bs": 1, "an": 2,
    "ml": 0, "mr": 0, "mv": 0,
}


class YangshiBianjiDialog(QtWidgets.QDialog):
    """样式编辑器（照 AEG 的 DialogStyleEditor 抄）

    只摆我们画面里真认的东西：样式名 / 字体 / 字号 / 粗斜下删 / 四色 / 边距 /
    对齐 / 边框 / 阴影 / 缩放 / 旋转 / 间距。AEG 里的"编码"我们没有（字幕一律
    按 UTF-8 读写），"特效"那个字段我们的渲染不认，所以都不摆。

    改哪儿就发 gaile(整份样式字段)：外面拿去实时重画画面。点「确定 / 应用」
    再发 queren(整份样式字段)，外面这才写回 ASS。
    """

    gaile = pyqtSignal(object)
    queren = pyqtSignal(object)

    def __init__(self, ziduan, parent=None, ziti_men=None,
                 jizhun=None, ziti_gongchang=None, yang_men=None,
                 ming_jian_cha=None, kai_ku=None):
        super().__init__(parent)
        self.setWindowTitle("样式编辑器")
        self.setModal(False)        # 非模态：开着这个窗口照样能操作工作台
        # 能最小化，不挡路；关掉就销毁
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumWidth(600)
        self._ziduan = dict(ziduan or {})
        # 打开编辑器时那条样式的名字：新建 / 复制出来的新名字就挂在它后面
        self._yuan_ming = str(self._ziduan.get("name") or "").strip() or "Default"
        # 样式名候选（给下面「自动化脚本」里的槽位配置用）
        self._yang_men = [str(x) for x in (yang_men or []) if str(x) != ""]
        # 名字检查：外面拿它拦「撞名 / 名字空」（返回 False 就不关窗口）。
        # 不传就是以前的规矩（外面自己收下再判）。
        self._ming_jian_cha = ming_jian_cha
        self._tian = False          # 正往控件里塞值：这会儿的变更信号不算用户改
        self._yanse = {}            # 四个色块现在的颜色（含不透明度）
        # 点过「新建样式 / 复制样式」：这一趟就是来加一条新的，不是改原来那条。
        # 外面靠它决定「要不要问用户把用旧名字的行一起改名」—— 新建就别问了。
        self._quan_xin = False
        self._sheng_cheng = set()   # 这个窗口里刚生成过的名字，免得连点两次重名

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(10, 10, 10, 10)
        wai.setSpacing(8)

        zhu = QtWidgets.QHBoxLayout()
        zhu.setSpacing(8)
        zuo = QtWidgets.QVBoxLayout()
        zuo.setSpacing(8)
        you = QtWidgets.QVBoxLayout()
        you.setSpacing(8)
        zhu.addLayout(zuo, 3)
        zhu.addLayout(you, 2)
        wai.addLayout(zhu, 1)

        # ---- 样式名 ----
        he = QtWidgets.QGroupBox("样式名")
        g = QtWidgets.QHBoxLayout(he)
        self.shuru_ming = QtWidgets.QLineEdit(str(self._ziduan.get("name") or ""))
        self.shuru_ming.setStyleSheet(_ys_shuru())
        self.shuru_ming.setToolTip("样式名（ASS 里 [V4+ Styles] 那一列的 Name）")
        g.addWidget(self.shuru_ming)
        zuo.addWidget(he)

        # ---- 字体 ----
        he = QtWidgets.QGroupBox("字体")
        g = QtWidgets.QVBoxLayout(he)
        h1 = QtWidgets.QHBoxLayout()
        xian = str(self._ziduan.get("font") or "").strip()
        ming_men = [str(x) for x in (ziti_men or [])]
        if not ming_men:
            try:
                ming_men = sorted(QtGui.QFontDatabase().families())
            except Exception:  # noqa
                ming_men = []
        if xian and xian not in ming_men:
            ming_men.insert(0, xian)        # 当前用的这支没在列表里也别丢了
        self.xia_ziti = _ZitiXia(ming_men, xian)
        self.xia_ziti.setEditable(True)     # 系统里没列出来的名字也能手打
        self.xia_ziti.setMaxVisibleItems(22)
        self.xia_ziti.setToolTip(
            "字体名（点右边箭头从系统装的字体里挑，也能手打）。"
            "前面带 @ 的是竖排，跟 AEG 一样"
        )
        h1.addWidget(self.xia_ziti, 1)
        self.shuzi_zihao = QtWidgets.QDoubleSpinBox()
        self.shuzi_zihao.setRange(0.0, 10000.0)
        self.shuzi_zihao.setDecimals(1)
        self.shuzi_zihao.setValue(float(self._ziduan.get("fs") or 40))
        self.shuzi_zihao.setFixedWidth(92)
        self.shuzi_zihao.setToolTip("字号（就是 ASS 里的 Fontsize）")
        h1.addWidget(self.shuzi_zihao)
        g.addLayout(h1)
        h2 = QtWidgets.QHBoxLayout()
        self.gou_zi = {}
        for jian, zi in (("b", "粗体"), ("i", "斜体"),
                        ("u", "下划线"), ("s", "删除线")):
            gou = QtWidgets.QCheckBox(zi)
            gou.setChecked(bool(self._ziduan.get(jian)))
            gou.setStyleSheet(_ys_xuanxiang())
            h2.addWidget(gou)
            self.gou_zi[jian] = gou
        h2.addStretch(1)
        g.addLayout(h2)
        zuo.addWidget(he)

        # ---- 颜色 ----
        he = QtWidgets.QGroupBox("颜色")
        g = QtWidgets.QHBoxLayout(he)
        self.an_yanse = {}
        for wei, zi in (("c1", "主要颜色"), ("c2", "次要颜色"),
                        ("c3", "边框"), ("c4", "阴影")):
            lie = QtWidgets.QVBoxLayout()
            biao = QtWidgets.QLabel(zi)
            biao.setAlignment(QtCore.Qt.AlignCenter)
            lie.addWidget(biao)
            an = QtWidgets.QPushButton()
            an.setFixedSize(66, 20)
            an.setToolTip(f"{zi}（点一下改，能调透明度）")
            an.setStyleSheet(_ys_ci_anniu())
            lie.addWidget(an)
            g.addLayout(lie)
            self.an_yanse[wei] = an
        g.addStretch(1)
        zuo.addWidget(he)

        # ---- 边距 / 对齐 ----
        he = QtWidgets.QGroupBox("边距 / 对齐")
        g = QtWidgets.QHBoxLayout(he)
        bian = QtWidgets.QHBoxLayout()
        self.shuzi_bian = {}
        for jian, zi in (("ml", "左"), ("mr", "右"), ("mv", "垂直")):
            lie = QtWidgets.QVBoxLayout()
            biao = QtWidgets.QLabel(zi)
            biao.setAlignment(QtCore.Qt.AlignCenter)
            lie.addWidget(biao)
            kuang = QtWidgets.QSpinBox()
            kuang.setRange(-9999, 99999)
            kuang.setValue(int(self._ziduan.get(jian) or 0))
            kuang.setFixedWidth(66)
            kuang.setToolTip(f"{zi}边距（像素）")
            lie.addWidget(kuang)
            bian.addLayout(lie)
            self.shuzi_bian[jian] = kuang
        g.addLayout(bian, 1)
        self.qian_duiqi = {}
        wang = QtWidgets.QGridLayout()
        for i, zhi in enumerate((7, 8, 9, 4, 5, 6, 1, 2, 3)):
            qian = QtWidgets.QRadioButton(str(zhi))
            qian.setToolTip(f"对齐 {zhi}（数字键盘那种排法）")
            qian.setStyleSheet(_ys_xuanxiang())
            wang.addWidget(qian, i // 3, i % 3)
            self.qian_duiqi[zhi] = qian
        g.addLayout(wang)
        zuo.addWidget(he)

        # ---- 自动化脚本 ----
        # 照 AEG 那两个 Lua 脚本：这里只放"配置"入口，改完存进配置文件；
        # 套用（批量加「」/ F1~F12 套槽位）放字幕列表右键菜单里。
        he = QtWidgets.QGroupBox("自动化脚本")
        g = QtWidgets.QVBoxLayout(he)
        g.setSpacing(6)
        self.an_caowei_peizhi = QtWidgets.QPushButton(
            "【设置说话人+样式】配置管理"
        )
        self.an_caowei_peizhi.setToolTip(
            "配 12 个槽位的说话人 / 样式。配完在字幕列表右键套用，或按 F1~F12\n"
            f"存在：{buju_peizhi_lu()}"
        )
        self.an_kuohao_peizhi = QtWidgets.QPushButton(
            "【-批量添加方括号】排除配置"
        )
        self.an_kuohao_peizhi.setToolTip(
            "配「批量加「」」要跳过的：说话人 / 样式里含这些词的、文本里已经有这些符号的\n"
            f"存在：{buju_peizhi_lu()}"
        )
        for x in (self.an_caowei_peizhi, self.an_kuohao_peizhi):
            x.setStyleSheet(_ys_ci_anniu())
            x.setFixedHeight(26)
            g.addWidget(x)
        zuo.addWidget(he)
        zuo.addStretch(1)

        # ---- 边框 ----
        he = QtWidgets.QGroupBox("边框")
        g = QtWidgets.QGridLayout(he)
        g.addWidget(QtWidgets.QLabel("边框宽度"), 0, 0)
        self.shuzi_bord = QtWidgets.QDoubleSpinBox()
        self.shuzi_bord.setRange(0.0, 1000.0)
        self.shuzi_bord.setDecimals(1)
        self.shuzi_bord.setValue(float(self._ziduan.get("bord") or 0))
        self.shuzi_bord.setToolTip("描边宽度（像素）")
        g.addWidget(self.shuzi_bord, 0, 1)
        g.addWidget(QtWidgets.QLabel("阴影"), 1, 0)
        self.shuzi_shad = QtWidgets.QDoubleSpinBox()
        self.shuzi_shad.setRange(0.0, 1000.0)
        self.shuzi_shad.setDecimals(1)
        self.shuzi_shad.setValue(float(self._ziduan.get("shad") or 0))
        self.shuzi_shad.setToolTip("阴影距离（像素）")
        g.addWidget(self.shuzi_shad, 1, 1)
        g.addWidget(QtWidgets.QLabel("边框样式"), 2, 0)
        self.xia_kuangshi = QtWidgets.QComboBox()
        self.xia_kuangshi.addItems(["轮廓", "不透明背景"])
        self.xia_kuangshi.setCurrentIndex(
            1 if int(self._ziduan.get("bs") or 1) == 3 else 0
        )
        self.xia_kuangshi.setToolTip(
            "轮廓 = 字外面描一圈；不透明背景 = 字后面垫一块实心色"
        )
        g.addWidget(self.xia_kuangshi, 2, 1)
        you.addWidget(he)

        # ---- 大小 ----
        he = QtWidgets.QGroupBox("大小")
        g = QtWidgets.QGridLayout(he)
        self.shuzi_fscx = self._shuzi_biao(g, 0, "水平缩放 %", "左右拉宽 / 压窄（%）", 100.0)
        self.shuzi_fscy = self._shuzi_biao(g, 1, "垂直缩放 %", "上下拉高 / 压扁（%）", 100.0)
        self.shuzi_frz = self._shuzi_biao(g, 2, "旋转", "整条字转多少度", 0.0, -360.0, 360.0)
        self.shuzi_fsp = self._shuzi_biao(g, 3, "间距", "字和字之间多留多少像素", 0.0, -100.0, 1000.0)
        you.addWidget(he)

        # ---- 预览 ----
        he = QtWidgets.QGroupBox("预览")
        g = QtWidgets.QVBoxLayout(he)
        self.yulan = _YangshiYulan(
            ziti_gongchang=ziti_gongchang, jizhun=jizhun
        )
        g.addWidget(self.yulan, 1)
        self.shuru_ceshi = QtWidgets.QLineEdit("字体测试内容")
        self.shuru_ceshi.setStyleSheet(_ys_shuru())
        self.shuru_ceshi.setToolTip("预览用的字")
        g.addWidget(self.shuru_ceshi)
        you.addWidget(he, 1)

        # ---- 底下那排按钮 ----
        # 左边两个是样式表上的动作（照 AEG 样式管理器的 New / Copy）：只是把控件
        # 换成新的一份，真写进字幕还是得点「确定 / 应用」。
        an = QtWidgets.QHBoxLayout()
        self.an_xin = QtWidgets.QPushButton("新建样式")
        self.an_xin.setStyleSheet(_ys_ci_anniu())
        self.an_xin.setToolTip("照 AEG 的 New：一份全新默认样式（Arial / 48 / 白字）")
        self.an_fuzhi = QtWidgets.QPushButton("复制样式")
        self.an_fuzhi.setStyleSheet(_ys_ci_anniu())
        self.an_fuzhi.setToolTip(
            "照 AEG 的 Copy：照现在这份复制一份，名字自动加「 - Copy」\n"
            "（重名就往后排序号）"
        )
        for x in (self.an_xin, self.an_fuzhi):
            x.setFixedHeight(26)
            x.setMinimumWidth(72)
            an.addWidget(x)
        # 「样式库」就摆这儿（原来在编辑区工具栏上，挪进来了）：
        # 点开样式管理器 —— 左边全局样式库、右边这份字幕的样式，两边互相搬。
        # 没给 kai_ku 的地方（比如从样式管理器里点开编辑器）就把这按钮收掉。
        self.an_yangku = QtWidgets.QPushButton("样式库")
        self.an_yangku.setStyleSheet(_ys_ci_anniu())
        self.an_yangku.setToolTip(
            "打开样式管理器：\n"
            "左边是全局样式库（换哪份字幕都能用），右边是这份字幕自己的样式；\n"
            "两边能互相搬（复制到当前字幕 / 存进样式库），也能新建 / 编辑 / 复制 / 删除"
        )
        self.an_yangku.setFixedHeight(26)
        self.an_yangku.setMinimumWidth(72)
        self.an_yangku.setVisible(kai_ku is not None)
        if kai_ku is not None:
            self.an_yangku.clicked.connect(lambda _c=False: kai_ku())
        an.addWidget(self.an_yangku)
        an.addStretch(1)
        self.an_quxiao = QtWidgets.QPushButton("取消")
        self.an_quxiao.setStyleSheet(_ys_ci_anniu())
        self.an_yingyong = QtWidgets.QPushButton("应用")
        self.an_yingyong.setStyleSheet(_ys_ci_anniu())
        self.an_queding = QtWidgets.QPushButton("确定")
        self.an_queding.setStyleSheet(_ys_zhu_anniu())
        for x in (self.an_quxiao, self.an_yingyong, self.an_queding):
            x.setFixedHeight(26)
            x.setMinimumWidth(72)
            an.addWidget(x)
        wai.addLayout(an)

        # ---- 填值 + 接线 ----
        self._tian_kongjian()
        self._jie_xinhao()
        self.an_quxiao.clicked.connect(self.reject)
        self.an_yingyong.clicked.connect(self._yingyong)
        self.an_queding.clicked.connect(self._queding)
        self.an_xin.clicked.connect(self._xin_jian_yangshi)
        self.an_fuzhi.clicked.connect(self._fuzhi_yangshi)

    def _shuzi_biao(self, wang, hang, zi, tishi, moren,
                    zui_xiao=0.0, zui_da=10000.0):
        """一行「标签 + 数字框」，返回那个数字框"""
        wang.addWidget(QtWidgets.QLabel(zi), hang, 0)
        kuang = QtWidgets.QDoubleSpinBox()
        kuang.setRange(zui_xiao, zui_da)
        kuang.setDecimals(2)
        kuang.setValue(moren)
        kuang.setToolTip(tishi)
        wang.addWidget(kuang, hang, 1)
        return kuang

    def _tian_kongjian(self):
        """把样式字段填进各个控件（填的时候控件发的信号不算用户改）"""
        self._tian = True
        try:
            for wei in ("c1", "c2", "c3", "c4"):
                ming = str(self._ziduan.get(wei) or "#FFFFFF")
                if not re.match(r"^#[0-9A-Fa-f]{6}$", ming):
                    ming = "#FFFFFF"
                try:
                    tou = max(0, min(255, int(self._ziduan.get("a" + wei[1]) or 255)))
                except (TypeError, ValueError):
                    tou = 255
                self._yanse[wei] = (ming, tou)
            self._hua_yanse()
            an = int(self._ziduan.get("an") or 2)
            (self.qian_duiqi.get(an) or self.qian_duiqi[2]).setChecked(True)
            self.yulan.shezhi(self._shou_ji(), self.shuru_ceshi.text())
        finally:
            self._tian = False

    def _hua_yanse(self):
        for wei, an in self.an_yanse.items():
            ming, tou = self._yanse.get(wei, ("#FFFFFF", 255))
            an.setStyleSheet(
                _ys_ci_anniu()
                + f"\nQPushButton {{ background-color: {ming}; }}"
            )
            an.setText("" if tou >= 250 else f"{int(tou * 100 / 255)}%")
            an.setToolTip(
                f"现在：{ming}，不透明度 {int(tou * 100 / 255)}%（点一下改）"
            )

    def _jie_xinhao(self):
        for kuang in (
            self.shuru_ming, self.xia_ziti, self.shuzi_zihao,
            self.shuzi_bord, self.shuzi_shad, self.shuzi_fscx,
            self.shuzi_fscy, self.shuzi_frz, self.shuzi_fsp,
            self.shuru_ceshi,
        ):
            if isinstance(kuang, QtWidgets.QLineEdit):
                kuang.textChanged.connect(self._bian_le)
            elif isinstance(kuang, QtWidgets.QComboBox):
                kuang.currentTextChanged.connect(self._bian_le)
                kuang.currentIndexChanged.connect(self._bian_le)
            else:
                kuang.valueChanged.connect(self._bian_le)
        for kuang in self.shuzi_bian.values():
            kuang.valueChanged.connect(self._bian_le)
        for kuang in self.gou_zi.values():
            kuang.toggled.connect(self._bian_le)
        for kuang in self.qian_duiqi.values():
            kuang.toggled.connect(self._bian_le)
        for wei, kuang in self.an_yanse.items():
            kuang.clicked.connect(lambda _=False, w=wei: self._xuan_yanse(w))
        self.an_caowei_peizhi.clicked.connect(self._kai_caowei_peizhi)
        self.an_kuohao_peizhi.clicked.connect(self._kai_kuohao_peizhi)

    def _kai_caowei_peizhi(self):
        """打开「说话人 + 样式」的槽位配置（12 个槽位）

        非模态：开着这个窗口照样能操作工作台（选中字幕、播放、改别的都行）。
        点「确定」才由 baocun 信号把配置交回来写文件。
        """
        jiu = getattr(self, "_caowei_chuang", None)
        if jiu is not None:
            jiu.showNormal()
            jiu.raise_()
            jiu.activateWindow()
            return
        pei = du_zidonghua_peizhi()
        dlg = ShuohuaCaoweiDialog(pei.get("caowei"), self._yang_men, self)
        self._caowei_chuang = dlg
        dlg.baocun.connect(self._cun_caowei_peizhi)
        dlg.finished.connect(
            lambda _=0: setattr(self, "_caowei_chuang", None)
        )
        dlg.show()

    def _cun_caowei_peizhi(self, cao):
        """槽位配置窗口点了「确定」：写进配置文件"""
        pei = du_zidonghua_peizhi()
        pei["caowei"] = cao
        if not xie_zidonghua_peizhi(pei):
            QtWidgets.QMessageBox.warning(
                self, "配置写不进去",
                f"写不了这个文件：\n{buju_peizhi_lu()}\n"
                "看看软件根目录能不能写。",
            )

    def _kai_kuohao_peizhi(self):
        """打开「批量加「」」的排除配置

        非模态：点「确定」才由 baocun 信号把配置交回来写文件。
        """
        jiu = getattr(self, "_kuohao_chuang", None)
        if jiu is not None:
            jiu.showNormal()
            jiu.raise_()
            jiu.activateWindow()
            return
        pei = du_zidonghua_peizhi()
        dlg = KuohaoPaichuDialog(pei, self)
        self._kuohao_chuang = dlg
        dlg.baocun.connect(self._cun_kuohao_peizhi)
        dlg.finished.connect(
            lambda _=0: setattr(self, "_kuohao_chuang", None)
        )
        dlg.show()

    def _cun_kuohao_peizhi(self, paichu):
        """排除配置窗口点了「确定」：写进配置文件"""
        pei = du_zidonghua_peizhi()
        pei.update(paichu)
        if not xie_zidonghua_peizhi(pei):
            QtWidgets.QMessageBox.warning(
                self, "配置写不进去",
                f"写不了这个文件：\n{buju_peizhi_lu()}\n"
                "看看软件根目录能不能写。",
            )

    def _xuan_yanse(self, wei):
        """色块：开取色器（带透明度那一档），改完实时重画"""
        if self._tian:
            return
        ming, tou = self._yanse.get(wei, ("#FFFFFF", 255))
        qi = QtGui.QColor(ming)
        qi.setAlpha(tou)
        yan = QtWidgets.QColorDialog.getColor(
            qi, self, "选颜色", QtWidgets.QColorDialog.ShowAlphaChannel
        )
        if not yan.isValid():
            return
        self._yanse[wei] = (yan.name().upper(), yan.alpha())
        self._hua_yanse()
        self._bian_le()

    def _shou_ji(self):
        """把现在各控件的值收成一份样式字段（跟 ASS 的样式行一一对应）"""
        zi = dict(self._ziduan)
        zi["name"] = self.shuru_ming.text().strip() or "Default"
        zi["font"] = self.xia_ziti.currentText().strip()
        zi["fs"] = max(1.0, float(self.shuzi_zihao.value()))
        for jian, gou in self.gou_zi.items():
            zi[jian] = bool(gou.isChecked())
        for wei in ("c1", "c2", "c3", "c4"):
            ming, tou = self._yanse.get(wei, ("#FFFFFF", 255))
            zi[wei] = ming
            zi["a" + wei[1]] = int(tou)
        for jian, kuang in self.shuzi_bian.items():
            zi[jian] = int(kuang.value())
        for zhi, qian in self.qian_duiqi.items():
            if qian.isChecked():
                zi["an"] = int(zhi)
                break
        zi["bord"] = max(0.0, float(self.shuzi_bord.value()))
        zi["shad"] = max(0.0, float(self.shuzi_shad.value()))
        zi["bs"] = 3 if self.xia_kuangshi.currentIndex() == 1 else 1
        zi["fscx"] = max(0.01, float(self.shuzi_fscx.value()))
        zi["fscy"] = max(0.01, float(self.shuzi_fscy.value()))
        zi["frz"] = float(self.shuzi_frz.value())
        zi["fsp"] = float(self.shuzi_fsp.value())
        return zi

    def _bian_le(self, *_a):
        if self._tian:
            return
        zi = self._shou_ji()
        self.yulan.shezhi(zi, self.shuru_ceshi.text())
        self.gaile.emit(zi)

    def _yingyong(self):
        """「应用」：把现在这份交出去；外面不认这个名字就不交、返回 False"""
        zi = self._shou_ji()
        if self._ming_jian_cha is not None and not self._ming_jian_cha(zi):
            return False
        self.queren.emit(zi)
        return True

    def _queding(self):
        if self._yingyong():
            self.accept()

    def ziduan(self):
        """现在这份字段（外面点完确定要拿它去改样式表）"""
        return self._shou_ji()

    def _ming_zhan_le(self, ming):
        """这个名字是不是已经被占了

        算三种：候选名单里的、名字框里现在这个、这个窗口里刚生成过的。
        """
        ming = str(ming or "").strip().lower()
        if not ming:
            return True
        if any(str(x).strip().lower() == ming for x in self._yang_men):
            return True
        if ming in self._sheng_cheng:
            return True
        kuang = getattr(self, "shuru_ming", None)
        return ming == str(kuang.text() if kuang is not None else "").strip().lower()

    def _wei_yi_ming(self, qi, hou="复制"):
        """照 AEG 的 unique_name，再在名字尾巴上写清楚这条是干什么来的

        「新建样式」叫「源样式名 - 新建」，「复制样式」叫「源样式名 - 复制」；
        还跟别的样式撞就往后面排序号。
        """
        qi = str(qi or "").strip() or "Default"
        ming = f"{qi} - {hou}"
        shu = 2
        while self._ming_zhan_le(ming):
            ming = f"{qi} - {hou} ({shu})"
            shu += 1
        self._sheng_cheng.add(ming.strip().lower())
        return ming

    def _tian_quanbu(self, ziduan):
        """把一整份样式字段重新填回各个控件（新建 / 复制样式时用）"""
        self._ziduan = dict(ziduan or {})
        self._tian = True
        try:
            self.shuru_ming.setText(str(self._ziduan.get("name") or ""))
            self.xia_ziti.shezhi_dangqian(str(self._ziduan.get("font") or ""))
            self.shuzi_zihao.setValue(float(self._ziduan.get("fs") or 40))
            for jian, gou in self.gou_zi.items():
                gou.setChecked(bool(self._ziduan.get(jian)))
            for jian, kuang in self.shuzi_bian.items():
                kuang.setValue(int(self._ziduan.get(jian) or 0))
            self.shuzi_bord.setValue(float(self._ziduan.get("bord") or 0))
            self.shuzi_shad.setValue(float(self._ziduan.get("shad") or 0))
            self.xia_kuangshi.setCurrentIndex(
                1 if int(self._ziduan.get("bs") or 1) == 3 else 0
            )
            self.shuzi_fscx.setValue(float(self._ziduan.get("fscx") or 100))
            self.shuzi_fscy.setValue(float(self._ziduan.get("fscy") or 100))
            self.shuzi_frz.setValue(float(self._ziduan.get("frz") or 0))
            self.shuzi_fsp.setValue(float(self._ziduan.get("fsp") or 0))
        finally:
            self._tian = False
        self._tian_kongjian()       # 颜色 / 对齐 / 预览（它自己管 _tian）
        self._bian_le()

    def _xin_jian_yangshi(self):
        """「新建样式」：控件换成一份干净默认（照 AEG 的 New）"""
        zi = dict(_YS_XIN_YANGSHI)
        zi["name"] = self._wei_yi_ming(self._yuan_ming, "新建")
        self._quan_xin = True       # 这是新建，不是改原来那条
        self._tian_quanbu(zi)

    def _fuzhi_yangshi(self):
        """「复制样式」：照现在这份复制一份（照 AEG 的 Copy）"""
        zi = self._shou_ji()
        zi["name"] = self._wei_yi_ming(self._yuan_ming, "复制")
        self._quan_xin = True       # 这是新增一条，不是改原来那条
        self._tian_quanbu(zi)

    def quan_xin(self):
        """这一趟是不是「新建 / 复制」出来的新样式（外面拿它跳过改名询问）"""
        return self._quan_xin

    def yuan_ming(self):
        """这条样式现在的名字：外面拿它判断这一次是「改」还是「新建」"""
        return self._yuan_ming

    def she_zhi_le(self, ming):
        """外面已经写进去了：这条以后就叫 ming，再来一趟就是改它"""
        ming = str(ming or "").strip()
        if ming:
            self._yuan_ming = ming
            self._ziduan["name"] = ming
        self._quan_xin = False


class YangshiGuanliDialog(QtWidgets.QDialog):
    """样式管理器（照 AEG 的 DialogStyleManager 抄）

    左：全局样式库 —— 一个库一个 .sty 文件（跟 gongzuotai.ini 摆在一块），
        换哪份字幕都能拿来用。
    右：当前这份字幕自己的样式（就是 ASS 里 [V4+ Styles] 那几张）。
    两边都能 新建 / 编辑 / 复制 / 删除、上下移 / 置顶置底 / 排序；中间两个
    按钮互相搬：库 -> 字幕、字幕 -> 库（同名会问一句要不要覆盖，照 AEG）。

    窗口自己不管字幕长什么样：字幕那份样式怎么读怎么写全靠「接口」那几招
    （见下面用到的 ym_jiao_* ），库这边它自己读写 .sty 文件。
    """

    gaile = pyqtSignal()        # 字幕那边的样式表动过了（外面刷新下拉 + 重画）

    def __init__(self, jiekou, du_hang, zuo_hang, parent=None,
                 ziti_men=None, jizhun=None, ziti_gongchang=None):
        super().__init__(parent)
        self.setWindowTitle("样式管理器")
        self.setModal(False)        # 非模态：开着照样能操作工作台
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumSize(780, 440)

        self._jiekou = jiekou                    # 字幕那边的口子
        self._du_hang = du_hang                  # 若干行原文 -> [(名字, 字段)]
        self._zuo_hang = zuo_hang                # (名字, 字段) -> 一行原文
        self._ziti_men = ziti_men
        self._jizhun = jizhun
        self._ziti_gongchang = ziti_gongchang

        self._ku = ""                            # 现在在哪个库
        self._ku_men = []                        # 这个库里的 [(名字, 字段)]
        self._bianji_chuang = None                # 正开着的样式编辑器

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(10, 10, 10, 10)
        wai.setSpacing(8)

        # ---- 上面：换库 / 新建库 / 删库（照 AEG 的 Catalog 那一排）----
        he = QtWidgets.QGroupBox("样式库目录")
        g = QtWidgets.QHBoxLayout(he)
        self.xia_ku = QtWidgets.QComboBox()
        self.xia_ku.setMinimumWidth(200)
        self.xia_ku.setToolTip("换一个样式库（一个库就是一个 .sty 文件）")
        g.addWidget(self.xia_ku, 1)
        self.an_ku_xin = QtWidgets.QPushButton("新建库")
        self.an_ku_xin.setStyleSheet(_ys_ci_anniu())
        self.an_ku_xin.setFixedHeight(24)
        g.addWidget(self.an_ku_xin)
        self.an_ku_shan = QtWidgets.QPushButton("删除库")
        self.an_ku_shan.setStyleSheet(_ys_ci_anniu())
        self.an_ku_shan.setFixedHeight(24)
        g.addWidget(self.an_ku_shan)
        wai.addWidget(he)

        # ---- 主体：左边库、右边当前字幕（两边摆法一模一样）----
        zhu = QtWidgets.QHBoxLayout()
        zhu.setSpacing(8)
        ku_kuang, self.lib_ku, self.an_ku, self.an_ku_dong = self._zuo_yi_bian(
            "样式库", "复制到当前字幕 →"
        )
        jiao_kuang, self.lib_jiao, self.an_jiao, self.an_jiao_dong = (
            self._zuo_yi_bian("当前字幕", "← 存进样式库")
        )
        zhu.addWidget(ku_kuang, 1)
        zhu.addWidget(jiao_kuang, 1)
        wai.addLayout(zhu, 1)

        # ---- 底下：提示行 + 关闭 ----
        di = QtWidgets.QHBoxLayout()
        di.setSpacing(8)
        self.lian_ti_shi = QtWidgets.QLabel("")
        self.lian_ti_shi.setStyleSheet(_ys_biao_ti())
        di.addWidget(self.lian_ti_shi, 1)
        self.an_guanbi = QtWidgets.QPushButton("关闭")
        self.an_guanbi.setStyleSheet(_ys_ci_anniu())
        self.an_guanbi.setFixedHeight(26)
        self.an_guanbi.setMinimumWidth(72)
        di.addWidget(self.an_guanbi)
        wai.addLayout(di)

        self._jie_xinhao()
        self._chong_ku()
        self._chong_ku_liebiao()
        self._chong_jiao_liebiao()

    # ---------------------------------------------------------- 摆界面
    def _zuo_yi_bian(self, biao_ti, shang_biao):
        """摆一边（两边一个样）：列表 + 右边一排上下移箭头 + 底下一排按钮

        返回 (这一块的框, 列表, {箭头名: 按钮}, {动作名: 按钮})。
        """
        kuang = QtWidgets.QGroupBox(biao_ti)
        w = QtWidgets.QVBoxLayout(kuang)
        w.setSpacing(6)

        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(4)
        libiao = QtWidgets.QListWidget()
        libiao.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        libiao.setMinimumHeight(220)
        hang.addWidget(libiao, 1)
        jian = QtWidgets.QVBoxLayout()
        jian.setSpacing(2)
        jian.addStretch(1)
        pao = {}
        for ming, zi, ti_shi in (
            ("shang", "↑", "往上挪一格"),
            ("ding", "⇈", "挪到最前"),
            ("xia", "↓", "往下挪一格"),
            ("di", "⇊", "挪到最后"),
            ("pai", "A", "按名字排序"),
        ):
            an = QtWidgets.QPushButton(zi)
            an.setFixedSize(26, 24)
            an.setToolTip(ti_shi)
            an.setStyleSheet(_ys_ci_anniu())
            jian.addWidget(an)
            pao[ming] = an
        jian.addStretch(1)
        hang.addLayout(jian)
        w.addLayout(hang, 1)

        an_ban = QtWidgets.QPushButton(shang_biao)
        an_ban.setStyleSheet(_ys_ci_anniu())
        an_ban.setFixedHeight(26)
        an_ban.setToolTip(shang_biao)
        w.addWidget(an_ban)
        pao["ban"] = an_ban      # 这个按钮也要能取到（_jie_xinhao / _geng_xin_anniu 都按名字拿）

        an_men = QtWidgets.QHBoxLayout()
        an_men.setSpacing(4)
        dong = {}
        for ming, zi in (("xin", "新建"), ("bianji", "编辑"),
                         ("fuzhi", "复制"), ("shan", "删除")):
            an = QtWidgets.QPushButton(zi)
            an.setStyleSheet(_ys_ci_anniu())
            an.setFixedHeight(26)
            an.setMinimumWidth(62)
            an_men.addWidget(an)
            dong[ming] = an
        w.addLayout(an_men)
        return kuang, libiao, pao, dong

    def _jie_xinhao(self):
        self.xia_ku.currentIndexChanged.connect(self._ku_huan_le)
        self.an_ku_xin.clicked.connect(self._ku_xin)
        self.an_ku_shan.clicked.connect(self._ku_shan_zhengge)
        self.an_guanbi.clicked.connect(self.close)

        self.an_ku["ban"].clicked.connect(self._ku_dao_jiao)
        self.an_jiao["ban"].clicked.connect(self._jiao_dao_ku)

        self.an_ku_dong["xin"].clicked.connect(self._ku_xin_yangshi)
        self.an_ku_dong["bianji"].clicked.connect(self._ku_bianji_yangshi)
        self.an_ku_dong["fuzhi"].clicked.connect(self._ku_fuzhi_yangshi)
        self.an_ku_dong["shan"].clicked.connect(self._ku_shan_yangshi)

        self.an_jiao_dong["xin"].clicked.connect(self._jiao_xin_yangshi)
        self.an_jiao_dong["bianji"].clicked.connect(self._jiao_bianji_yangshi)
        self.an_jiao_dong["fuzhi"].clicked.connect(self._jiao_fuzhi_yangshi)
        self.an_jiao_dong["shan"].clicked.connect(self._jiao_shan_yangshi)

        for libiao, an_men, ku_bian in (
            (self.lib_ku, self.an_ku, True),
            (self.lib_jiao, self.an_jiao, False),
        ):
            libiao.itemSelectionChanged.connect(self._geng_xin_anniu)
            libiao.itemDoubleClicked.connect(
                lambda _x, k=ku_bian: (
                    self._ku_bianji_yangshi() if k else self._jiao_bianji_yangshi()
                )
            )
            for ming in ("shang", "ding", "xia", "di", "pai"):
                an_men[ming].clicked.connect(
                    lambda _c=False, k=ku_bian, m=ming: self._nuo_yi_nuo(k, m)
                )

    # ---------------------------------------------------------- 填 / 刷新
    @staticmethod
    def _tian_liebiao(libiao, ming_men, xuan=None):
        """往列表里塞名字，塞完把 xuan 里那几个选上"""
        libiao.clear()
        for x in ming_men or []:
            libiao.addItem(str(x))
        if xuan:
            yao = {str(x).strip().lower() for x in xuan}
            for i in range(libiao.count()):
                if libiao.item(i).text().strip().lower() in yao:
                    libiao.item(i).setSelected(True)

    @staticmethod
    def _xuan_ming(libiao):
        """列表里选中的那些名字"""
        return [x.text() for x in libiao.selectedItems()]

    def _ku_huan_le(self):
        self._ku = self.xia_ku.currentText()
        self._chong_ku_liebiao()

    def _chong_ku(self, xuan=None):
        """重画上面的库下拉（尽量还选着原来那个）"""
        jiu = self._ku or self.xia_ku.currentText()
        ming_men = lie_ku()
        self.xia_ku.blockSignals(True)
        self.xia_ku.clear()
        self.xia_ku.addItems(ming_men)
        if jiu and jiu in ming_men:
            self.xia_ku.setCurrentText(jiu)
        self.xia_ku.blockSignals(False)
        self._ku = self.xia_ku.currentText()
        self._shuo()

    def _chong_ku_liebiao(self, xuan=None):
        """重画库里的样式列表"""
        self._ku_men = self._du_hang(du_ku(self._ku)) if self._ku else []
        self._tian_liebiao(self.lib_ku, [x for x, _z in self._ku_men], xuan)
        self._shuo()
        self._geng_xin_anniu()

    def _chong_jiao_liebiao(self, xuan=None):
        """重画当前字幕的样式列表"""
        ming_men = list(self._jiekou.ym_jiao_lie())
        self._tian_liebiao(self.lib_jiao, ming_men, xuan)
        self._shuo()
        self._geng_xin_anniu()

    def _shuo(self, wen=""):
        """底下那行提示：说一句 + 现在两边各有多少条"""
        zi = f"样式库「{self._ku or '（没有）'}」{len(self._ku_men)} 条"
        try:
            zi += f" · 当前字幕 {len(self._jiekou.ym_jiao_lie())} 条"
        except Exception:  # noqa
            pass
        self.lian_ti_shi.setText((str(wen) + "　" if wen else "") + zi)

    def _geng_xin_anniu(self, *_a):
        """选了什么就开什么按钮（照 AEG 的 UpdateButtons）"""
        self.an_ku_shan.setEnabled(bool(lie_ku()) and len(lie_ku()) > 1)
        ku_xuan = self._xuan_ming(self.lib_ku)
        jiao_xuan = self._xuan_ming(self.lib_jiao)

        self.an_ku_dong["bianji"].setEnabled(len(ku_xuan) == 1)
        self.an_ku_dong["fuzhi"].setEnabled(len(ku_xuan) == 1)
        self.an_ku_dong["shan"].setEnabled(bool(ku_xuan))
        self.an_ku["ban"].setEnabled(bool(ku_xuan))

        self.an_jiao_dong["bianji"].setEnabled(len(jiao_xuan) == 1)
        self.an_jiao_dong["fuzhi"].setEnabled(len(jiao_xuan) == 1)
        self.an_jiao_dong["shan"].setEnabled(bool(jiao_xuan))
        self.an_jiao["ban"].setEnabled(bool(jiao_xuan))

        for libiao, an_men in (
            (self.lib_ku, self.an_ku), (self.lib_jiao, self.an_jiao)
        ):
            wei = [libiao.row(x) for x in libiao.selectedItems()]
            shu = libiao.count()
            yi = len(wei) == 1
            an_men["shang"].setEnabled(yi and wei[0] > 0)
            an_men["ding"].setEnabled(yi and wei[0] > 0)
            an_men["xia"].setEnabled(yi and wei[0] < shu - 1)
            an_men["di"].setEnabled(yi and wei[0] < shu - 1)
            an_men["pai"].setEnabled(shu > 1)

    # ---------------------------------------------------------- 库：整个库
    def _ku_xin(self):
        """新建一个库"""
        ming, _ok = QtWidgets.QInputDialog.getText(
            self, "新建样式库", "库名（就是一个 .sty 文件的名字）："
        )
        ming = str(ming or "").strip()
        if not ming:
            return
        if ku_zai(ming):
            QtWidgets.QMessageBox.warning(
                self, "样式库名重复", f"已经有一个叫「{ming}」的库了，换个名字。"
            )
            return
        ming = xin_ku(ming)
        if not ming:
            self._shuo("这个库建不出来（名字里的字文件名里不能用？）")
            return
        self._ku = ming
        self._chong_ku()
        self._chong_ku_liebiao()
        self._shuo(f"样式库「{ming}」建好了")

    def _ku_shan_zhengge(self):
        """删掉整个库"""
        if len(lie_ku()) <= 1:
            return
        if not wen_shi_fou(
            self, "确认删除",
            f"把样式库「{self._ku}」连同里面的样式一起删掉？",
        ):
            return
        jiu = self._ku
        shan_ku(jiu)
        self._ku = ""
        self._chong_ku()
        self._chong_ku_liebiao()
        self._shuo(f"样式库「{jiu}」删掉了")

    # ---------------------------------------------------------- 库：里面的样式
    def _ku_cun_hang(self, hang, xuan=None, shuo=""):
        """把库里的样式行写回 .sty 文件，然后重画"""
        if not xie_ku(self._ku, hang):
            self._shuo(f"这个库写不进去（{self._ku}）")
            return False
        self._chong_ku_liebiao(xuan)
        if shuo:
            self._shuo(shuo)
        return True

    @staticmethod
    def _wei_yi_ming(ming, yi_you):
        """照 AEG 的 unique_name：重名加「 - Copy」，还重就往后排序号"""
        ming = str(ming or "").strip() or "Default"
        di = {str(x).strip().lower() for x in yi_you}
        if ming.lower() not in di:
            return ming
        xin = f"{ming} - Copy"
        shu = 2
        while xin.lower() in di:
            xin = f"{ming} - Copy ({shu})"
            shu += 1
        return xin

    def _ku_zhao(self, ming):
        """在当前库里找一条样式，返回字段；找不到给 None"""
        ming = str(ming or "").strip().lower()
        for x, zi in self._ku_men:
            if x.strip().lower() == ming:
                return zi
        return None

    def _ku_xin_yangshi(self):
        """库里新建一条样式"""
        if not self._ku:
            return
        zi = dict(_YS_XIN_YANGSHI)
        zi["name"] = self._wei_yi_ming(zi["name"], [x for x, _z in self._ku_men])
        self._kai_bianji(True, zi, "")

    def _ku_bianji_yangshi(self):
        """改库里选中的那一条"""
        xuan = self._xuan_ming(self.lib_ku)
        if len(xuan) != 1:
            return
        zi = self._ku_zhao(xuan[0])
        if zi is None:
            return
        self._kai_bianji(True, dict(zi), xuan[0])

    def _ku_fuzhi_yangshi(self):
        """照库里这条复制一份（名字自动不重）"""
        xuan = self._xuan_ming(self.lib_ku)
        if len(xuan) != 1:
            return
        zi = self._ku_zhao(xuan[0])
        if zi is None:
            return
        zi = dict(zi)
        zi["name"] = self._wei_yi_ming(
            zi.get("name"), [x for x, _z in self._ku_men]
        )
        self._kai_bianji(True, zi, "")

    def _ku_shan_yangshi(self):
        """从库里删掉选中的那几条"""
        xuan = self._xuan_ming(self.lib_ku)
        if not xuan:
            return
        if not wen_shi_fou(
            self, "确认删除",
            f"从样式库「{self._ku}」里删掉这 {len(xuan)} 条样式？",
        ):
            return
        di = {x.strip().lower() for x in xuan}
        hang = [
            self._zuo_hang(x, z) for x, z in self._ku_men
            if x.strip().lower() not in di
        ]
        self._ku_cun_hang(hang, shuo=f"从库里删掉 {len(xuan)} 条")

    # ---------------------------------------------------------- 字幕：里面的样式
    def _jiao_xin_yangshi(self):
        """字幕里新建一条样式"""
        zi = dict(_YS_XIN_YANGSHI)
        zi["name"] = self._wei_yi_ming(
            zi["name"], list(self._jiekou.ym_jiao_lie())
        )
        self._kai_bianji(False, zi, "")

    def _jiao_bianji_yangshi(self):
        xuan = self._xuan_ming(self.lib_jiao)
        if len(xuan) != 1:
            return
        self._kai_bianji(
            False, dict(self._jiekou.ym_jiao_du(xuan[0])), xuan[0]
        )

    def _jiao_fuzhi_yangshi(self):
        xuan = self._xuan_ming(self.lib_jiao)
        if len(xuan) != 1:
            return
        zi = dict(self._jiekou.ym_jiao_du(xuan[0]))
        zi["name"] = self._wei_yi_ming(
            zi.get("name"), list(self._jiekou.ym_jiao_lie())
        )
        self._kai_bianji(False, zi, "")

    def _jiao_shan_yangshi(self):
        xuan = self._xuan_ming(self.lib_jiao)
        if not xuan:
            return
        if not wen_shi_fou(
            self, "确认删除",
            f"把当前字幕里的这 {len(xuan)} 条样式删掉？",
        ):
            return
        if self._jiekou.ym_jiao_shan(xuan):
            self._chong_jiao_liebiao()
            self.gaile.emit()
            self._shuo(f"字幕里删掉 {len(xuan)} 条样式")

    # ---------------------------------------------------------- 两边互相搬
    def _wen_yi_ci_chong_ming(self, chong, zong, na_bian):
        """同名的一次问完（AEG 是撞一个问一次，一次搬几十条就要点几十回）

        点「是」= 同名的全都覆盖；点「否」= 跳过同名的，只把新的搬过去。
        """
        li = "、".join(f"「{x}」" for x in chong[:6])
        if len(chong) > 6:
            li += f"等 {len(chong)} 条"
        return wen_shi_fou(
            self, "样式名冲突",
            f"这次要搬 {zong} 条样式，其中 {len(chong)} 条在{na_bian}里已经有同名的：{li}。",
            "点「是」同名的全部覆盖；点「否」跳过同名的，只加新样式。",
        )

    def _ku_dao_jiao(self):
        """「复制到当前字幕 →」（照 AEG 的 Copy to current script）

        同名的不逐条弹框，一次问完。
        """
        xuan = self._xuan_ming(self.lib_ku)
        if not xuan:
            return
        yao = [(m, self._ku_zhao(m)) for m in xuan]
        yao = [(m, z) for m, z in yao if z is not None]     # 库里查不到的跳过
        if not yao:
            return
        yi = {x.strip().lower() for x in self._jiekou.ym_jiao_lie()}
        chong = [m for m, _z in yao if m.strip().lower() in yi]
        if chong and not self._wen_yi_ci_chong_ming(
            chong, len(yao), "当前字幕"
        ):
            yao = [(m, z) for m, z in yao if m.strip().lower() not in yi]
        cheng = []
        for ming, zi in yao:
            if self._jiekou.ym_jiao_cun("", zi):
                cheng.append(ming)
        if cheng:
            self._chong_jiao_liebiao(cheng)
            self.gaile.emit()
            self._shuo(f"复制了 {len(cheng)} 条样式到当前字幕")
        else:
            self._shuo("一条也没搬过去（同名的都跳过了？）")

    def _jiao_dao_ku(self):
        """「← 存进样式库」（照 AEG 的 Copy to storage）

        同名的不逐条弹框，一次问完。
        """
        xuan = self._xuan_ming(self.lib_jiao)
        if not xuan or not self._ku:
            return
        yi = {x.strip().lower() for x, _z in self._ku_men}
        chong = [m for m in xuan if m.strip().lower() in yi]
        if chong and not self._wen_yi_ci_chong_ming(
            chong, len(xuan), f"样式库「{self._ku}」"
        ):
            xuan = [m for m in xuan if m.strip().lower() not in yi]
        hang = [self._zuo_hang(x, z) for x, z in self._ku_men]
        cheng = []
        for ming in xuan:
            xin_hang = self._zuo_hang(ming, self._jiekou.ym_jiao_du(ming))
            if not xin_hang:
                continue
            if ming.strip().lower() in yi:
                for i, (x, _z) in enumerate(self._ku_men):
                    if x.strip().lower() == ming.strip().lower():
                        hang[i] = xin_hang
                        break
            else:
                hang.append(xin_hang)
                yi.add(ming.strip().lower())
            cheng.append(ming)
        if cheng:
            self._ku_cun_hang(
                hang, cheng, f"存进库「{self._ku}」 {len(cheng)} 条"
            )
        else:
            self._shuo("一条也没存进去（同名的都跳过了？）")

    # ---------------------------------------------------------- 上下移 / 排序
    @staticmethod
    def _pai_hou(ming_men, xuan, dong):
        """照 AEG 的 MoveStyles：把选中的那几条挪一挪，返回新的顺序"""
        men = list(ming_men)
        xuan = [x for x in (xuan or []) if x in men]
        if not xuan:
            return men
        wei = sorted(men.index(x) for x in xuan)
        if dong == "pai":
            return sorted(men)
        if len(wei) == 1:
            i = wei[0]
            if dong == "shang" and i > 0:
                men[i - 1], men[i] = men[i], men[i - 1]
            elif dong == "xia" and i < len(men) - 1:
                men[i], men[i + 1] = men[i + 1], men[i]
            elif dong == "ding":
                men.insert(0, men.pop(i))
            elif dong == "di":
                men.append(men.pop(i))
            return men
        kuai = [men[i] for i in wei]
        if dong == "shang" and wei[0] > 0:
            for i in wei:
                men[i - 1], men[i] = men[i], men[i - 1]
        elif dong == "xia" and wei[-1] < len(men) - 1:
            for i in reversed(wei):
                men[i], men[i + 1] = men[i + 1], men[i]
        elif dong == "ding":
            men = kuai + [x for x in men if x not in kuai]
        elif dong == "di":
            men = [x for x in men if x not in kuai] + kuai
        return men

    def _nuo_yi_nuo(self, ku_bian, dong):
        """点了上下移 / 排序：库那边就写文件，字幕那边就交回工作台去重排"""
        if ku_bian:
            xuan = self._xuan_ming(self.lib_ku)
            shun = self._pai_hou([x for x, _z in self._ku_men], xuan, dong)
            zi_men = {x: z for x, z in self._ku_men}
            hang = []
            for x in shun:
                xin = self._zuo_hang(x, zi_men.get(x) or {})
                if xin:
                    hang.append(xin)
            self._ku_cun_hang(hang, xuan)
            return
        xuan = self._xuan_ming(self.lib_jiao)
        shun = self._pai_hou(list(self._jiekou.ym_jiao_lie()), xuan, dong)
        if self._jiekou.ym_jiao_pai(shun):
            self._chong_jiao_liebiao(xuan)
            self.gaile.emit()

    # ---------------------------------------------------------- 样式编辑器
    def _ming_jian_cha(self, ku_bian, jiu_ming):
        """名字检查：空的 / 跟同一边别的样式撞了就拦下来（照 AEG 的 Apply）

        返回一个函数给样式编辑器，它返回 False 就不关窗口、让他改名字。
        """
        def _cha(z):
            xin = str(z.get("name") or "").strip()
            if not xin:
                QtWidgets.QMessageBox.warning(
                    self, "名字不能空", "样式名不能是空的，随便起一个。"
                )
                return False
            if xin.lower() == str(jiu_ming or "").strip().lower() and jiu_ming:
                return True
            if ku_bian:
                yi = [x for x, _z in self._ku_men]
            else:
                yi = list(self._jiekou.ym_jiao_lie())
            if any(str(x).strip().lower() == xin.lower() for x in yi):
                QtWidgets.QMessageBox.warning(
                    self, "样式名重复",
                    f"这边已经有叫「{xin}」的样式了，换个名字。",
                )
                return False
            return True

        return _cha

    def _kai_bianji(self, ku_bian, zi, jiu_ming):
        """开样式编辑器（库那边 / 字幕那边共用同一个编辑器）"""
        jiu = self._bianji_chuang
        if jiu is not None:
            jiu.showNormal()
            jiu.raise_()
            jiu.activateWindow()
            return
        yi = (
            [x for x, _z in self._ku_men] if ku_bian
            else list(self._jiekou.ym_jiao_lie())
        )
        dlg = YangshiBianjiDialog(
            zi, self, self._ziti_men, self._jizhun, self._ziti_gongchang,
            yang_men=yi, ming_jian_cha=self._ming_jian_cha(ku_bian, jiu_ming),
        )
        self._bianji_chuang = dlg
        dlg.queren.connect(
            lambda z: self._luo_ku(jiu_ming, z) if ku_bian
            else self._luo_jiao(jiu_ming, z)
        )
        dlg.finished.connect(lambda _=0: setattr(self, "_bianji_chuang", None))
        dlg.show()

    def _luo_ku(self, jiu_ming, zi):
        """样式编辑器点了「确定 / 应用」，改的是样式库里的那一条"""
        xin = str(zi.get("name") or "").strip() or "Default"
        xin_hang = self._zuo_hang(xin, zi)
        if not xin_hang:
            self._shuo("这份样式写不出来，没存")
            return
        zhao = -1
        for i, (x, _z) in enumerate(self._ku_men):
            if x.strip().lower() == str(jiu_ming or "").strip().lower():
                zhao = i
                break
        hang = [self._zuo_hang(x, z) for x, z in self._ku_men]
        if zhao >= 0:
            hang[zhao] = xin_hang
            shuo = f"样式「{xin}」改好了"
        else:
            hang.append(xin_hang)
            shuo = f"样式「{xin}」加进库了"
        if self._ku_cun_hang(hang, [xin]):
            self._shuo(shuo)

    def _luo_jiao(self, jiu_ming, zi):
        """样式编辑器点了「确定 / 应用」，改的是当前字幕里的那一条"""
        xin = str(zi.get("name") or "").strip() or "Default"
        if not self._jiekou.ym_jiao_cun(jiu_ming, zi):
            self._shuo("字幕这份样式没写进去（字幕还没打开样式表？）")
            return
        self._chong_jiao_liebiao([xin])
        self.gaile.emit()
        self._shuo(f"字幕的样式「{xin}」{'改好了' if jiu_ming else '加上了'}")


class _ZimuLiebiaoBiao(QtWidgets.QTableWidget):
    """字幕列表那张表：Ctrl+滚轮 改字号

    字号滚完在表的右上角闪一下（跟编辑框那个浮标一个意思，免得瞎调），
    存在 gongzuotai.ini 里，下次开软件还是这个大小。
    字号一改，行高、表头高、各列宽一起按比例跟着变（照 AEG：表格整个跟着字走）。
    列宽只看字号、不看窗口多宽（AEG 的 SetColumnWidths 也是字号一变才重算），
    文本列写死一个很宽的值 —— 字幕长了直接出屏，表格不跟着撑大，也没横向滚动。
    """

    xiayitiao = pyqtSignal()    # 焦点在这张表上按回车：换到下一条字幕
    yao_yidong = pyqtSignal(int)    # Alt+↑ / Alt+↓：选中的行整批挪一格（-1 上 / 1 下）

    HANG_GAO_JICHU = 24         # 基准行高（字号没超过它的时候就这么高）
    TOU_GAO_JICHU = 26          # 基准表头高
    SHIJIAN_LIE = (1, 2)        # 开始时间 / 结束时间这两列（宽度按文字实算）
    WENBEN_KUAN = 5000          # 文本列的宽（照 AEG 源码里文本列 Width() 的 5000）

    def __init__(self, hang=0, lie=0, parent=None):
        super().__init__(hang, lie, parent)
        self._zihao = du_liebiao_zihao()
        self._lie_kuan_jichu = ()       # 基准列宽（按 ZIMU_LIEBIAO_ZIHAO_MOREN 号字定的），外面给
        self._wenben_lie = 0            # 文本列在第几列（外面给；它吃剩余宽度）
        self._pai_zhu = False           # 正在铺列宽（防 resize 来回触发）
        self._yingyong()
        self._zihao_ti = QtWidgets.QLabel(self)
        self._zihao_ti.setStyleSheet(_ys_zihao_ti())
        self._zihao_ti.hide()
        self._zihao_ti_ji = QtCore.QTimer(self)
        self._zihao_ti_ji.setSingleShot(True)
        self._zihao_ti_ji.timeout.connect(self._zihao_ti.hide)

    def zihao(self):
        return self._zihao

    def shezhi_liekuan_jizhun(self, lie_kuan, wenben_lie=None):
        """把"默认字号下各列多宽"交给表管，之后字号一变列宽自己按比例缩

        wenben_lie 是文本列在第几列（不给就当成最后一列）—— 文本列吃剩余的
        宽度，得知道是哪一列。
        """
        self._lie_kuan_jichu = tuple(lie_kuan or ())
        if wenben_lie is not None:
            self._wenben_lie = int(wenben_lie)
        elif self._lie_kuan_jichu:
            self._wenben_lie = len(self._lie_kuan_jichu) - 1
        self._yingyong()

    def _yingyong(self):
        """把样式、字体、行高、列宽按当前字号摆上（字号一改就得重摆一次）

        字体除了写进样式表，这里再用代码设一遍：样式表里那个 font-family
        在表头那一块不总是认，代码设上去表头和表体才是一个脸。
        列宽交给 _pai_liekuan：它保证所有列加起来不超出视口宽（不出现横向滚动）。
        """
        self.setStyleSheet(_ys_zimu_biao(self._zihao))
        ziti = _liebiao_ziti_ming()
        zi = QtGui.QFont(self.font())
        if ziti:
            zi.setFamily(ziti)
        zi.setPixelSize(int(self._zihao))
        zi.setBold(True)         # 编辑框也是粗的，列表跟它一个脸
        self.setFont(zi)
        tou = self.horizontalHeader()
        if tou is not None:
            tou.setFont(zi)
        self.verticalHeader().setDefaultSectionSize(
            max(self.HANG_GAO_JICHU, self._zihao + 12)
        )
        bi = self._zihao / float(max(1, ZIMU_LIEBIAO_ZIHAO_MOREN))
        self._pai_liekuan()
        if tou is not None:
            tou.setFixedHeight(
                max(self.TOU_GAO_JICHU, int(round(self.TOU_GAO_JICHU * bi)))
            )

    def _pai_liekuan(self):
        """列宽只跟字号走，不看窗口多大（照 AEG 的 SetColumnWidths：字号/字体一变
        重算一次，窗口拉伸不重算）

        各列宽 = 基准宽 × 字号比；时间那两列按文字实宽兜底。文本列宽度写死
        WENBEN_KUAN（AEG 源码里文本列 Width() 直接 return 5000）—— 字幕长了就是
        出屏幕，表格不跟着撑大，也没有横向滚动条。
        """
        if not self._lie_kuan_jichu or self._pai_zhu:
            return
        lie_shu = self.columnCount()
        if lie_shu <= 0:
            return
        self._pai_zhu = True
        try:
            # 基准宽只给了前几列（文本列没给），少的用最后一个补
            jichu = list(self._lie_kuan_jichu)
            if len(jichu) < lie_shu:
                jichu += [jichu[-1]] * (lie_shu - len(jichu))
            wen = max(0, min(self._wenben_lie, lie_shu - 1))
            fm = QtGui.QFontMetrics(self.font())
            bi = self._zihao / float(max(1, ZIMU_LIEBIAO_ZIHAO_MOREN))
            # 时间那两列按字的实际宽度算：光按比例缩不够，字号一大会被省略号截成 0:02:18…
            shi_kuan = fm.horizontalAdvance("0:00:00.000") + 20
            for lie in range(lie_shu):
                if lie == wen:
                    kuan = self.WENBEN_KUAN
                else:
                    kuan = int(round(jichu[lie] * bi))
                    if lie in self.SHIJIAN_LIE:
                        kuan = max(kuan, shi_kuan)
                self.setColumnWidth(lie, max(10, kuan))
        finally:
            self._pai_zhu = False

    def scrollContentsBy(self, dx, dy):
        """横向一律不滚（AEG 的字幕列表也没有横向滚动），只让它竖着滚"""
        super().scrollContentsBy(0, dy)

    def keyPressEvent(self, event):
        """回车 = 换到下一条字幕（AEG 的列表里也是这个手感）

        Alt + ↑ / Alt + ↓ = 把选中的行整批往上 / 往下挪一格（照 AEG 的
        Move line up / Move line down，挪的是行在文件里的先后，时间不改）。
        """
        if event.modifiers() & Qt.AltModifier and event.key() in (
            Qt.Key_Up, Qt.Key_Down
        ):
            self.yao_yidong.emit(-1 if event.key() == Qt.Key_Up else 1)
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.xiayitiao.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def shezhi_zihao(self, zihao, ji=True):
        """改字号（超范围夹到上下限）；返回改完之后的字号"""
        zihao = max(
            ZIMU_LIEBIAO_ZIHAO_ZUI_XIAO,
            min(ZIMU_LIEBIAO_ZIHAO_ZUIDA, int(zihao)),
        )
        if zihao != self._zihao:
            self._zihao = zihao
            self._yingyong()
            if ji:
                xie_liebiao_zihao(zihao)     # 记住，下次开软件还是这个大小
        return self._zihao

    def _tan_zihao_ti(self):
        """滚完字号在表右上角闪一下：「字号 20」「字号 100（最大）」"""
        shuo = f"字号 {self._zihao}"
        if self._zihao >= ZIMU_LIEBIAO_ZIHAO_ZUIDA:
            shuo += "（最大）"
        elif self._zihao <= ZIMU_LIEBIAO_ZIHAO_ZUI_XIAO:
            shuo += "（最小）"
        self._zihao_ti.setText(shuo)
        self._zihao_ti.adjustSize()
        self._zihao_ti.move(
            max(4, self.width() - self._zihao_ti.width() - 18), 8
        )
        self._zihao_ti.show()
        self._zihao_ti.raise_()
        self._zihao_ti_ji.start(1200)

    def wheelEvent(self, event):
        """带 Ctrl 的滚轮改字号；不带 Ctrl 的还是正常上下滚"""
        if not (event.modifiers() & Qt.ControlModifier):
            super().wheelEvent(event)
            return
        gun = event.angleDelta().y()
        if gun == 0:
            super().wheelEvent(event)
            return
        self.shezhi_zihao(self._zihao + (1 if gun > 0 else -1))
        self._tan_zihao_ti()
        event.accept()


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
    xuan_zhong_duo = pyqtSignal(list)       # 列表里多选 / 全选：选中的这一批行号
    tiaozheng = pyqtSignal(int)             # 请求跳到某个 ms（这个面板里等价于"某条的开头"）
    zimu_qi_tiao = pyqtSignal(int)          # 选了某条 -> 请求跳到这条的开头（块头，认帧按 AEG 的 START）
    wenben_gaile = pyqtSignal(int, str)     # 正在打字：第几条的文字改成了什么
    bianji_wancheng = pyqtSignal(int, str)  # 离开编辑框：这一条改完了
    gongju_gaile = pyqtSignal(str, object)  # 工具栏改了哪一样（见 _ZimuGongjulan）
    yao_yidong = pyqtSignal(int)            # 列表里 Alt+↑ / Alt+↓：选中的行整批挪一格

    LIE = ("#", "开始时间", "结束时间", "字/秒", "样式", "说话人", "文本")
    LIE_ZHEN = ("#", "开始帧", "结束帧", "字/秒", "样式", "说话人", "文本")
    LIE_KUAN = (40, 78, 78, 44, 68, 74)     # 前几列的宽度（文本列吃掉剩下的）
    LIE_TOU_ZUO = (4, 5)                    # 表头左对齐的列：样式 / 说话人
    LIE_WENBEN = 6                          # 文本列是第几列
    HANG_GAO = 24                           # 一行多高

    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setHandleWidth(9)
        self.setChildrenCollapsible(False)
        self._zimu = []
        self._fujia = []        # 每条的字段表（样式 / 说话人 / 层 / 边距 / 特效 / 注释）
        self._xu = -1           # 手动选中（正在编辑）的是第几条
        self._bofang_hang = -1  # 播放头正停在第几条（列表里涂蓝底的那行）
        self._tian = False      # 正往框里塞文字，这时候的 textChanged 不算用户改
        self._fps = 0.0         # 视频帧率：按帧看时间用（外面 shezhi_fps 给）
        self._zhen = False      # 时间那两列显示帧号还是时间
        self.zidong_tiao = True  # 列表里点一条 -> 画面跟不跟着跳（外面那个开关定）

        # 上块：字幕列表
        shang = QtWidgets.QFrame()
        shang.setObjectName("YsgPanel")
        # 「字幕列表」这一整块（标题 + 那张表）留着给外面搬：做字幕时整块挪到
        # 窗口下方，打轴时留在右栏。搬的时候连标题、条数、表一起走。
        self.lie_biao_kuang = shang
        self._shang_bu = QtWidgets.QVBoxLayout(shang)
        shang_bu = self._shang_bu
        shang_bu.setContentsMargins(12, 10, 12, 10)
        shang_bu.setSpacing(6)

        # 「字幕列表」标题 + 条数那行。铺到窗口下方时要整行收掉（照 AEG 顶格
        # 显示），所以裹成一个控件 —— 收的时候连它占的间距一起没收
        self.lie_biao_tou = QtWidgets.QWidget()
        ding = QtWidgets.QHBoxLayout(self.lie_biao_tou)
        ding.setContentsMargins(0, 0, 0, 0)
        ding.setSpacing(6)
        ding.addWidget(_biaoti_wenben("字幕列表"))
        ding.addStretch(1)
        self.shu_wenben = QtWidgets.QLabel("0 条")
        self.shu_wenben.setObjectName("YsgHint")
        ding.addWidget(self.shu_wenben)
        shang_bu.addWidget(self.lie_biao_tou)

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

        # 编辑区上方那排字段（照 AEG 的主工具栏）：改哪一样就往外发哪一样
        self.gongjulan = _ZimuGongjulan()
        self.gongjulan.gongju_gaile.connect(self.gongju_gaile.emit)
        xia_bu.addWidget(self.gongjulan)

        self.kuang = _ZimuWenbenKuang()
        self.kuang.setStyleSheet(_ys_bianji_kuang(self.kuang.zihao()))
        self.kuang.setMinimumHeight(80)
        self.kuang.setEnabled(False)
        self.kuang.textChanged.connect(self._wenben_bian)
        self.kuang.likai.connect(self._bianji_wancheng)
        self.kuang.xiayitiao.connect(self._xia_yi_tiao)
        # 上排那些按钮得能看到编辑框里选中了哪一段（选了就只改那一段）
        self.gongjulan.shezhi_wenben_kuang(self.kuang)
        xia_bu.addWidget(self.kuang, 1)

        self.addWidget(shang)
        self.addWidget(xia)
        self.setSizes([320, 240])

    def shezhi_liebiao_dingge(self, dingge):
        """字幕列表铺到窗口下方时把「标题 + 条数」那行收掉（照 AEG 顶格显示）

        AEG 的列表铺在下面时就直接顶到边上，头上不留标题；收掉之后上边距
        也一并归零，不然还空着一条。
        """
        self.lie_biao_tou.setVisible(not dingge)
        self._shang_bu.setContentsMargins(12, 0 if dingge else 10, 12, 10)

    def _jian_biao(self):
        """字幕列表本体：ASS 那套字段摆成一张带网格的表"""
        biao = _ZimuLiebiaoBiao(0, len(self.LIE))
        self.biao = biao
        self._she_tou()
        biao.verticalHeader().setVisible(False)
        biao.setShowGrid(True)
        biao.setWordWrap(False)
        biao.setAlternatingRowColors(False)
        biao.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # 能多选：Ctrl+A 全选、Shift + 鼠标点击连选一片、Ctrl + 鼠标点击点一条
        biao.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        biao.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        biao.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )
        biao.setVerticalScrollMode(
            # 竖着按"行"滚，不按像素滚：按像素滚会停在半行上，
            # 顶上一行只露出半截字，看着很怪
            QtWidgets.QAbstractItemView.ScrollPerItem
        )
        # 样式、行高、列宽、表头高都归 _ZimuLiebiaoBiao 管（跟着 Ctrl+滚轮 调的字号走）
        tou = biao.horizontalHeader()
        tou.setHighlightSections(False)
        tou.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)  # 列宽能拖着改
        tou.setStretchLastSection(False)    # 文本列宽自己算，不交给 Qt 拉
        # 横向滚动条永远不出来：列宽由 _pai_liekuan 铺满视口，字幕再长也只
        # 在文本列里被裁（照 AEG：字幕出屏幕就是出屏幕，表格不会跟着撑大）
        biao.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        biao.shezhi_liekuan_jizhun(self.LIE_KUAN, self.LIE_WENBEN)
        biao.itemSelectionChanged.connect(self._xuan_zhong_bian)
        biao.xiayitiao.connect(self._xia_yi_tiao)   # 列表里按回车 = 换下一条
        # Alt+↑ / Alt+↓ = 整批挪行，具体怎么挪是外面的事（照 AEG 的 Move line up/down）
        biao.yao_yidong.connect(self.yao_yidong)
        return biao

    def _tou_wenben(self):
        """表头文字：看帧的时候那两列写成「开始帧 / 结束帧」"""
        return self.LIE_ZHEN if self._zhen else self.LIE

    def _she_tou(self):
        """铺表头（文字 + 对齐）

        对齐照 AEG：样式 / 说话人这两列的数据是左对齐的，表头也跟着左对齐，
        居中会跟下面的字对不上、很跳。其余列仍是居中。
        """
        self.biao.setHorizontalHeaderLabels(list(self._tou_wenben()))
        for lie in self.LIE_TOU_ZUO:
            xiang = self.biao.horizontalHeaderItem(lie)
            if xiang is not None:
                xiang.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def shezhi_fps(self, fps):
        """视频帧率给过来（按帧看时间要用）"""
        try:
            fps = float(fps or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        self._fps = max(0.0, fps)
        self.gongjulan.shezhi_fps(self._fps)
        if self._zhen:      # 帧率变了，帧号得重算
            self._chong_xie_shijian()

    def shezhi_xianshi_moshi(self, zhen):
        """看时间 / 看帧：表头和时间那两列一起换（"zhen"/"shijian" 字符串也认）"""
        if isinstance(zhen, str):
            zhen = str(zhen).strip().lower() == "zhen"
        zhen = bool(zhen)
        if zhen == self._zhen:
            return
        self._zhen = zhen
        self._she_tou()
        self.gongjulan.shezhi_moshi(zhen)
        self._chong_xie_shijian()

    def _chong_xie_shijian(self):
        """按当前模式把每一行的时间列重写一遍"""
        for i, (qi, zhi, wenben) in enumerate(self._zimu):
            self._xie_hang(i, qi, zhi, wenben)

    # ---- 外部接口 ----
    def shezhi_zimu(self, zimu, xuan=None, fujia=None):
        """整份字幕换掉（打开字幕 / 删除 / 合并 / 时间改完）

        fujia = 每条的字段表（样式 / 说话人 / 层 / 边距 / 特效 / 注释），跟 zimu
        一一对应；不给就全用默认。老写法（样式, 说话人）两元的也认。
        """
        self._zimu = [tuple(x) for x in (zimu or [])]
        self._fujia = []
        for i in range(len(self._zimu)):
            fu = {}
            if fujia is not None and i < len(fujia) and fujia[i]:
                jiu = fujia[i]
                if isinstance(jiu, dict):
                    fu = dict(jiu)
                else:
                    fu = {"yang": jiu[0], "shuo": jiu[1] if len(jiu) > 1 else ""}
            fu["yang"] = str(fu.get("yang") or "Default")
            fu["shuo"] = str(fu.get("shuo") or "")
            self._fujia.append(fu)
        self.biao.blockSignals(True)
        self.biao.clearContents()
        self.biao.setRowCount(len(self._zimu))
        for i, (qi, zhi, wenben) in enumerate(self._zimu):
            self._xie_hang(i, qi, zhi, wenben)
        self.biao.blockSignals(False)
        self.shu_wenben.setText(f"{len(self._zimu)} 条")
        self._xu = -1
        self._bofang_hang = -1      # 整表重建了，蓝底由下一次刷播放头重涂
        self.shezhi_xuan_zhong(-1 if xuan is None else int(xuan))

    def shezhi_bofang_ms(self, ms, gensui=True):
        """播放头到哪一条了：把那一行涂成"正播到"的底色（蓝），跟着滚动

        gensui=False 就只涂色、不滚动列表。暂停着的时候必须这样 —— 整表重
        建（打注释、改字段）会把"上一行是哪一行"忘掉，紧接着那次刷新就拿着
        还没动的播放头位置去滚列表，列表"啪"地跳回最顶上；等真正的定位回来
        又滚下去，看着就是列表来回闪。播放中才跟着滚。

        跟手动选中的那一行各管各的 —— 正播行用格子自己的背景色（蓝），
        选中行用样式表里的选中色（绿），样式表的优先级更高，所以两行撞在
        一起时显示的是绿色（编辑优先），播放走开了又回到本来的配色。
        """
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            return
        xu = -1
        for i, (qi, zhi, _wenben) in enumerate(self._zimu):
            if int(qi) <= ms <= int(zhi):
                xu = i
                break
        if xu == self._bofang_hang:
            return
        jiu = self._bofang_hang
        self._bofang_hang = xu
        zhu = QtGui.QBrush(QtGui.QColor(ZIMU_LIEBIAO_BOFANG))  # 正播行：蓝底 + 白字
        bai = QtGui.QBrush(QtGui.QColor("#ffffff"))
        yuan = QtGui.QBrush()                               # 空刷子 = 还原默认
        for hang in (jiu, xu):
            if hang < 0:
                continue
            for lie in range(self.biao.columnCount()):
                xiang = self.biao.item(hang, lie)
                if xiang is None:
                    continue
                if hang == xu:
                    xiang.setBackground(zhu)
                    xiang.setForeground(bai)
                elif lie == 0:
                    # 序号列有自己那套黑底白字（照 AEG），别还原成默认配色
                    xiang.setBackground(
                        QtGui.QBrush(QtGui.QColor(ZIMU_LIEBIAO_XU_DISE))
                    )
                    xiang.setForeground(
                        QtGui.QBrush(QtGui.QColor(ZIMU_LIEBIAO_XU_ZI))
                    )
                else:
                    xiang.setBackground(self._hang_dise(hang))
                    xiang.setForeground(yuan)
        if xu >= 0 and gensui:
            xiang = self.biao.item(xu, 0)
            if xiang is not None:
                self.biao.scrollToItem(
                    xiang, QtWidgets.QAbstractItemView.EnsureVisible
                )

    def shezhi_yangshi_ming(self, ming_liebiao):
        """样式下拉的候选（外面从 ASS 的样式表里读出来给）"""
        self.gongjulan.shezhi_yangshi_ming(ming_liebiao)

    def shezhi_shuo_ming(self, ming_liebiao):
        """说话人下拉的候选"""
        self.gongjulan.shezhi_shuo_ming(ming_liebiao)

    def shezhi_gongju(self, fu, gs=None, wenben=None):
        """把当前这条的字段和"实际生效的样式"填进上面那排"""
        self.gongjulan.shezhi_gongju(fu, gs, wenben)

    def hang_fu(self, xu):
        """第 xu 条的字段表（外面要往 ASS 里写的时候用）"""
        xu = int(xu)
        if not (0 <= xu < len(self._fujia)):
            return {}
        return dict(self._fujia[xu])

    def gai_hang_fu(self, xu, fu):
        """外面改了这一条的字段：本地这份和列表里那两列跟上"""
        xu = int(xu)
        if not (0 <= xu < len(self._fujia)) or not fu:
            return
        self._fujia[xu].update(dict(fu))
        qi, zhi, wenben = self._zimu[xu]
        self._xie_hang(xu, qi, zhi, wenben)

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
        if self.kuang.wenben() != wenben:
            self._tian = True
            self.kuang.shezhi_wenben(wenben)
            self._tian = False
        # 光标落到这条的末尾：换过来接着敲字就是往后接，不会插到最前面
        self.kuang.moveCursor(QtGui.QTextCursor.End)
        # 上面那排跟着换成这一条的（外面随后还会把"实际生效的样式"再填一遍）
        fu = {}
        if xu >= 0:
            fu = dict(self._fujia[xu]) if xu < len(self._fujia) else {}
            fu["qi"], fu["zhi"] = self._zimu[xu][0], self._zimu[xu][1]
        self.gongjulan.shezhi_gongju(fu, None)
        self._shua_jishu()

    def _shua_jishu(self):
        """右下角计数刷新：当前选中的单条 / 总条数 / 剩余条数"""
        self.gongjulan.shezhi_jishu(self._xu, len(self._zimu))

    def zimu_liebiao(self):
        return list(self._zimu)

    def dangqian_xu(self):
        """现在编辑的是第几条；没选中给 -1"""
        return self._xu

    def xia_yi_tiao(self):
        """换到下一条字幕（时间轴上按回车时外面喊这个），跟编辑区回车一个走法"""
        self._xia_yi_tiao()

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
    def _hang_dise(self, hang):
        """这一行本来的底色：注释行涂橘红，普通行不涂（空刷子 = 跟样式表走）"""
        fu = self._fujia[hang] if 0 <= int(hang) < len(self._fujia) else {}
        if fu.get("zhushi"):
            return QtGui.QBrush(QtGui.QColor(YANSE_ZHUSHI_HANG))
        return QtGui.QBrush()

    def _xie_hang(self, i, qi, zhi, wenben):
        """把第 i 行那几格填上（格子还没有就新建）"""
        fu = self._fujia[i] if i < len(self._fujia) else {}
        yang = str(fu.get("yang") or "Default")
        shuo = str(fu.get("shuo") or "")
        if self._zhen:
            # 开始列按"块头"认、结束列按"块尾"认（照 AEG，见 ms_tou_zhen）
            qi_wen = _zhen_wenben(qi, self._fps, wei=False)
            zhi_wen = _zhen_wenben(zhi, self._fps, wei=True)
        else:
            qi_wen = _ass_shi_jian_wenben(qi)
            zhi_wen = _ass_shi_jian_wenben(zhi)
        zi = (
            str(i + 1),
            qi_wen,
            zhi_wen,
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
                if lie == 0:
                    # 序号列照 AEG：黑底白字（这一列不吃注释行那层暗色，见下面）
                    xiang.setBackground(
                        QtGui.QBrush(QtGui.QColor(ZIMU_LIEBIAO_XU_DISE))
                    )
                    xiang.setForeground(
                        QtGui.QBrush(QtGui.QColor(ZIMU_LIEBIAO_XU_ZI))
                    )
                self.biao.setItem(i, lie, xiang)
            elif xiang.text() != t:
                xiang.setText(t)
        # 注释行整行涂暗一档，一眼看出哪几条是藏起来的
        # （正在播的那一行归 shezhi_bofang_ms 管，别抢它的蓝底）
        if i != self._bofang_hang:
            di = self._hang_dise(i)
            # 从第 1 列起：第 0 列是序号列，它有自己那套黑底白字，别覆盖掉
            for lie in range(1, self.biao.columnCount()):
                xiang = self.biao.item(i, lie)
                if xiang is not None:
                    xiang.setBackground(di)

    def xuan_zhong_hang(self):
        """列表里现在选中了哪几行（从小到大排好）；一条没选是空表"""
        mo = self.biao.selectionModel()
        if mo is None:
            return []
        return sorted(int(i.row()) for i in mo.selectedRows())

    def shezhi_xuan_zhong_duo(self, hang):
        """外部（批量改完注释 / 别处选了）要把列表整批选中：只动高亮，编辑区不动"""
        hang = sorted(
            {int(x) for x in (hang or []) if 0 <= int(x) < len(self._zimu)}
        )
        mo = self.biao.model()
        if not hang or mo is None:
            return
        xuan = QtCore.QItemSelection()
        for i in hang:
            xuan.select(mo.index(i, 0), mo.index(i, self.LIE_WENBEN))
        self.biao.blockSignals(True)
        self.biao.selectionModel().select(
            xuan, QtCore.QItemSelectionModel.ClearAndSelect
        )
        self.biao.blockSignals(False)
        self._shuaxin_zhushi_kuang(hang)

    def _shuaxin_zhushi_kuang(self, hang):
        """多选时把「注释」勾选框按这一批的状态显示：全都已经是注释才勾上

        这样"再点一下"就是把这批全设成注释，不会反过来把它们全取消掉。
        """
        dou = all(
            bool((self._fujia[i] if i < len(self._fujia) else {}).get("zhushi"))
            for i in hang
        )
        gou = self.gongjulan.gou_zhushi
        jiu = gou.blockSignals(True)
        gou.setChecked(bool(dou))
        gou.blockSignals(jiu)

    def _xuan_zhong_bian(self, *_a):
        """列表里选中变了：往外同步

        只选一条 = 老规矩（时间轴选中它，播放头跳到它开头）；多选
        （Ctrl+A 全选 / Shift + 鼠标点击连选 / Ctrl + 鼠标点击点选）= 只把
        这一批行号给外面，不跳播放头 —— 批量设注释、批量删除都用它。
        """
        hang = self.xuan_zhong_hang()
        if len(hang) >= 2:
            self._shuaxin_zhushi_kuang(hang)
            self.xuan_zhong_duo.emit(hang)
            return
        if not hang:
            self.shezhi_xuan_zhong(-1)
            self.xuan_zhong.emit(-1)
            return
        hang = int(hang[0])
        if hang == self._xu:
            return          # 同一行里换列，不算又选了一次
        self.shezhi_xuan_zhong(hang)
        self.xuan_zhong.emit(hang)
        if self.zidong_tiao:
            # 「选中字幕时画面跟着跳」关掉时：只选中，播放头和画面都不动
            self.zimu_qi_tiao.emit(int(self._zimu[hang][0]))

    def shezhi_zidong_tiao(self, kai):
        """「选中字幕时画面跟着跳」开关（照 AEG）：关了 = 点列表只选中，画面不动"""
        self.zidong_tiao = bool(kai)

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
        self.zimu_qi_tiao.emit(int(self._zimu[xia][0]))

    def _wenben_bian(self):
        """编辑区里打字 -> 实时往后传（不重建列表，只改这一条）"""
        if self._tian or self._xu < 0:
            return
        self.wenben_gaile.emit(self._xu, self.kuang.wenben())

    def _bianji_wancheng(self):
        """点走 / 焦点离开编辑框 -> 这一条改完了"""
        if self._xu < 0:
            return
        self.bianji_wancheng.emit(self._xu, self.kuang.wenben())


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
