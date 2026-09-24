# -*- coding: utf-8 -*-
"""视频工作台的查找 / 替换 / 选择（照 Aegisub 那三个窗口）

Ctrl+R 选择、Ctrl+F 查找、Ctrl+H 替换。三个都是非模态窗口：开着照样能操作
视频工作台（播放、拖时间轴、改字幕都不耽误）。

匹配规则照 Aegisub 的 search_replace_engine.cpp：
    精确匹配     = 整栏内容跟查找目标完全一样
    包含         = 在栏里找子串
    正则表达式   = 按正则找（"区分大小写"不勾就忽略大小写）
    忽略特效标签 = 先在正文里找（跳过 {...} 那种标签），找到再映回原文，
                   替换时不会伤到标签

查找和替换是同一个窗口类（照 AEG 的 dialog_search_replace.cpp）：替换模式
多一行「替换为」和多两个按钮；打开一个就把另一个关掉。

窗口里的数据都是问宿主（视频工作台）要的，就这几个方法：
    ss_hang_men()                   整份字幕，一条一个字典
    ss_xuan_hang()                  列表里现在选中了哪几行
    ss_dangqian()                   现在编辑的是第几条（没选中给 -1）
    ss_she_xuan(hang)               把选中的设成这几行（列表 / 时间轴跟着高亮）
    ss_dingwei(xu, qi, zhi)         定位到第几条，文本栏还能圈出匹配的那一段
    ss_gai(jian, xiugai)            批量改某一栏：jian 见 LAN_JIAN，xiugai 是
                                    [(行号, 新内容), ...]
    ss_zhuangtai(wen)               底部状态栏说一句
    ss_tishi(biaoti, wen)           弹个提示框（只有"一条都没匹配上"这种才弹）

下面文件里那几个 _ys_* 样式函数是从 video_infer_panel 借来的（它们就是那个
模块里的模块级函数）：这样这几个窗口的配色跟工作台其它地方是同一套，以后改
主题色只改一处。
"""

import re

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from anylabeling.views.labeling.widgets.video_infer_panel import (
    _ys,
    _ys_ci_anniu,
    _ys_shuru,
    _ys_xuanxiang,
    _ys_zhu_anniu,
    buju_qsettings,
)


# =====================================================================
# 存在 gongzuotai.ini 里的配置项名（跟界面记忆那些放一起）
# =====================================================================
# 存进 gongzuotai.ini 的配置项。三个窗口的勾选/选项/历史各记各的：
# 键名 = 窗口前缀 + 下面的后半截（查找 sousuo/zhao_*、替换 sousuo/huan_*、
# 选择 sousuo/xuan_*）。只有「查找目标」的下拉历史是查找和替换公用一份。
QIAN_ZHUI = {
    "zhao": "sousuo/zhao_",
    "huan": "sousuo/huan_",
    "xuan": "sousuo/xuan_",
}
PEI_CHAZHAO = "sousuo/chazhao"      # 查找目标的历史（查找 / 替换公用这一份）
PEI_TIHUAN = "tihuan"               # 替换为的历史
PEI_XUAN_WEN = "wen"                # 「选择」窗口里那个文本框
PEI_QUFEN = "qufen"                 # 区分大小写
PEI_ZHENGZE = "zhengze"             # 使用正则表达式
PEI_HULVE_ZHUSHI = "hulve_zhushi"   # 忽略注释
PEI_HULVE_TEXIAO = "hulve_texiao"   # 忽略特效标签
PEI_LAN = "lan"                     # 在哪一栏里找：0 文本 1 样式 2 说话人
PEI_FANWEI = "fanwei"               # 0 所有行 / 1 所选行
PEI_XUAN_PIPEI = "pipei"            # 0 匹配项 / 1 不匹配
PEI_XUAN_MOSHI = "moshi"            # 0 精确匹配 / 1 包含 / 2 正则表达式
PEI_XUAN_DONGZUO = "dongzuo"        # 0 设为所选 / 1 加入 / 2 移出 / 3 取交集
PEI_XUAN_DUIHUA = "duihua"          # 「选择」：匹配对话行
PEI_XUAN_ZHUSHI = "zhushi"          # 「选择」：匹配注释行
LISHI_ZUIDA = 20                    # 查找 / 替换的历史最多记几条（多了挤掉最早的）

LAN_MING = ("文本", "样式", "说话人")
LAN_JIAN = ("wenben", "yang", "shuo")   # 对应宿主那边一条字幕的字段名


# =====================================================================
# 配置读写
# =====================================================================
def _du_zhi(jian, moren=None):
    """读一个配置值（读不出来就给默认）"""
    try:
        zhi = buju_qsettings().value(jian, moren)
    except Exception:  # noqa
        return moren
    return moren if zhi is None else zhi


def _du_kai(jian, moren=False):
    """读一个开关

    坑：QSettings 从 ini 读回来的布尔值是**字符串**（"true" / "false" /
    "1" / "0"），直接 bool("false") 会得 True —— 所以这儿自己认一遍。
    """
    zhi = _du_zhi(jian, moren)
    if isinstance(zhi, str):
        return zhi.strip().lower() in ("1", "true", "yes", "on")
    return bool(zhi)


def _cun_zhi(jian, zhi):
    """写一个配置值（写不进去就算了，不影响用）"""
    try:
        pei = buju_qsettings()
        pei.setValue(jian, zhi)
        pei.sync()
    except Exception:  # noqa
        pass


def _du_lishi(jian):
    """读历史列表（存的时候是一条一条写的，读回来可能是张表，也可能就一条）"""
    zhi = _du_zhi(jian, [])
    if zhi is None:
        return []
    if isinstance(zhi, str):
        return [zhi] if zhi else []
    try:
        return [str(x) for x in zhi if str(x)]
    except TypeError:
        return [str(zhi)]


def _jia_lishi(jian, wen):
    """把一个新词插到历史最前面（已有的先挪走，最多留 LISHI_ZUIDA 条）"""
    wen = str(wen or "")
    if not wen:
        return
    jiu = [x for x in _du_lishi(jian) if x != wen]
    _cun_zhi(jian, [wen] + jiu[: LISHI_ZUIDA - 1])


# =====================================================================
# 匹配（照 Aegisub 的 search_replace_engine.cpp）
# =====================================================================
def _bo_biaoqian(wenben):
    """把 {...} 花括号块剥掉，给出「剥完的正文 ↔ 原串下标」的对照表

    返回 (正文, dui_zhao)：dui_zhao[i] 是正文第 i 个字在原串里的下标。
    忽略特效标签时按正文找，找到的范围再映回原文，替换就不会伤到标签。
    """
    wen = str(wenben or "")
    zheng = []
    dui = []
    shen = 0
    for i, zi in enumerate(wen):
        if zi == "{":
            shen += 1
            continue
        if shen > 0:
            if zi == "}":
                shen -= 1
            continue
        zheng.append(zi)
        dui.append(i)
    return "".join(zheng), dui


def _zheng_dao_bo(dui, wei):
    """把原串里的下标 wei 换成剥完串里的下标（找不到就给末尾）"""
    for i, y in enumerate(dui):
        if y >= wei:
            return i
    return len(dui)


def zhao_pipei(wenben, chazhao, qufen=False, zhengze=False, jingque=False,
               hulve_texiao=False, congyi=0):
    """在 wenben 里从 congyi 往后找一个匹配；返回 (开始, 结束) 或 None

    范围一律是"原串里的下标"（跳过标签时会映回去）。
    正则写错了会抛 re.error，调用那边接住弹提示。
    能匹配空串的正则当没找到 —— 不然"全部替换"会原地死循环。
    """
    wen = str(wenben or "")
    zhao = str(chazhao or "")
    if zhao == "":
        return None
    ying = None
    if hulve_texiao:
        wen, ying = _bo_biaoqian(wen)
        congyi = _zheng_dao_bo(ying, congyi)
    if congyi > len(wen):
        return None

    if zhengze:
        biao = re.compile(zhao, 0 if qufen else re.IGNORECASE)
        m = biao.search(wen, congyi)
        if m is None or m.start() == m.end():
            return None
        qi, zhi = m.start(), m.end()
    elif jingque:
        # 精确匹配：整栏内容跟查找目标一样才算（从头比，不接在上次后头找）
        if congyi > 0:
            return None
        if qufen:
            if wen != zhao:
                return None
        elif wen.lower() != zhao.lower():
            return None
        qi, zhi = 0, len(wen)
    else:
        if qufen:
            k = wen.find(zhao, congyi)
        else:
            k = wen.lower().find(zhao.lower(), congyi)
        if k < 0:
            return None
        qi, zhi = k, k + len(zhao)

    if ying is not None:
        if qi >= len(ying):
            return None
        zhi = min(zhi, len(ying))
        if zhi <= qi:
            return None
        return ying[qi], ying[zhi - 1] + 1
    return qi, zhi


def huan_yiduan(wenben, qi, zhi, xin):
    """把 [qi, zhi) 这一段换成 xin"""
    wen = str(wenben or "")
    return wen[:qi] + str(xin) + wen[zhi:]


# =====================================================================
# 窗口通用样式
# =====================================================================
def _ys_kuang():
    """窗口底色 / 小字 / 分组框 / 下拉框（其余控件各自套 _ys_* 那几套）"""
    c = _ys()
    return f"""
    QDialog {{ background-color: {c['beijing']}; }}
    QLabel {{ color: {c['wenzi']}; font-size: 12px; }}
    QGroupBox {{
        color: {c['wenzi_ci']};
        border: 1px solid {c['biankuang']};
        border-radius: 6px;
        margin-top: 9px;
        padding: 9px 8px 7px 8px;
        font-size: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    QComboBox {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        border: 1px solid {c['biankuang_liang']};
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 12px;
        min-height: 22px;
    }}
    QComboBox QLineEdit {{
        background: transparent;
        color: {c['wenzi']};
        border: none;
        padding: 0;
        min-height: 0;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['beijing2']};
        color: {c['wenzi']};
        selection-background-color: {c['zhuse']};
        selection-color: #ffffff;
    }}
    QToolTip {{
        background-color: {c['wenzi']};
        color: {c['beijing2']};
        border: none;
        padding: 3px 6px;
    }}
    """


def _zuo_biaoqian(wenben):
    """窗口里的小标签"""
    ti = QtWidgets.QLabel(wenben)
    ti.setStyleSheet(f"color: {_ys()['wenzi']}; font-size: 12px;")
    return ti


def _ys_xuan_tishi():
    """选择窗口底下那行「选中了 N 行」（照 AEG 底部状态栏那句，挪进来）"""
    c = _ys()
    return f"color: {c['wenzi']}; font-size: 12px;"


def _danxuan_zu(mingzi, xiang_men, pai=1):
    """一个单选分组（照 AEG 的 wxRadioBox）：pai = 排几列"""
    he = QtWidgets.QGroupBox(mingzi)
    # 分组框自己那圈边线在 _ys_kuang 里，选项文字在 _ys_xuanxiang 里，
    # 两套都得带上（控件自己的样式表会盖掉窗口那份）
    he.setStyleSheet(_ys_kuang() + _ys_xuanxiang())
    wang = QtWidgets.QGridLayout(he)
    wang.setContentsMargins(8, 6, 8, 6)
    wang.setHorizontalSpacing(14)
    wang.setVerticalSpacing(4)
    an_men = []
    for i, xiang in enumerate(xiang_men):
        an = QtWidgets.QRadioButton(xiang)
        an_men.append(an)
        wang.addWidget(an, i // pai, i % pai)
    if an_men:
        an_men[0].setChecked(True)
    he.an_men = an_men
    return he


def _gou_zu(mingzi, xiang_men):
    """一个勾选分组"""
    he = QtWidgets.QGroupBox(mingzi)
    he.setStyleSheet(_ys_kuang() + _ys_xuanxiang())
    wang = QtWidgets.QHBoxLayout(he)
    wang.setContentsMargins(8, 6, 8, 6)
    wang.setSpacing(14)
    an_men = []
    for xiang in xiang_men:
        an = QtWidgets.QCheckBox(xiang)
        an_men.append(an)
        wang.addWidget(an)
    he.an_men = an_men
    return he


def _zhi_zhong(an_men):
    """单选组现在选的是第几个（一个都没选给 0）"""
    for i, an in enumerate(an_men):
        if an.isChecked():
            return i
    return 0


def _gua_chexiao(chuang):
    """给这三个窗口各挂一份 Ctrl+Z / Ctrl+Y，转发给视频工作台

    这三个窗口是独立顶层窗口，焦点落在它们身上时工作台那份快捷键不响应
    （窗口级快捷键只认自己这个窗口），所以在这儿补一份。撤的是工作台那边
    的字幕 —— 跟 AEG 一样，撤销栈是整个字幕的，不分窗口。
    焦点在本窗口输入框里、且那框里确实有字可撤时，让输入框自己撤。
    """
    for an, na in (("Ctrl+Z", "_chexiao"), ("Ctrl+Y", "_chongzuo")):
        jian = QtWidgets.QShortcut(QtGui.QKeySequence(an), chuang)
        jian.setContext(Qt.WindowShortcut)
        jian.activated.connect(
            lambda c=chuang, n=na: _zhuansong_chexiao(c, n)
        )


def _zhuansong_chexiao(chuang, na):
    """窗口里按了 Ctrl+Z / Ctrl+Y：该谁办谁办"""
    zhu = QtWidgets.QApplication.focusWidget()
    shuru = isinstance(
        zhu, (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
              QtWidgets.QPlainTextEdit),
    )
    if shuru and chuang.isAncestorOf(zhu):
        # 本窗口的输入框里还有字可撤 / 可重做：这一下给它
        if (zhu.isRedoAvailable() if na == "_chongzuo"
                else zhu.isUndoAvailable()):
            zhu.redo() if na == "_chongzuo" else zhu.undo()
            return
    zhuren = getattr(chuang, "_zhuren", None)
    if zhuren is not None:
        getattr(zhuren, na)()


# =====================================================================
# 查找 / 替换窗口（Ctrl+F / Ctrl+H）
# =====================================================================
class SuoSuoDialog(QtWidgets.QDialog):
    """查找（Ctrl+F）和替换（Ctrl+H）：同一个窗口，替换模式多一行「替换为」

    照 AEG：查找和替换共用一个类，互相打开时把另一个关掉（宿主那边管）。
    """

    def __init__(self, zhuren, huan=False):
        super().__init__(zhuren)
        self._zhuren = zhuren
        self._huan = bool(huan)
        # 自己的配置键前缀：查找和替换各存各的（改一个不影响另一个）
        self._q = QIAN_ZHUI["huan" if self._huan else "zhao"]
        self._pipei = None      # 上次命中的 (行号, 开始, 结束)；"替换下一个"靠它
        self._huan_le = 0       # 替换模式：这个窗口里已经换掉几处（底部提示行用）
        self._qian_pei = None   # 上一次算提示行时的查找条件（变了就把上面那个清零）
        self.setWindowTitle("替换" if self._huan else "查找")
        self.setModal(False)    # 非模态：开着照样能操作视频工作台
        # 三个窗口要能同时开着：给个最小化按钮，用不上的时候收走
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet(_ys_kuang())

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(12, 12, 12, 12)
        wai.setSpacing(8)

        ding = QtWidgets.QHBoxLayout()
        ding.setSpacing(12)
        zuo = QtWidgets.QVBoxLayout()
        zuo.setSpacing(8)

        # ---- 查找目标 / 替换为 ----
        ge = QtWidgets.QGridLayout()
        ge.setHorizontalSpacing(8)
        ge.setVerticalSpacing(6)
        ge.addWidget(_zuo_biaoqian("查找目标："), 0, 0)
        self.chazhao_kuang = self._jian_xialakuang(PEI_CHAZHAO)
        ge.addWidget(self.chazhao_kuang, 0, 1)
        self.tihuan_kuang = None
        if self._huan:
            ge.addWidget(_zuo_biaoqian("替换为："), 1, 0)
            self.tihuan_kuang = self._jian_xialakuang(self._q + PEI_TIHUAN)
            ge.addWidget(self.tihuan_kuang, 1, 1)
        zuo.addLayout(ge)

        # ---- 几个勾 ----
        gou = QtWidgets.QVBoxLayout()
        gou.setSpacing(4)
        self.gou_qufen = QtWidgets.QCheckBox("区分大小写")
        self.gou_zhengze = QtWidgets.QCheckBox("使用正则表达式")
        self.gou_zhushi = QtWidgets.QCheckBox("忽略注释")
        self.gou_texiao = QtWidgets.QCheckBox("忽略特效标签")
        self.gou_qufen.setChecked(_du_kai(self._q + PEI_QUFEN, False))
        self.gou_zhengze.setChecked(_du_kai(self._q + PEI_ZHENGZE, False))
        self.gou_zhushi.setChecked(_du_kai(self._q + PEI_HULVE_ZHUSHI, False))
        self.gou_texiao.setChecked(_du_kai(self._q + PEI_HULVE_TEXIAO, False))
        self.gou_qufen.setStyleSheet(_ys_xuanxiang())
        self.gou_zhengze.setStyleSheet(_ys_xuanxiang())
        self.gou_zhushi.setStyleSheet(_ys_xuanxiang())
        self.gou_texiao.setStyleSheet(_ys_xuanxiang())
        for an in (self.gou_qufen, self.gou_zhengze,
                   self.gou_zhushi, self.gou_texiao):
            gou.addWidget(an)
        zuo.addLayout(gou)
        zuo.addStretch(1)

        # ---- 在下列栏中搜索 ----
        self.he_lan = _danxuan_zu(
            "在下列栏中搜索", LAN_MING, pai=3
        )
        zuo.addWidget(self.he_lan)

        # ---- 限制 ----
        self.he_fanwei = _danxuan_zu("限制", ("所有行", "所选行"), pai=2)
        zuo.addWidget(self.he_fanwei)

        # ---- 右边那排按钮 ----
        you = QtWidgets.QVBoxLayout()
        you.setSpacing(6)
        self.an_zhao = QtWidgets.QPushButton("查找下一个")
        self.an_zhao.setStyleSheet(_ys_zhu_anniu())
        self.an_zhao.clicked.connect(self._zhao_xiayige)
        you.addWidget(self.an_zhao)
        if self._huan:
            self.an_huan = QtWidgets.QPushButton("替换下一个")
            self.an_huan.setStyleSheet(_ys_ci_anniu())
            self.an_huan.clicked.connect(self._huan_xiayige)
            you.addWidget(self.an_huan)
            self.an_quanbu = QtWidgets.QPushButton("全部替换")
            self.an_quanbu.setStyleSheet(_ys_ci_anniu())
            self.an_quanbu.clicked.connect(self._quanbu_huan)
            you.addWidget(self.an_quanbu)
        self.an_guan = QtWidgets.QPushButton("关闭")
        self.an_guan.setStyleSheet(_ys_ci_anniu())
        self.an_guan.clicked.connect(self.close)
        you.addWidget(self.an_guan)
        you.addStretch(1)

        ding.addLayout(zuo, 1)
        ding.addLayout(you)
        wai.addLayout(ding)

        # ---- 底部提示行：找到几条 / 当前第几（照 AEG 底部状态栏那句，挪进来）----
        self.ti_shi = QtWidgets.QLabel("")
        self.ti_shi.setStyleSheet(_ys_xuan_tishi())
        wai.addWidget(self.ti_shi)

        # ---- 取值 / 还原 ----
        lan = int(_du_zhi(self._q + PEI_LAN, 0) or 0)
        # 旧配置里存过"特效"（3）那栏，现在没这栏了：越界就不勾，留默认的「文本」
        if 0 <= lan < len(self.he_lan.an_men):
            self.he_lan.an_men[lan].setChecked(True)
        fan = int(_du_zhi(self._q + PEI_FANWEI, 0) or 0)
        if 0 <= fan < len(self.he_fanwei.an_men):
            self.he_fanwei.an_men[fan].setChecked(True)

        # 输入框里回车 = 往下找一个 / 替换这一处（跟 AEG 一样）
        yiyang = self.chazhao_kuang.lineEdit()
        if yiyang is not None:
            yiyang.returnPressed.connect(self._zhao_xiayige)
        if self._huan and self.tihuan_kuang is not None:
            yiyang = self.tihuan_kuang.lineEdit()
            if yiyang is not None:
                yiyang.returnPressed.connect(self._huan_xiayige)

        self._shuaxin_tishi()
        _gua_chexiao(self)      # 焦点在这个窗口里也能 Ctrl+Z 撤工作台的字幕

    # ----------------------------------------------------------------
    def yi_tihuan(self):
        """这个窗口是替换模式还是查找模式（宿主靠它决定要不要换窗口）"""
        return self._huan

    def closeEvent(self, event):
        """关窗口时把这一套设置和输入框里的词再存一遍

        光改勾选、没按按钮就把窗口关了也不丢；顺手把现在框里的词记进历史。
        """
        pei = self._qu_peizhi()
        self._ji_peizhi(pei)
        _jia_lishi(PEI_CHAZHAO, pei["chazhao"])
        _jia_lishi(self._q + PEI_TIHUAN, pei["tihuan"])
        super().closeEvent(event)

    def qing_jiaodian(self):
        """从快捷键叫回来时：焦点落到「查找目标」上并全选（照 AEG）"""
        self.chazhao_kuang.setFocus(Qt.OtherFocusReason)
        yiyang = self.chazhao_kuang.lineEdit()
        if yiyang is not None:
            yiyang.selectAll()

    def _jian_xialakuang(self, jian):
        """一个能打字、带历史记录的下拉框"""
        kuang = QtWidgets.QComboBox()
        kuang.setEditable(True)
        kuang.setMinimumWidth(320)
        kuang.setStyleSheet(_ys_kuang())
        kuang.addItems(_du_lishi(jian))
        return kuang

    def _shuaxin_lishi(self):
        """执行过一次以后：把刚用的词记进历史（顺手更新下拉里的候选）"""
        for kuang, jian in ((self.chazhao_kuang, PEI_CHAZHAO),
                            (self.tihuan_kuang, self._q + PEI_TIHUAN)):
            if kuang is None:
                continue
            jiu = str(kuang.currentText() or "")
            kuang.blockSignals(True)
            kuang.clear()
            kuang.addItems(_du_lishi(jian))
            kuang.setCurrentText(jiu)
            kuang.blockSignals(False)

    def _qu_peizhi(self):
        """把界面上现在这一套收成一个字典"""
        return {
            "chazhao": str(self.chazhao_kuang.currentText() or ""),
            "tihuan": ("" if self.tihuan_kuang is None
                       else str(self.tihuan_kuang.currentText() or "")),
            "qufen": self.gou_qufen.isChecked(),
            "zhengze": self.gou_zhengze.isChecked(),
            "hulve_zhushi": self.gou_zhushi.isChecked(),
            "hulve_texiao": self.gou_texiao.isChecked(),
            "lan": _zhi_zhong(self.he_lan.an_men),
            "fanwei": _zhi_zhong(self.he_fanwei.an_men),
        }

    def _ji_peizhi(self, pei):
        """下次开窗口还是这一套"""
        _cun_zhi(self._q + PEI_QUFEN, pei["qufen"])
        _cun_zhi(self._q + PEI_ZHENGZE, pei["zhengze"])
        _cun_zhi(self._q + PEI_HULVE_ZHUSHI, pei["hulve_zhushi"])
        _cun_zhi(self._q + PEI_HULVE_TEXIAO, pei["hulve_texiao"])
        _cun_zhi(self._q + PEI_LAN, pei["lan"])
        _cun_zhi(self._q + PEI_FANWEI, pei["fanwei"])

    def _jian_bi_you(self, pei):
        """按"限制 = 所选行"算出来的那一批行号；不限就是 None"""
        if pei["fanwei"] != 1:
            return None
        xuan = self._zhuren.ss_xuan_hang()
        if len(xuan) < 2:
            return None         # 只选中一条不算"所选行"（照 AEG）
        return set(int(x) for x in xuan)

    def _suan_pipei(self, pei):
        """照现在这套条件把整份字幕数一遍，返回命中的行号表（按顺序）"""
        hang = self._zhuren.ss_hang_men()
        jian = LAN_JIAN[pei["lan"]]
        suoding = self._jian_bi_you(pei)
        hao = []
        for i, yi in enumerate(hang):
            if suoding is not None and i not in suoding:
                continue
            if pei["hulve_zhushi"] and yi["zhushi"]:
                continue
            m = zhao_pipei(
                yi[jian], pei["chazhao"],
                qufen=pei["qufen"], zhengze=pei["zhengze"],
                hulve_texiao=pei["hulve_texiao"], congyi=0,
            )
            if m is not None:
                hao.append(i)
        return hao

    def _shuaxin_tishi(self, dangqian=None):
        """底部提示行：找到几条 / 当前第几（替换模式再多一句已替换几处）"""
        pei = self._qu_peizhi()
        # 查找条件一变，已替换那个计数就从头算（换了个词，攒的数没意义了）
        zhi = (pei["chazhao"], pei["qufen"], pei["zhengze"],
               pei["hulve_zhushi"], pei["hulve_texiao"],
               pei["lan"], pei["fanwei"])
        if zhi != self._qian_pei:
            self._qian_pei = zhi
            self._huan_le = 0
        if not pei["chazhao"]:
            self.ti_shi.setText("")
            return
        try:
            hao = self._suan_pipei(pei)
        except re.error:
            self.ti_shi.setText("正则表达式写错了")
            return
        if dangqian is None:
            dangqian = self._zhuren.ss_dangqian()
        di = 0
        if dangqian is not None and int(dangqian) in hao:
            di = hao.index(int(dangqian)) + 1
        shuo = f"找到 {len(hao)} 条"
        if self._huan:
            shuo += f"，已替换 {self._huan_le} 处"
        elif di:
            shuo += f"，当前第 {di} 条"
        self.ti_shi.setText(shuo)

    # ----------------------------------------------------------------
    def _zhao_yige(self):
        """往下找一个匹配；返回 (行号, 开始, 结束) 或 None

        从"现在编辑的那一条"接着上次的位置往下找，绕一圈回到它自己就停
        （照 AEG 的 circular_next）。
        """
        hang = self._zhuren.ss_hang_men()
        if not hang:
            return None
        pei = self._qu_peizhi()
        if not pei["chazhao"]:
            return None
        jian = LAN_JIAN[pei["lan"]]
        suoding = self._jian_bi_you(pei)

        kai = int(self._zhuren.ss_dangqian())
        if not (0 <= kai < len(hang)):
            kai = 0
        qi = 0
        # 光标还停在上次命中的那一条上，就接着它后面往下找；
        # 用户中途自己点了别的行，就从那一行重新找（照 AEG 的"从选区尾巴往后"）
        if self._pipei is not None and int(self._pipei[0]) == kai:
            qi = int(self._pipei[2])

        for bu in range(len(hang)):
            i = (kai + bu) % len(hang)
            if suoding is not None and i not in suoding:
                continue
            if pei["hulve_zhushi"] and hang[i]["zhushi"]:
                continue
            m = zhao_pipei(
                hang[i][jian], pei["chazhao"],
                qufen=pei["qufen"], zhengze=pei["zhengze"],
                hulve_texiao=pei["hulve_texiao"],
                congyi=(qi if i == kai else 0),
            )
            if m is not None:
                return (i, m[0], m[1])
        return None

    def _dingwei(self, jieguo):
        """命中：选中那一条，文本栏的话把匹配的那一段圈出来"""
        xu, qi, zhi = jieguo
        self._pipei = (int(xu), int(qi), int(zhi))
        pei = self._qu_peizhi()
        if LAN_JIAN[pei["lan"]] == "wenben":
            self._zhuren.ss_dingwei(int(xu), int(qi), int(zhi))
        else:
            self._zhuren.ss_dingwei(int(xu))

    def _shou_shang_mingzhong(self, pei):
        """手上是不是正握着一个匹配（上次查找 / 替换定位到的那一处）

        按「替换下一个」时，画面上如果正显示着一个匹配，就该换它 ——
        不能再跑去查下一个（那样「替换下一个」就变成「查找下一个」了）。
        得拿当前文本重新验一遍：那一处可能已经被改掉，那就不能瞎换。
        """
        if self._pipei is None:
            return None
        if pei["hulve_texiao"]:
            return None         # 带 {...} 标签时下标会漂，干脆重新找
        xu, qi, zhi = (int(x) for x in self._pipei)
        hang = self._zhuren.ss_hang_men()
        if not (0 <= xu < len(hang)):
            return None
        wen = str(hang[xu][LAN_JIAN[pei["lan"]]] or "")
        if not (0 <= qi < zhi <= len(wen)):
            return None
        yi = wen[qi:zhi]
        if pei["zhengze"]:
            biao = re.compile(
                pei["chazhao"], 0 if pei["qufen"] else re.IGNORECASE
            )
            if biao.fullmatch(yi) is None:
                return None
        elif pei["qufen"]:
            if yi != pei["chazhao"]:
                return None
        elif yi.lower() != pei["chazhao"].lower():
            return None
        return (xu, qi, zhi)

    def _zhao_xiayige(self):
        pei = self._qu_peizhi()
        if not pei["chazhao"]:
            self._zhuren.ss_zhuangtai("先写上要查找的内容")
            return
        try:
            jieguo = self._zhao_yige()
        except re.error as cuowu:
            self._zhuren.ss_tishi("正则表达式写错了", str(cuowu))
            return
        if jieguo is None:
            self._zhuren.ss_zhuangtai(f"没找到「{pei['chazhao']}」")
            return
        self._dingwei(jieguo)
        self._ji_peizhi(pei)
        _jia_lishi(PEI_CHAZHAO, pei["chazhao"])
        self._shuaxin_lishi()
        self._shuaxin_tishi(jieguo[0])
        self._zhuren.ss_zhuangtai(
            f"第 {jieguo[0] + 1} 条：找到「{pei['chazhao']}」"
        )

    # ----------------------------------------------------------------
    def _huan_yichu(self, jieguo, pei):
        """把命中的这一处换掉；返回换完之后这一段的结尾位置"""
        xu, qi, zhi = jieguo
        hang = self._zhuren.ss_hang_men()
        if not (0 <= xu < len(hang)):
            return qi
        jian = LAN_JIAN[pei["lan"]]
        jiu = str(hang[xu][jian] or "")
        if pei["zhengze"]:
            biao = re.compile(pei["chazhao"], 0 if pei["qufen"] else re.IGNORECASE)
            xin = biao.sub(pei["tihuan"], jiu[qi:zhi], count=1)
        else:
            xin = pei["tihuan"]
        xin_wen = huan_yiduan(jiu, qi, zhi, xin)
        self._zhuren.ss_gai(jian, [(xu, xin_wen)])
        return qi + len(xin)

    def _huan_xiayige(self):
        """替换一个：手上握着匹配就换它，否则先找一个再换 —— 点一下换一处"""
        pei = self._qu_peizhi()
        if not pei["chazhao"]:
            self._zhuren.ss_zhuangtai("先写上要查找的内容")
            return
        try:
            jieguo = self._shou_shang_mingzhong(pei)
            if jieguo is None:
                jieguo = self._zhao_yige()
        except re.error as cuowu:
            self._zhuren.ss_tishi("正则表达式写错了", str(cuowu))
            return
        if jieguo is None:
            self._zhuren.ss_zhuangtai(f"没找到「{pei['chazhao']}」")
            return
        jiewei = self._huan_yichu(jieguo, pei)
        # 记住换完的位置：再按一次就从这儿往后接着换
        self._pipei = (jieguo[0], jieguo[1], jiewei)
        if LAN_JIAN[pei["lan"]] == "wenben":
            self._zhuren.ss_dingwei(jieguo[0], jieguo[1], jiewei)
        else:
            self._zhuren.ss_dingwei(jieguo[0])
        pei = self._qu_peizhi()
        self._ji_peizhi(pei)
        _jia_lishi(PEI_CHAZHAO, pei["chazhao"])
        _jia_lishi(self._q + PEI_TIHUAN, pei["tihuan"])
        self._shuaxin_lishi()
        self._huan_le += 1
        self._shuaxin_tishi(jieguo[0])
        self._zhuren.ss_zhuangtai(
            f"第 {jieguo[0] + 1} 条：替换了「{pei['chazhao']}」"
        )

    def _quanbu_huan(self):
        pei = self._qu_peizhi()
        if not pei["chazhao"]:
            self._zhuren.ss_zhuangtai("先写上要查找的内容")
            return
        hang = self._zhuren.ss_hang_men()
        if not hang:
            self._zhuren.ss_tishi("替换", "还没有字幕")
            return
        jian = LAN_JIAN[pei["lan"]]
        suoding = self._jian_bi_you(pei)
        xiugai = []
        zong = 0
        try:
            for i, yi in enumerate(hang):
                if suoding is not None and i not in suoding:
                    continue
                if pei["hulve_zhushi"] and yi["zhushi"]:
                    continue
                jiu = str(yi[jian] or "")
                if pei["zhengze"]:
                    biao = re.compile(
                        pei["chazhao"], 0 if pei["qufen"] else re.IGNORECASE
                    )
                    xin, ci = biao.subn(pei["tihuan"], jiu)
                else:
                    xin = jiu
                    ci = 0
                    wei = 0
                    while True:
                        m = zhao_pipei(
                            xin, pei["chazhao"], qufen=pei["qufen"],
                            hulve_texiao=pei["hulve_texiao"], congyi=wei,
                        )
                        if m is None:
                            break
                        xin = huan_yiduan(xin, m[0], m[1], pei["tihuan"])
                        wei = m[0] + len(pei["tihuan"])
                        ci += 1
                if ci > 0 and xin != jiu:
                    xiugai.append((i, xin))
                    zong += ci
        except re.error as cuowu:
            self._zhuren.ss_tishi("正则表达式写错了", str(cuowu))
            return
        if not xiugai:
            self._shuaxin_tishi()
            self._zhuren.ss_tishi("替换", "没有找到匹配")
            return
        self._zhuren.ss_gai(jian, xiugai)
        self._ji_peizhi(pei)
        _jia_lishi(PEI_CHAZHAO, pei["chazhao"])
        _jia_lishi(self._q + PEI_TIHUAN, pei["tihuan"])
        self._shuaxin_lishi()
        self._pipei = None
        self._huan_le += zong
        self._shuaxin_tishi()
        self._zhuren.ss_tishi(
            "替换", f"换了 {zong} 处，动了 {len(xiugai)} 条字幕"
        )


# =====================================================================
# 选择窗口（Ctrl+R）
# =====================================================================
class XuanZeDialog(QtWidgets.QDialog):
    """选择（Ctrl+R）：按内容挑一批字幕，设为所选 / 加入 / 移出 / 取交集

    照 AEG 的 dialog_selection.cpp。挑出来的那批直接反映到字幕列表和时间轴
    的高亮上（就是 Ctrl+A / Shift 连选那种多选），不改任何字幕内容。
    """

    def __init__(self, zhuren):
        super().__init__(zhuren)
        self._zhuren = zhuren
        # 自己的配置键前缀：三个窗口的勾选/选项各存各的
        self._x = QIAN_ZHUI["xuan"]
        self.setWindowTitle("选择")
        self.setModal(False)
        # 三个窗口要能同时开着：给个最小化按钮，用不上的时候收走
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        base = _du_zhi(self._x + PEI_LAN, 0)
        self.setStyleSheet(_ys_kuang())

        wai = QtWidgets.QVBoxLayout(self)
        wai.setContentsMargins(12, 12, 12, 12)
        wai.setSpacing(8)

        # ---- 匹配 ----
        he_pipei = QtWidgets.QGroupBox("匹配")
        he_pipei.setStyleSheet(_ys_kuang() + _ys_xuanxiang())
        wang = QtWidgets.QVBoxLayout(he_pipei)
        wang.setContentsMargins(8, 6, 8, 6)
        wang.setSpacing(6)
        hang = QtWidgets.QHBoxLayout()
        hang.setSpacing(14)
        self.an_pipei = QtWidgets.QRadioButton("匹配项")
        self.an_bu_pipei = QtWidgets.QRadioButton("不匹配")
        self.gou_qufen = QtWidgets.QCheckBox("区分大小写")
        hang.addWidget(self.an_pipei)
        hang.addWidget(self.an_bu_pipei)
        hang.addWidget(self.gou_qufen)
        hang.addStretch(1)
        wang.addLayout(hang)
        self.shuru_wen = QtWidgets.QLineEdit(
            str(_du_zhi(self._x + PEI_XUAN_WEN, "") or "")
        )
        self.shuru_wen.setStyleSheet(_ys_shuru())
        self.shuru_wen.setPlaceholderText("要挑什么样的字幕？")
        wang.addWidget(self.shuru_wen)
        wai.addWidget(he_pipei)

        # ---- 模式 / 栏 / 对话注释 / 动作 ----
        self.he_moshi = _danxuan_zu(
            "模式", ("精确匹配", "包含", "正则表达式匹配"), pai=1
        )
        wai.addWidget(self.he_moshi)
        self.he_lan = _danxuan_zu("在下列栏中搜索", LAN_MING, pai=3)
        wai.addWidget(self.he_lan)
        self.he_leixing = _gou_zu("匹配对话 / 注释", ("对话", "注释"))
        wai.addWidget(self.he_leixing)
        self.he_dongzuo = _danxuan_zu(
            "动作",
            ("设为所选", "加入所选", "移出所选", "选中与所选之交集"),
            pai=1,
        )
        wai.addWidget(self.he_dongzuo)

        # ---- 提示行（按钮上面）：现在选中了几行 ----
        self.ti_shi = QtWidgets.QLabel("")
        self.ti_shi.setStyleSheet(_ys_xuan_tishi())
        wai.addWidget(self.ti_shi)

        # ---- 按钮 ----
        an = QtWidgets.QHBoxLayout()
        an.setSpacing(8)
        an.addStretch(1)
        self.an_queding = QtWidgets.QPushButton("确定")
        self.an_queding.setStyleSheet(_ys_zhu_anniu())
        self.an_queding.clicked.connect(lambda: self._zhixing(guan=True))
        self.an_yingyong = QtWidgets.QPushButton("应用")
        self.an_yingyong.setStyleSheet(_ys_ci_anniu())
        self.an_yingyong.clicked.connect(lambda: self._zhixing(guan=False))
        self.an_guan = QtWidgets.QPushButton("关闭")
        self.an_guan.setStyleSheet(_ys_ci_anniu())
        self.an_guan.clicked.connect(self.close)
        an.addWidget(self.an_yingyong)
        an.addWidget(self.an_queding)
        an.addWidget(self.an_guan)
        wai.addLayout(an)

        # ---- 还原上次那一套 ----
        self.an_bu_pipei.setChecked(bool(int(_du_zhi(self._x + PEI_XUAN_PIPEI, 0) or 0)))
        self.an_pipei.setChecked(not self.an_bu_pipei.isChecked())
        self.gou_qufen.setChecked(_du_kai(self._x + PEI_QUFEN, False))
        moshi = int(_du_zhi(self._x + PEI_XUAN_MOSHI, 1) or 0)
        if 0 <= moshi < len(self.he_moshi.an_men):
            self.he_moshi.an_men[moshi].setChecked(True)
        if 0 <= int(base or 0) < len(self.he_lan.an_men):
            # 越界（旧配置里存过"特效"）就不勾，留默认的「文本」
            self.he_lan.an_men[int(base or 0)].setChecked(True)
        dong = int(_du_zhi(self._x + PEI_XUAN_DONGZUO, 0) or 0)
        if 0 <= dong < len(self.he_dongzuo.an_men):
            self.he_dongzuo.an_men[dong].setChecked(True)
        duihua = _du_kai(self._x + PEI_XUAN_DUIHUA, True)
        zhushi = _du_kai(self._x + PEI_XUAN_ZHUSHI, True)
        if not duihua and not zhushi:
            duihua = True        # 头一回 / 存坏了：至少留一个
        self.he_leixing.an_men[0].setChecked(duihua)
        self.he_leixing.an_men[1].setChecked(zhushi)
        # 两个都不勾就勾回另一个（照 AEG 的 OnDialogueCheckbox）
        for i, an2 in enumerate(self.he_leixing.an_men):
            an2.toggled.connect(
                lambda _kai, wo=i: self._kan_leixing(wo)
            )

        self.shuru_wen.returnPressed.connect(
            lambda: self._zhixing(guan=False)
        )

        self._shuaxin_tishi()
        _gua_chexiao(self)      # 焦点在这个窗口里也能 Ctrl+Z 撤工作台的字幕

    def closeEvent(self, event):
        """关窗口时也把这一套存下来（只改了勾选、没按按钮也不丢）"""
        self._ji_peizhi(self._qu_peizhi())
        super().closeEvent(event)

    def _shuaxin_tishi(self):
        """提示行：现在选中了几行"""
        try:
            shu = len(self._zhuren.ss_xuan_hang())
        except Exception:  # noqa
            shu = 0
        self.ti_shi.setText(f"选中了 {shu} 行")

    def _kan_leixing(self, wo):
        """对话 / 注释两个勾：两个都不勾就把刚点掉的那个再勾回来"""
        if all(not an.isChecked() for an in self.he_leixing.an_men):
            self.he_leixing.an_men[wo].setChecked(True)

    def qing_jiaodian(self):
        self.shuru_wen.setFocus(Qt.OtherFocusReason)
        self.shuru_wen.selectAll()

    def _qu_peizhi(self):
        moshi = _zhi_zhong(self.he_moshi.an_men)
        return {
            "wen": str(self.shuru_wen.text() or ""),
            "fan": self.an_bu_pipei.isChecked(),
            "qufen": self.gou_qufen.isChecked(),
            "moshi": moshi,
            "lan": _zhi_zhong(self.he_lan.an_men),
            "duihua": self.he_leixing.an_men[0].isChecked(),
            "zhushi": self.he_leixing.an_men[1].isChecked(),
            "dongzuo": _zhi_zhong(self.he_dongzuo.an_men),
        }

    def _ji_peizhi(self, pei):
        _cun_zhi(self._x + PEI_XUAN_WEN, pei["wen"])
        _cun_zhi(self._x + PEI_XUAN_PIPEI, 1 if pei["fan"] else 0)
        _cun_zhi(self._x + PEI_QUFEN, pei["qufen"])
        _cun_zhi(self._x + PEI_XUAN_MOSHI, pei["moshi"])
        _cun_zhi(self._x + PEI_LAN, pei["lan"])
        _cun_zhi(self._x + PEI_XUAN_DONGZUO, pei["dongzuo"])
        _cun_zhi(self._x + PEI_XUAN_DUIHUA, pei["duihua"])
        _cun_zhi(self._x + PEI_XUAN_ZHUSHI, pei["zhushi"])

    def _zhixing(self, guan=False):
        """按现在这一套挑一批字幕，再把选择按"动作"改掉"""
        pei = self._qu_peizhi()
        hang = self._zhuren.ss_hang_men()
        jian = LAN_JIAN[pei["lan"]]
        pipei = set()
        try:
            for i, yi in enumerate(hang):
                if yi["zhushi"] and not pei["zhushi"]:
                    continue
                if (not yi["zhushi"]) and not pei["duihua"]:
                    continue
                m = zhao_pipei(
                    yi[jian], pei["wen"], qufen=pei["qufen"],
                    zhengze=pei["moshi"] == 2, jingque=pei["moshi"] == 0,
                    hulve_texiao=False, congyi=0,
                )
                if bool(pei["fan"]) != bool(m):
                    pipei.add(i)
        except re.error as cuowu:
            self._zhuren.ss_tishi("正则表达式写错了", str(cuowu))
            return

        jiu = set(int(x) for x in self._zhuren.ss_xuan_hang())
        if pei["dongzuo"] == 0:
            xin = set(pipei)
        elif pei["dongzuo"] == 1:
            xin = jiu | pipei
        elif pei["dongzuo"] == 2:
            xin = jiu - pipei
        else:
            xin = jiu & pipei

        self._zhuren.ss_she_xuan(sorted(xin))
        self._ji_peizhi(pei)
        self.ti_shi.setText(f"选中了 {len(xin)} 行")
        # 一条都没选就只把提示行改掉，不弹框也不关窗口（照用户改的，别学 AEG 那套）
        if guan and xin:
            self.close()
