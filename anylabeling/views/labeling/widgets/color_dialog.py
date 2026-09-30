from PyQt5 import QtWidgets


class ColorDialog(QtWidgets.QColorDialog):
    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        self.setOption(QtWidgets.QColorDialog.ShowAlphaChannel)
        # The Mac native dialog does not support our restore button.
        self.setOption(QtWidgets.QColorDialog.DontUseNativeDialog)
        # Add a restore defaults button.
        # The default is set at invocation time, so that it
        # works across dialogs for different elements.
        self.default = None
        self.bb = self.layout().itemAt(1).widget()
        self.bb.addButton(QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.bb.clicked.connect(self.check_restore)

    def get_color(self, value=None, title=None, default=None):
        self.default = default
        if title:
            self.setWindowTitle(title)
        if value:
            self.setCurrentColor(value)
        return self.currentColor() if self.exec_() else None

    def check_restore(self, button):
        if (
            self.bb.buttonRole(button) & QtWidgets.QDialogButtonBox.ResetRole
            and self.default
        ):
            self.setCurrentColor(self.default)


# ===========================================================================
# 下面这个是照 Aegisub 的调色板复刻的（AEG 源码 src/dialog_colorpicker.cpp）
# ===========================================================================
#
# 为什么不用上面那个 QColorDialog：Windows 上 Qt 的调色板默认走**系统原生**那
# 一个，原生那套没有 alpha（QColorDialog.ShowAlphaChannel 在原生对话框上是空转
# 的，只有加了 DontUseNativeDialog 才生 alpha 那一栏），而字幕颜色（样式里那四个
# 色、行内 \c \3c）恰恰要调透明度。AEG 用的是自己画的调色板，所以照它复刻一份：
#
#    左边  色彩光谱：256×256 的色板 + 一条色条 + 一条透明度条；五种光谱模式
#           RGB/红、RGB/绿、RGB/蓝、HSL/亮度、HSV/色相（照 AEG 的 Make*Spectrum）
#    右上  RGB色彩：红/绿/蓝 三个数值框 + ASS(&HBBGGRR&) + HTML(#RRGGBB) + 透明度
#    右中  HSL色彩 / HSV色彩 各三个数值框
#    右下  取色器（屏幕放大镜，照着屏幕上的像素取色）+ 最近用过的颜色（8×4 格）
#    底下  确定 / 取消
#
# 口径全照 AEG：
#   * h/s/l/v 都是 0-255（不是 0-359；AEG 的 colorspace.cpp 就是这么算的）
#   * 透明度 0 = 不透明、255 = 全透 —— AEG 那个"Alpha"字段、ASS 里 &HAABBGGRR
#     的 AA 都是这个口径（所以不透明的黑显示成"透明度 0"，跟 AEG 一模一样）

import numpy as np
from PyQt5 import QtCore, QtGui

try:
    from anylabeling.views.labeling.logger import logger
except Exception:                # noqa
    logger = None


def _ji_ri_zhi(shi, cuowu):
    """控件里出了意外别把整个软件带崩（PyQt5 里槽/事件里抛异常 = Qt FATAL 直接退）

    正常情况不该走到这儿；真走到了记一条日志，界面继续能用。
    """
    if logger is not None:
        try:
            logger.error("调色板：%s 出错：%s" % (shi, cuowu))
            return
        except Exception:        # noqa
            pass
    try:
        import traceback
        traceback.print_exc()
    except Exception:            # noqa
        pass

# 最近用过的颜色 / 上次用的光谱模式，存进 gongzuotai.ini 的 [tiaoseban] 子项里
# （照 AEG 的 OPT_GET("Tool/Colour Picker")）。**不另开配置文件**：调色板本来就
# 是视频工作台的一部分，工作台的记忆全在那个 ini 里，往那儿写就完事了 ——
# 也不许往 C 盘用户目录写（2026-10-01 用户为这个骂过一次）。
_TIAOSE_ZU = "tiaoseban"    # gongzuotai.ini 里的子项名
_TIAOSE_MOSHI = "moshi"     # 上次用的光谱模式（0-4）
_TIAOSE_ZUIJIN = "zuijin"   # 最近用过的颜色（#RRGGBB 一串）


def _tiaose_pei():
    """拿到工作台那个 QSettings（gongzuotai.ini）

    懒导入 video_infer_panel：color_dialog 是被它 import 的，顶上一 import 就转圈。
    """
    from anylabeling.views.labeling.widgets.video_infer_panel import (
        buju_qsettings,
    )

    return buju_qsettings()


_GUANGPU_MOSHI = ("RGB/红", "RGB/绿", "RGB/蓝", "HSL/亮度", "HSV/色相")
_SEPU_BIAN = 256            # 色板边长
_SE_TIAO_KUAN = 10          # 色条宽度（AEG 的 slider_width）
_ALPHA_GESHI = 5            # 透明度条一格多少像素（AEG 的 alpha_box_size）
_DENGDA_BAN = 3             # 小窗抓光标周围 ±3 像素（7×7，AEG 的 resx/resy）
_DENGDA_BEI = 8             # 放大倍数（AEG 的 magnification）
_DENGDA_XIAO = 7 * 8        # 小窗里那张图的边长 = 7×8 = 56（照 AEG 的 SetClientSize）
_DENGDA_BIAN = 1            # 边框占 1 像素（窗口边框画在控件里面，图从这之后开始）
_DENGDA_KUANG = _DENGDA_XIAO + _DENGDA_BIAN * 2     # 控件本身的大小 = 58×58
_ZUIJIN_LIE = 8             # 最近颜色格子：8 列（AEG 的 ColorPickerRecent(…, 8, 4, 16)）
_ZUIJIN_HANG = 4            # 4 行
_ZUIJIN_GE = 16             # 格子边长
_SHURU_KUAN = 76            # 数值框 / ASS / HTML 那几栏的宽度（AEG 按 "&H10117B&" 量的）
# 出厂先塞满一格 32 个：第一排照 AEG 出厂那八个（default_config.json 的
# Tool/Colour Picker/Recent Colours：黑 红 黄 绿 青 蓝 品 白），后面三排补上
# 常见的字幕色（灰阶 / 米黄肤色 / 深色 / 淡色），省得开出来是一片空的。
# 之后随用随换，规矩跟 AEG 一样：新颜色插到最前、挤掉最后一个；已经在格子里
# 的挪到最前（见 _ZuiJinSe.jia）。
_ZUIJIN_MOREN = (
    # 第一排：AEG 出厂那八个，位置照它的来
    "#000000", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF",
    "#FF00FF", "#FFFFFF",
    # 第二排：灰阶 + 米黄/肤色/棕
    "#404040", "#808080", "#C0C0C0", "#F5F5F5", "#FFF8E7", "#FFE4B5",
    "#D2B48C", "#8B4513",
    # 第三排：深色 / 饱和色
    "#800000", "#808000", "#008000", "#008080", "#000080", "#800080",
    "#FF8000", "#FF0080",
    # 第四排：淡色 / 粉紫 / 灰蓝
    "#FFC0CB", "#C8A2C8", "#B0E0E6", "#98FB98", "#FFFFE0", "#F0E68C",
    "#708090", "#2F4F4F",
)


# ---- 颜色换算：全部照抄 Aegisub 的 src/colorspace.cpp（h/s/l/v 0-255）----
def _qie(zhi):
    return 0 if zhi < 0 else (255 if zhi > 255 else int(zhi))


def hsl_dao_rgb(H, S, L):
    """AEG 的 hsl_to_rgb（连它那几个特例点一起搬：色相 0/43/85/128/171/213）"""
    H, S, L = int(H), int(S), int(L)
    if S == 0:
        return (L, L, L)
    if L == 128 and S == 255:
        te = {0: (255, 0, 0), 255: (255, 0, 0), 43: (255, 255, 0), 85: (0, 255, 0),
              128: (0, 255, 255), 171: (0, 0, 255), 213: (255, 0, 255)}
        if H in te:
            return te[H]
    h, s, l = H / 255.0, S / 255.0, L / 255.0
    temp2 = l * (1.0 + s) if l < .5 else l + s - l * s
    temp1 = 2.0 * l - temp2
    t3 = [h + 1.0 / 3.0, h, h - 1.0 / 3.0]
    if t3[0] > 1.0:
        t3[0] -= 1.0
    if t3[2] < 0.0:
        t3[2] += 1.0
    chu = []
    for t in t3:
        if 6.0 * t < 1.0:
            chu.append(temp1 + (temp2 - temp1) * 6.0 * t)
        elif 2.0 * t < 1.0:
            chu.append(temp2)
        elif 3.0 * t < 2.0:
            chu.append(temp1 + (temp2 - temp1) * ((2.0 / 3.0) - t) * 6.0)
        else:
            chu.append(temp1)
    return (_qie(chu[0] * 255), _qie(chu[1] * 255), _qie(chu[2] * 255))


def hsv_dao_rgb(H, S, V):
    """AEG 的 hsv_to_rgb（色相 0-255 的整数版）"""
    H, S, V = int(H), int(S), int(V)
    if S == 255:
        te = {0: (V, 0, 0), 255: (V, 0, 0), 43: (V, V, 0), 85: (0, V, 0),
              128: (0, V, V), 171: (0, 0, V), 213: (V, 0, V)}
        if H in te:
            return te[H]
    if S == 0:
        return (V, V, V)
    h = H * 360
    s = _qie(S) * 256
    v = _qie(V) * 256
    Hi = h // 60 // 256
    f = h // 60 - Hi * 256
    p = v * (65535 - s) // 65536
    q = v * (65535 - (f * s) // 256) // 65536
    t = v * (65535 - ((255 - f) * s) // 256) // 65536
    biao = {0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
            3: (p, q, v), 4: (t, p, v), 5: (v, p, q)}
    r, g, b = biao.get(Hi, biao[5])
    return (_qie(r // 256), _qie(g // 256), _qie(b // 256))


def rgb_dao_hsl(R, G, B):
    """AEG 的 rgb_to_hsl"""
    r, g, b = int(R) / 255.0, int(G) / 255.0, int(B) / 255.0
    ma, mi = max(r, g, b), min(r, g, b)
    l = (mi + ma) / 2.0
    if mi == ma:
        h = s = 0.0
    else:
        s = ((ma - mi) / (ma + mi)) if l < .5 else ((ma - mi) / (2.0 - ma - mi))
        if r == ma:
            h = (g - b) / (ma - mi)
        elif g == ma:
            h = (b - r) / (ma - mi) + 2.0
        else:
            h = (r - g) / (ma - mi) + 4.0
    if h < 0:
        h += 6.0
    if h >= 6:
        h -= 6.0
    return (_qie(h * 256 / 6), _qie(s * 255), _qie(l * 255))


def rgb_dao_hsv(R, G, B):
    """AEG 的 rgb_to_hsv"""
    r, g, b = int(R) / 255.0, int(G) / 255.0, int(B) / 255.0
    ma, mi = max(r, g, b), min(r, g, b)
    v = ma
    s = 1.0 if ma < .001 else (ma - mi) / ma
    if mi == ma:
        h = 0.0
    elif ma == r:
        h = (g - b) / (ma - mi)
    elif ma == g:
        h = (b - r) / (ma - mi) + 2.0
    else:
        h = (r - g) / (ma - mi) + 4.0
    if h < 0:
        h += 6.0
    if h >= 6:
        h -= 6.0
    return (_qie(h * 256 / 6), _qie(s * 255), _qie(v * 255))


# ---- 颜色写法：照 AEG 的 libaegisub/common/parser.cpp + color.cpp ----
def se_dao_ass(r, g, b):
    """&HBBGGRR&（AEG 的 GetAssOverrideFormatted，调色板 ASS 那一栏就是它）"""
    return "&H%02X%02X%02X&" % (int(b) & 255, int(g) & 255, int(r) & 255)


def se_dao_html(r, g, b):
    """#RRGGBB（AEG 的 GetHexFormatted(false)）"""
    return "#%02X%02X%02X" % (int(r) & 255, int(g) & 255, int(b) & 255)


def se_dao_4(r, g, b, tou=0):
    """整份存着用的写法：[红, 绿, 蓝, 透明度]（透明度 0 = 不透明）"""
    return [int(r) & 255, int(g) & 255, int(b) & 255, int(tou) & 255]


def se_du(wen):
    """按 AEG 那套认颜色写法 -> [r, g, b, 透明度]；认不出来给 None

    AEG 认这些（parser.cpp 的 ass_color / css_color）：
        &HBBGGRR& / &HAABBGGRR / AABBGGRR / #RRGGBB / #RGB / rgb(...) / rgba(...)
        还有十进制（SSA 那种 a<<24|b<<16|g<<8|r）和颜色名（red、yellow…）。
    8 位那种头一个字节是**透明度**（0 = 不透明），跟 ASS 文件里一个口径。
    """
    wen = str(wen or "").strip()
    if not wen:
        return None
    chun = wen.replace(" ", "")
    zhi = chun
    if zhi.startswith("&") or zhi[:1].lower() == "h":
        zhi = zhi.lstrip("&")
        if zhi[:1].lower() == "h":
            zhi = zhi[1:]
        zhi = zhi.rstrip("&")
    else:
        zhi = ""
    if zhi and all(c in "0123456789abcdefABCDEF" for c in zhi):
        if len(zhi) == 6:
            return se_dao_4(int(zhi[4:6], 16), int(zhi[2:4], 16), int(zhi[0:2], 16), 0)
        if len(zhi) == 8:
            return se_dao_4(int(zhi[6:8], 16), int(zhi[4:6], 16),
                            int(zhi[2:4], 16), int(zhi[0:2], 16))
    if chun.startswith("#"):
        shi = chun[1:]
        if len(shi) == 3:
            shi = "".join(c * 2 for c in shi)
        if len(shi) == 6 and all(c in "0123456789abcdefABCDEF" for c in shi):
            return se_dao_4(int(shi[0:2], 16), int(shi[2:4], 16), int(shi[4:6], 16), 0)
        if len(shi) == 8 and all(c in "0123456789abcdefABCDEF" for c in shi):
            return se_dao_4(int(shi[0:2], 16), int(shi[2:4], 16),
                            int(shi[4:6], 16), 255 - int(shi[6:8], 16))
    if chun.lower().startswith(("rgb(", "rgba(")):
        shu = [x for x in chun[chun.index("(") + 1:chun.rindex(")")].split(",") if x]
        try:
            shu = [int(float(x)) for x in shu[:4]]
        except ValueError:
            shu = []
        if len(shu) >= 3:
            tou = 0 if len(shu) < 4 else max(0, 255 - _qie(shu[3]))
            return se_dao_4(_qie(shu[0]), _qie(shu[1]), _qie(shu[2]), tou)
    if chun.isdigit():          # SSA 那种十进制
        try:
            zhi = int(chun)
        except ValueError:
            zhi = 0
        if 0 <= zhi <= 0xFFFFFFFF:
            return se_dao_4(zhi & 255, (zhi >> 8) & 255,
                            (zhi >> 16) & 255, (zhi >> 24) & 255)
    se = QtGui.QColor(wen)      # 颜色名等交给 Qt（AEG 也认 CSS 名字）
    if se.isValid():
        return se_dao_4(se.red(), se.green(), se.blue(), 0)
    return None


def se_dao_qcolor(se):
    """[r, g, b, 透明度] -> QColor（Qt 的 alpha：255 = 不透明）"""
    se = list(se or [0, 0, 0, 0]) + [0] * 4
    yan = QtGui.QColor(int(se[0]) & 255, int(se[1]) & 255, int(se[2]) & 255)
    yan.setAlpha(255 - max(0, min(255, int(se[3]))))
    return yan


def guiyi_yanse(yanse):
    """外面给的东西（QColor / "#RRGGBB" / [r,g,b,透明度] / ASS 写法 / 颜色名）
    -> [r, g, b, 透明度]；给不了给 None"""
    if yanse is None:
        return None
    if isinstance(yanse, QtGui.QColor):
        return se_dao_4(yanse.red(), yanse.green(), yanse.blue(), 255 - yanse.alpha())
    if isinstance(yanse, (list, tuple)):
        return se_dao_4(*(list(yanse) + [0] * (4 - len(yanse)))[:4])
    return se_du(yanse)


def _tu_cong_rgb(zhen):
    """numpy (h, w, 3) uint8 -> QImage（拷一份，别让 QImage 吊着 numpy 的缓冲）"""
    zhen = np.ascontiguousarray(np.clip(zhen, 0, 255).astype(np.uint8))
    gao, kuan = int(zhen.shape[0]), int(zhen.shape[1])
    tu = QtGui.QImage(zhen.data, kuan, gao, 3 * kuan, QtGui.QImage.Format_RGB888)
    return tu.copy()


# ---- 预览色块（自己画，别指望 QLabel 的样式表在有些主题下生效）----
class _YulanSe(QtWidgets.QFrame):
    """40×40 的小方块，显示当前颜色（带透明度的话底下垫棋盘格）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._se = [255, 255, 255, 0]
        self.setFixedSize(40, 40)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)

    def shezhi_se(self, se):
        self._se = list(se) + [0] * 4
        self.update()

    def paintEvent(self, _shi):
        huabi = QtGui.QPainter(self)
        kuang = self.rect().adjusted(1, 1, -1, -1)
        ge = 5
        for y in range(kuang.top(), kuang.bottom() + 1, ge):
            for x in range(kuang.left(), kuang.right() + 1, ge):
                hei = ((x - kuang.left()) // ge + (y - kuang.top()) // ge) % 2
                huabi.fillRect(x, y, ge, ge,
                               QtGui.QColor("#FFFFFF" if hei else "#C0C0C0"))
        huabi.fillRect(kuang, se_dao_qcolor(self._se))
        huabi.end()


# ---- 色板 / 色条（AEG 的 ColorPickerSpectrum）----
class _SePu(QtWidgets.QWidget):
    """一块色板（2d）或一条色条（tiao），光标照 AEG：色板是小十字、色条是横线 + 三角"""

    dong_le = QtCore.pyqtSignal()

    def __init__(self, fangxiang="2d", parent=None):
        super().__init__(parent)
        self._fangxiang = "tiao" if fangxiang == "tiao" else "2d"
        self._tu = None
        self._x = -1
        self._y = -1
        self._zai_tuo = False
        self._jian = 4              # 三角边长（AEG 的 spectrum_horz_vert_arrow_size）
        if self._fangxiang == "2d":
            self.setFixedSize(_SEPU_BIAN + 2, _SEPU_BIAN + 2)
        else:
            self.setFixedSize(_SE_TIAO_KUAN + 2 + self._jian + 1, _SEPU_BIAN + 2)
        self.setCursor(QtCore.Qt.CrossCursor)

    def shezhi_tu(self, tu):
        self._tu = tu
        self.update()

    def dian(self):
        return (self._x, self._y)

    def shezhi_dian(self, x, y):
        x, y = int(x), int(y)
        if self._x == x and self._y == y:
            return
        self._x, self._y = x, y
        self.update()

    def paintEvent(self, _shi):
        huabi = QtGui.QPainter(self)
        kuan = self.width() - 2 - (self._jian + 1 if self._fangxiang == "tiao" else 0)
        gao = self.height() - 2
        if self._tu is not None:
            huabi.drawImage(1, 1, self._tu)
        if self._x >= 0 and self._y >= 0:
            huabi.save()
            huabi.setCompositionMode(QtGui.QPainter.CompositionMode_Xor)
            bi = QtGui.QPen(QtGui.QColor(255, 255, 255), 3)
            bi.setCapStyle(QtCore.Qt.FlatCap)
            huabi.setPen(bi)
            if self._fangxiang == "2d":
                huabi.drawLine(self._x - 4, self._y + 1, self._x + 7, self._y + 1)
                huabi.drawLine(self._x + 1, self._y - 4, self._x + 1, self._y + 7)
            else:
                huabi.drawLine(1, self._y + 1, kuan + 1, self._y + 1)
            huabi.restore()
            if self._fangxiang == "tiao":
                huabi.setPen(QtCore.Qt.NoPen)
                huabi.setBrush(QtGui.QColor("#000000"))
                huabi.drawPolygon(QtGui.QPolygonF([
                    QtCore.QPointF(kuan + 2, self._y + 1),
                    QtCore.QPointF(kuan + 2 + self._jian, self._y + 1 - self._jian),
                    QtCore.QPointF(kuan + 2 + self._jian, self._y + 1 + self._jian),
                ]))
        huabi.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.WindowText), 1))
        huabi.setBrush(QtCore.Qt.NoBrush)
        huabi.drawRect(0, 0, kuan + 1, gao + 1)
        huabi.end()

    def _an(self, dian):
        x = max(0, min(int(dian.x()) - 1, self.width() - 3 - self._jian))
        y = max(0, min(int(dian.y()) - 1, self.height() - 3))
        if self._fangxiang == "tiao":
            x = 0
        self.shezhi_dian(x, y)

    def mousePressEvent(self, shi_jian):
        if shi_jian.button() == QtCore.Qt.LeftButton:
            self._zai_tuo = True
            self.grabMouse()
            self._an(shi_jian.pos())
            self.dong_le.emit()
            shi_jian.accept()
            return
        super().mousePressEvent(shi_jian)

    def mouseMoveEvent(self, shi_jian):
        if self._zai_tuo:
            self._an(shi_jian.pos())
            self.dong_le.emit()
            shi_jian.accept()
            return
        super().mouseMoveEvent(shi_jian)

    def mouseReleaseEvent(self, shi_jian):
        if self._zai_tuo:
            self._zai_tuo = False
            self.releaseMouse()
            self._an(shi_jian.pos())
            self.dong_le.emit()
            shi_jian.accept()
            return
        super().mouseReleaseEvent(shi_jian)


# ---- 最近用过的颜色（AEG 的 ColorPickerRecent）----
class _ZuiJinSe(QtWidgets.QFrame):
    xuan_le = QtCore.pyqtSignal(object)      # [r, g, b, 透明度]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._se = [None] * (_ZUIJIN_LIE * _ZUIJIN_HANG)
        self.setFixedSize(_ZUIJIN_LIE * _ZUIJIN_GE + 2, _ZUIJIN_HANG * _ZUIJIN_GE + 2)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)

    def zhuang(self, se_liebiao):
        dian = [None] * (_ZUIJIN_LIE * _ZUIJIN_HANG)
        for i, se in enumerate(se_liebiao or []):
            if i >= len(dian):
                break
            dian[i] = se_dao_4(*se) if se else None
        self._se = dian
        self.update()

    def qu(self):
        return [x for x in self._se if x]

    def jia(self, se):
        """加一个颜色到最前面（已经在里面就挪到最前面）—— 照 AEG 的 AddColor"""
        se = se_dao_4(*se)
        if se in self._se:
            self._se.remove(se)
        self._se.insert(0, se)
        self._se = self._se[:_ZUIJIN_LIE * _ZUIJIN_HANG]
        self.update()

    def paintEvent(self, _shi):
        huabi = QtGui.QPainter(self)
        huabi.fillRect(self.rect(), self.palette().color(QtGui.QPalette.Window))
        huabi.setPen(QtCore.Qt.NoPen)
        for i, se in enumerate(self._se):
            if not se:
                continue
            lie = i % _ZUIJIN_LIE
            hang = i // _ZUIJIN_LIE
            huabi.fillRect(1 + lie * _ZUIJIN_GE, 1 + hang * _ZUIJIN_GE,
                           _ZUIJIN_GE, _ZUIJIN_GE, se_dao_qcolor(se))
        huabi.end()

    def mousePressEvent(self, shi_jian):
        if shi_jian.button() == QtCore.Qt.LeftButton:
            lie = (int(shi_jian.pos().x()) - 1) // _ZUIJIN_GE
            hang = (int(shi_jian.pos().y()) - 1) // _ZUIJIN_GE
            i = hang * _ZUIJIN_LIE + lie
            if 0 <= lie < _ZUIJIN_LIE and 0 <= hang < _ZUIJIN_HANG and 0 <= i < len(self._se):
                if self._se[i]:
                    self.xuan_le.emit(list(self._se[i]))
                    shi_jian.accept()
                    return
        super().mousePressEvent(shi_jian)


# ---- 取色器：滴管图标 + 屏幕放大镜（AEG 的 screen_dropper_icon + ColorPickerScreenDropper）----
def _diguan_tubiao(chicun=22):
    """画一个小滴管（AEG 用的是它自己的 eyedropper_tool 图标，我们这儿画一个）"""
    tu = QtGui.QPixmap(chicun, chicun)
    tu.fill(QtCore.Qt.transparent)
    h = QtGui.QPainter(tu)
    h.setRenderHint(QtGui.QPainter.Antialiasing, True)
    h.setPen(QtGui.QPen(QtGui.QColor("#4A4A4A"), 3, QtCore.Qt.SolidLine,
                        QtCore.Qt.RoundCap))
    h.drawLine(4, chicun - 4, chicun - 9, 9)            # 管身（斜着）
    h.setPen(QtCore.Qt.NoPen)
    h.setBrush(QtGui.QColor("#9A9A9A"))                 # 尖头
    h.drawPolygon(QtGui.QPolygonF([
        QtCore.QPointF(chicun - 10, 9),
        QtCore.QPointF(chicun - 3, 2),
        QtCore.QPointF(chicun - 1, 4),
        QtCore.QPointF(chicun - 8, 11),
    ]))
    h.setBrush(QtGui.QColor("#D0D0D0"))                 # 捏的那一头
    h.drawEllipse(QtCore.QPointF(5, chicun - 5), 4, 4)
    h.end()
    return tu


class _DiGuan(QtWidgets.QToolButton):
    """= AEG 的 screen_dropper_icon（滴管那个图标）

    它只管把三个鼠标事件原样报出去（带上图标内的坐标），逻辑全在对话框那边，
    照 dialog_colorpicker.cpp:1122-1148 的 OnDropperMouse 走 —— AEG 也是这么分的。
    """

    an_le = QtCore.pyqtSignal(QtCore.QPoint)      # 在图标上按下
    dong_le = QtCore.pyqtSignal(QtCore.QPoint)    # 在图标上拖动
    fang_le = QtCore.pyqtSignal(QtCore.QPoint)    # 在图标上松开

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(QtGui.QIcon(_diguan_tubiao()))
        self.setIconSize(QtCore.QSize(22, 22))
        self.setAutoRaise(True)
        self.setFixedSize(26, 26)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setToolTip("点一下滴管：光标走到哪，那块 7×7 就进小窗；在小窗里点那格取色")

    def mousePressEvent(self, shi_jian):
        try:
            if shi_jian.button() == QtCore.Qt.LeftButton:
                self.an_le.emit(shi_jian.pos())
                shi_jian.accept()
                return
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("滴管按下", cuowu)
        super().mousePressEvent(shi_jian)

    def mouseMoveEvent(self, shi_jian):
        try:
            # PyQt5 里没有 QWidget.hasMouseGrab()（那是 Qt6 才有的），
            # 判断"鼠标是不是被我抓着"要用 QWidget.mouseGrabber()
            if QtWidgets.QWidget.mouseGrabber() is self:
                self.dong_le.emit(shi_jian.pos())
                shi_jian.accept()
                return
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("滴管拖动", cuowu)
        super().mouseMoveEvent(shi_jian)

    def mouseReleaseEvent(self, shi_jian):
        try:
            if shi_jian.button() == QtCore.Qt.LeftButton:
                self.fang_le.emit(shi_jian.pos())
                shi_jian.accept()
                return
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("滴管松开", cuowu)
        super().mouseReleaseEvent(shi_jian)


class _QuSeQi(QtWidgets.QFrame):
    """= AEG 的 ColorPickerScreenDropper（取色小窗）

    一个 7×7 的屏幕截图放进 56×56 的 capture 里（放大 8 倍，构造时填白）。
      · drop_from_screen_xy() = AEG 的 DropFromScreenXY（374-423 行）
            抓 (x-resx/2, y-resy/2) 起 7×7 那块屏幕，拉伸 ×8 铺进 capture
      · mousePressEvent      = AEG 的 OnMouse（330-342 行）
            在 capture 里按下的那个像素就是取到的颜色（按下就取，不是松开）
    本身没有"抓着/没抓着"的状态 —— 那是图标那边的事（AEG 的 HasCapture）。
    """

    qu_le = QtCore.pyqtSignal(object)        # [r, g, b, 透明 0]

    def __init__(self, resx=7, resy=7, magnification=8, parent=None):
        super().__init__(parent)
        self._resx = int(resx)
        self._resy = int(resy)
        self._magni = int(magnification)
        # AEG 是 SetClientSize(resx*magnification, resy*magnification)：客户区 56×56，
        # 边框在客户区外面。Qt 的 setFrameStyle 画在控件里面，所以控件开 58×58，
        # 图从 (1,1) 开始铺，客户区还是 56×56、每格正好 8 像素
        self.setFixedSize(self._resx * self._magni + 2, self._resy * self._magni + 2)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self._capture = QtGui.QImage(
            self._resx * self._magni, self._resy * self._magni,
            QtGui.QImage.Format_RGB888)
        self._capture.fill(QtGui.QColor(255, 255, 255))     # AEG 构造时填成白的
        self._zhong = (self._resx // 2, self._resy // 2)     # 光标落在哪一格（画个框）

    def capture(self):
        return self._capture

    # = AEG 的 DropFromScreenXY
    def drop_from_screen_xy(self, x, y):
        try:
            self._drop(x, y)
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("抓屏", cuowu)

    def _drop(self, x, y):
        ban_x = self._resx // 2
        ban_y = self._resy // 2
        ping = QtWidgets.QApplication.screenAt(QtCore.QPoint(int(x), int(y)))
        if ping is None:
            ping = QtWidgets.QApplication.primaryScreen()
        if ping is None:
            return
        ju = ping.geometry()
        # grabWindow 要的是"这块屏幕内部"的坐标，给进来的是全桌面的，得减掉屏幕原点；
        # 再夹进屏幕里 —— 抓出界会返回空图。夹了之后光标就不在正中间了，所以
        # 记下它落在那 7×7 的第几格，画框和取色都照它来
        cx = ban_x
        cy = ban_y
        gx = int(x) - ju.x() - ban_x
        gy = int(y) - ju.y() - ban_y
        if gx < 0:
            cx += gx
            gx = 0
        if gy < 0:
            cy += gy
            gy = 0
        if gx + self._resx > ju.width():
            cx -= gx + self._resx - ju.width()
            gx = max(0, ju.width() - self._resx)
        if gy + self._resy > ju.height():
            cy -= gy + self._resy - ju.height()
            gy = max(0, ju.height() - self._resy)
        try:
            tu = ping.grabWindow(0, gx, gy, self._resx, self._resy)
        except Exception:            # noqa
            return
        if tu is None or tu.isNull():
            return
        # StretchBlit：7×7 那点像素拉成 56×56 铺满 capture
        self._capture = tu.toImage().scaled(
            self._resx * self._magni, self._resy * self._magni,
            QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation)
        self._zhong = (max(0, min(self._resx - 1, cx)),
                       max(0, min(self._resy - 1, cy)))
        self.update()

    # = AEG 的 OnMouse：在 capture 里按下，取那个像素
    def mousePressEvent(self, shi_jian):
        try:
            if shi_jian.button() != QtCore.Qt.LeftButton:
                super().mousePressEvent(shi_jian)
                return
            x = int(shi_jian.pos().x()) - 1    # 减掉边框那 1 像素，换成 capture 坐标
            y = int(shi_jian.pos().y()) - 1
            if 0 <= x < self._capture.width() and 0 <= y < self._capture.height():
                se = self._capture.pixelColor(x, y)
                self._zhong = (min(self._resx - 1, x // self._magni),
                               min(self._resy - 1, y // self._magni))
                self.update()
                self.qu_le.emit([se.red(), se.green(), se.blue(), 0])
                shi_jian.accept()
                return
            super().mousePressEvent(shi_jian)
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("小窗按下", cuowu)

    def mouseMoveEvent(self, shi_jian):
        """光标在小窗里挪：把要在哪一格取色圈出来（capture 本身不动）"""
        try:
            x = int(shi_jian.pos().x()) - 1
            y = int(shi_jian.pos().y()) - 1
            if 0 <= x < self._capture.width() and 0 <= y < self._capture.height():
                self._zhong = (min(self._resx - 1, x // self._magni),
                               min(self._resy - 1, y // self._magni))
                self.update()
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("小窗拖动", cuowu)
        super().mouseMoveEvent(shi_jian)

    # = AEG 的 OnPaint：就是 DrawBitmap(capture, 0, 0)
    def paintEvent(self, _shi):
        try:
            huabi = QtGui.QPainter(self)
            # 别做平滑：放大镜要看得出一个个像素（FastTransformation 是
            # Qt.TransformationMode 那边的枚举，不是 QPainter 的 RenderHint，
            # 写成 setRenderHint(QPainter.FastTransformation) 会 AttributeError）
            huabi.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
            huabi.drawImage(1, 1, self._capture)
            ge = self._magni
            cx, cy = self._zhong
            if 0 <= cx < self._resx and 0 <= cy < self._resy:
                huabi.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
                huabi.setBrush(QtCore.Qt.NoBrush)
                huabi.drawRect(1 + cx * ge, 1 + cy * ge, ge, ge)
            huabi.end()
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("小窗重画", cuowu)


# ---- 调色板本体 ----
class YsgColorDialog(QtWidgets.QDialog):
    """照 AEG 复刻的调色板；用法跟 QColorDialog.getColor 差不多：

        yan, hao = YsgColorDialog.tiao(self, yan, alpha=True, biaoti="选颜色")
    """

    def __init__(self, parent=None, yanse=None, alpha=True, biaoti="选择颜色"):
        super().__init__(parent)
        self.setWindowTitle(biaoti or "选择颜色")
        self._alpha = bool(alpha)
        self._tian = True                     # 正在程序化填值：别互相触发
        self._se = [0, 0, 0, 0]               # r, g, b, 透明度（0 = 不透明，照 AEG）
        self._guangpu_zang = True
        cun = self._du_cun()
        try:
            moshi = int(cun.get("moshi", 4))
        except (TypeError, ValueError):
            moshi = 4
        # 出厂默认 4 = HSV/色相（照 AEG 的 default_config.json 的 Mode: 4）
        self._moshi = moshi if 0 <= moshi < len(_GUANGPU_MOSHI) else 3

        # ---- 控件 ----
        self.xia_moshi = QtWidgets.QComboBox(self)
        self.xia_moshi.addItems(list(_GUANGPU_MOSHI))
        self.xia_moshi.setCurrentIndex(self._moshi)
        self.yulan = _YulanSe(self)

        self.sepu = _SePu("2d", self)
        self.setiao = _SePu("tiao", self)
        self.alpha_tiao = _SePu("tiao", self)
        if not self._alpha:
            self.alpha_tiao.hide()

        self.shuru = {}
        for jian in ("rgb", "hsl", "hsv"):
            self.shuru[jian] = []
            for _i in range(3):
                kuang = QtWidgets.QSpinBox(self)
                kuang.setRange(0, 255)
                kuang.setFixedWidth(_SHURU_KUAN)
                self.shuru[jian].append(kuang)
        self.shuru_ass = QtWidgets.QLineEdit(self)
        self.shuru_html = QtWidgets.QLineEdit(self)
        self.shuru_ass.setFixedWidth(_SHURU_KUAN)     # AEG 里这些框都是一个尺寸
        self.shuru_html.setFixedWidth(_SHURU_KUAN)
        self.shuru_alpha = QtWidgets.QSpinBox(self)
        self.shuru_alpha.setRange(0, 255)
        self.shuru_alpha.setFixedWidth(_SHURU_KUAN)
        if not self._alpha:
            self.shuru_alpha.hide()

        self.qu_se_qi = _QuSeQi(7, 7, 8, self)
        self.di_guan = _DiGuan(self)
        # 滴管图标那三个鼠标事件，全走对话框这边 —— AEG 也是 DialogColorPicker::
        # OnDropperMouse 一手管（dialog_colorpicker.cpp:1122-1148）
        self.di_guan.an_le.connect(self._quse_an)
        self.di_guan.dong_le.connect(self._quse_dong)
        self.di_guan.fang_le.connect(self._quse_fang)
        # AEG 抓的是"图标"那只鼠标（screen_dropper_icon->CaptureMouse()），靠事件流转发；
        # 我们多垫一层定时器：万一 grab 没抓住、事件到不了，也能靠 QCursor.pos() 把
        # 区域跟下去（只是取位置的手段不同，判断还是 AEG 那套 HasCapture）
        self._quse_zhua = False             # = AEG 的 screen_dropper_icon->HasCapture()
        self._quse_yijing = False           # = AEG 的 eyedropper_is_grabbed
        self._quse_an_dian = None           # = AEG 的 eyedropper_grab_point
        self._quse_jishi = QtCore.QTimer(self)
        self._quse_jishi.setInterval(30)
        self._quse_jishi.timeout.connect(self._quse_gen)
        self.zuijin = _ZuiJinSe(self)
        # 没存过就用 AEG 出厂那八个色（不是空格子）；存过就照存的那份来
        self.zuijin.zhuang([
            se_du(x) for x in (cun.get("zuijin") or _ZUIJIN_MOREN) if se_du(x)
        ])

        # ---- 摆位（照 AEG 的 dialog_colorpicker.cpp 那套 sizer 一比一抄）----
        # AEG 是 wxFlexGridSizer(3)：第一行放"光谱模式 + 预览"（只占色板那一格
        # 的宽，预览顶到色板右边），第二行依次是 色板 / 色条 / 透明度条。
        guangpu_kuang = QtWidgets.QGroupBox("色彩光谱", self)
        ding = QtWidgets.QHBoxLayout()
        ding.setContentsMargins(0, 0, 0, 0)
        ding.setSpacing(5)
        ding.addWidget(QtWidgets.QLabel("光谱模式:", self), 0, QtCore.Qt.AlignVCenter)
        ding.addWidget(self.xia_moshi, 0, QtCore.Qt.AlignVCenter)
        ding.addStretch(1)
        ding.addWidget(self.yulan, 0, QtCore.Qt.AlignVCenter)
        guangpu_wang = QtWidgets.QGridLayout()
        guangpu_wang.setContentsMargins(0, 0, 0, 0)
        guangpu_wang.setHorizontalSpacing(5)
        guangpu_wang.setVerticalSpacing(5)
        guangpu_wang.addLayout(ding, 0, 0)                  # 只压在色板这一列上
        guangpu_wang.addWidget(self.sepu, 1, 0)
        guangpu_wang.addWidget(self.setiao, 1, 1)
        guangpu_wang.addWidget(self.alpha_tiao, 1, 2)
        guangpu_wang.setColumnStretch(0, 1)
        guangpu = QtWidgets.QVBoxLayout(guangpu_kuang)
        guangpu.setContentsMargins(3, 3, 3, 3)              # AEG: wxALL, 3
        guangpu.addLayout(guangpu_wang)
        guangpu.addStretch(1)

        rgb_kuang = QtWidgets.QGroupBox("RGB色彩", self)
        rgb_wang = QtWidgets.QGridLayout(rgb_kuang)
        rgb_wang.setContentsMargins(3, 3, 3, 3)
        rgb_wang.setHorizontalSpacing(8)
        for i, ming in enumerate(("红:", "绿:", "蓝:")):
            rgb_wang.addWidget(QtWidgets.QLabel(ming, self), i, 0)
            rgb_wang.addWidget(self.shuru["rgb"][i], i, 1, QtCore.Qt.AlignLeft)
        rgb_wang.addWidget(QtWidgets.QLabel("ASS:", self), 0, 2)
        rgb_wang.addWidget(self.shuru_ass, 0, 3)
        rgb_wang.addWidget(QtWidgets.QLabel("HTML:", self), 1, 2)
        rgb_wang.addWidget(self.shuru_html, 1, 3)
        rgb_wang.addWidget(QtWidgets.QLabel("透明度:", self), 2, 2)
        rgb_wang.addWidget(self.shuru_alpha, 2, 3, QtCore.Qt.AlignLeft)
        rgb_wang.setColumnStretch(1, 1)
        rgb_wang.setColumnStretch(3, 1)

        hsl_kuang = QtWidgets.QGroupBox("HSL色彩", self)
        hsl_wang = QtWidgets.QGridLayout(hsl_kuang)
        hsl_wang.setContentsMargins(3, 3, 3, 3)
        for i, ming in enumerate(("色相:", "饱和度:", "亮度:")):
            hsl_wang.addWidget(QtWidgets.QLabel(ming, self), i, 0)
            hsl_wang.addWidget(self.shuru["hsl"][i], i, 1, QtCore.Qt.AlignLeft)

        hsv_kuang = QtWidgets.QGroupBox("HSV色彩", self)
        hsv_wang = QtWidgets.QGridLayout(hsv_kuang)
        hsv_wang.setContentsMargins(3, 3, 3, 3)
        for i, ming in enumerate(("色相:", "饱和度:", "明度:")):
            hsv_wang.addWidget(QtWidgets.QLabel(ming, self), i, 0)
            hsv_wang.addWidget(self.shuru["hsv"][i], i, 1, QtCore.Qt.AlignLeft)

        # 取色器那一行 AEG 没用分组框：左边滴管图标、中间放大镜、右边色格子
        qu_wang = QtWidgets.QHBoxLayout()
        qu_wang.setContentsMargins(0, 0, 0, 0)
        qu_wang.addStretch(1)
        qu_wang.addWidget(self.di_guan, 0, QtCore.Qt.AlignVCenter)
        qu_wang.addSpacing(5)
        qu_wang.addWidget(self.qu_se_qi, 0, QtCore.Qt.AlignVCenter)
        qu_wang.addStretch(1)
        qu_wang.addWidget(self.zuijin, 0, QtCore.Qt.AlignVCenter)
        qu_wang.addStretch(1)

        an = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        an.button(QtWidgets.QDialogButtonBox.Ok).setText("确认")
        an.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        an.accepted.connect(self.accept)
        an.rejected.connect(self.reject)

        you = QtWidgets.QVBoxLayout()
        you.setContentsMargins(0, 0, 0, 0)
        you.addWidget(rgb_kuang)
        hs = QtWidgets.QHBoxLayout()
        hs.setSpacing(5)                                    # AEG: AddSpacer(5)
        hs.addWidget(hsl_kuang)
        hs.addWidget(hsv_kuang)
        you.addLayout(hs)
        you.addLayout(qu_wang)
        you.addStretch(1)
        you.addWidget(an, 0, QtCore.Qt.AlignRight)

        zong = QtWidgets.QHBoxLayout(self)
        zong.setContentsMargins(5, 5, 5, 5)                 # AEG: wxALL, 5
        zong.setSpacing(5)
        zong.addWidget(guangpu_kuang, 1)
        zong.addLayout(you)

        # ---- 串起来 ----
        self.xia_moshi.currentIndexChanged.connect(self._huan_moshi)
        for jian in ("rgb", "hsl", "hsv"):
            for kuang in self.shuru[jian]:
                kuang.valueChanged.connect(lambda _v, j=jian: self._cong_shuzi(j))
        self.shuru_alpha.valueChanged.connect(self._cong_touming)
        self.shuru_ass.editingFinished.connect(self._cong_ass)
        self.shuru_html.editingFinished.connect(self._cong_html)
        self.sepu.dong_le.connect(self._cong_sepu)
        self.setiao.dong_le.connect(self._cong_setiao)
        self.alpha_tiao.dong_le.connect(self._cong_alpha_tiao)
        self.zuijin.xuan_le.connect(self._cong_zuijin)
        self.qu_se_qi.qu_le.connect(self._cong_qu_se)

        self._zhuang_se(guiyi_yanse(yanse) or se_dao_4(255, 255, 255, 0))
        self._tian = False
        self._shuaxin_guangpu()

    # ---- 存的那点东西：一律写进 gongzuotai.ini 的 [tiaoseban] 子项 ----
    @staticmethod
    def _du_cun():
        try:
            pei = _tiaose_pei()
            mo = pei.value("%s/%s" % (_TIAOSE_ZU, _TIAOSE_MOSHI), 4)
            zu = pei.value("%s/%s" % (_TIAOSE_ZU, _TIAOSE_ZUIJIN), [])
            if isinstance(zu, str):          # ini 里只有一个 / 读成字符串的情况
                zu = [x.strip() for x in zu.split(",") if x.strip()]
            return {"moshi": mo, "zuijin": list(zu or [])}
        except Exception:            # noqa
            return {}

    def _cun_dong(self):
        try:
            pei = _tiaose_pei()
            pei.setValue("%s/%s" % (_TIAOSE_ZU, _TIAOSE_MOSHI),
                         int(self.xia_moshi.currentIndex()))
            pei.setValue("%s/%s" % (_TIAOSE_ZU, _TIAOSE_ZUIJIN),
                         [se_dao_html(*x[:3]) for x in self.zuijin.qu()])
            pei.sync()
        except Exception:            # noqa
            pass

    # ---- 当前颜色（AEG 口径：透明度 0 = 不透明）----
    def se(self):
        return list(self._se)

    def yanse(self):
        """当前颜色的 QColor（Qt 的 alpha：255 = 不透明）"""
        return se_dao_qcolor(self._se)

    def _zhuang_se(self, se, zang=True):
        """把一份颜色铺到所有控件上（RGB/HSL/HSV/ASS/HTML/三个光标/预览）"""
        if not se:
            return
        se = list(se) + [0] * 4
        self._se = [int(se[0]) & 255, int(se[1]) & 255, int(se[2]) & 255, int(se[3]) & 255]
        if zang:
            self._guangpu_zang = True
        self._tian = True
        r, g, b = self._se[0], self._se[1], self._se[2]
        for i, zhi in enumerate((r, g, b)):
            self.shuru["rgb"][i].setValue(zhi)
        for i, zhi in enumerate(rgb_dao_hsl(r, g, b)):
            self.shuru["hsl"][i].setValue(zhi)
        for i, zhi in enumerate(rgb_dao_hsv(r, g, b)):
            self.shuru["hsv"][i].setValue(zhi)
        self.shuru_alpha.setValue(self._se[3])
        self.shuru_ass.setText(se_dao_ass(r, g, b))
        self.shuru_html.setText(se_dao_html(r, g, b))
        self._tian = False
        self._shuaxin_guangpu()

    # ---- 光谱模式 / 色板 ----
    def _huan_moshi(self, _i=None):
        self._guangpu_zang = True
        self._shuaxin_guangpu()

    def _sepu_tu(self):
        """按当前模式画 256×256 的色板（照 AEG 的 Make*Spectrum，整数算法）"""
        i = self.xia_moshi.currentIndex()
        r, g, b = self._se[0], self._se[1], self._se[2]
        if i in (0, 1, 2):
            # 0 = RGB/红（色板 = G×B）、1 = RGB/绿（R×B）、2 = RGB/蓝（R×G）
            # AEG 的色板：行是名字里第一个字母、列是第二个字母
            gu = np.arange(256, dtype=np.uint8)
            zhen = np.empty((256, 256, 3), np.uint8)
            if i == 0:
                zhen[..., 0] = r
                zhen[..., 1] = gu.reshape(256, 1)
                zhen[..., 2] = gu.reshape(1, 256)
            elif i == 1:
                zhen[..., 0] = gu.reshape(256, 1)
                zhen[..., 1] = g
                zhen[..., 2] = gu.reshape(1, 256)
            else:
                zhen[..., 0] = gu.reshape(256, 1)
                zhen[..., 1] = gu.reshape(1, 256)
                zhen[..., 2] = b
            return _tu_cong_rgb(zhen)
        if i == 3:
            # HSL/亮度：x = 饱和度（列）、y = 色相（行），亮度固定
            l = self.shuru["hsl"][2].value()
            chun = np.array([hsl_dao_rgb(h, 255, l) for h in range(256)], np.int32)
            s = np.arange(256, dtype=np.int32).reshape(1, 256, 1)
            zhen = (chun.reshape(256, 1, 3) * s // 256
                    + (255 - s) * l // 256)
            return _tu_cong_rgb(zhen)
        # HSV/色相：x = 饱和度（列）、y = 明度（行），色相固定
        h = self.shuru["hsv"][0].value()
        ma = np.array(hsv_dao_rgb(h, 255, 255), np.int32)
        v = np.arange(256, dtype=np.int32).reshape(256, 1, 1)
        s = np.arange(256, dtype=np.int32).reshape(1, 256, 1)
        rr = (255 - ma).reshape(1, 1, 3) * v // 256
        zhen = 255 - rr * s // 256 - (255 - v)
        return _tu_cong_rgb(zhen)

    def _tiao_tu(self):
        """色条（AEG 的 rgb_slider / hsl_slider / hsv_slider）"""
        i = self.xia_moshi.currentIndex()
        y = np.arange(256, dtype=np.uint8)
        zhen = np.zeros((256, _SE_TIAO_KUAN, 3), np.uint8)
        if i in (0, 1, 2):
            zhen[..., i] = y.reshape(256, 1)
        elif i == 3:
            zhen[...] = y.reshape(256, 1, 1)
        else:
            chun = np.array([hsv_dao_rgb(h, 255, 255) for h in range(256)], np.uint8)
            zhen[...] = chun.reshape(256, 1, 3)
        return _tu_cong_rgb(zhen)

    def _alpha_tu(self):
        """透明度条：底下垫一层棋盘格，越往下越透（照 AEG 的算法）"""
        r, g, b = self._se[0], self._se[1], self._se[2]
        y = np.arange(256, dtype=np.int32)
        inv = 0xFF - y
        ge = np.zeros((256, 2), np.int32)
        ge[:, 0] = 0x66 - inv * 0x66 // 0xFF
        ge[:, 1] = 0x99 - inv * 0x99 // 0xFF
        fan = ((y // _ALPHA_GESHI) & 1).astype(bool)
        ge[fan] = ge[fan][:, ::-1]
        zhen = np.empty((256, _SE_TIAO_KUAN, 3), np.uint8)
        for x in range(_SE_TIAO_KUAN):
            ban = ge[:, 0 if x < _SE_TIAO_KUAN // 2 else 1]
            zhen[:, x, 0] = np.clip(r * inv // 0xFF + ban, 0, 255)
            zhen[:, x, 1] = np.clip(g * inv // 0xFF + ban, 0, 255)
            zhen[:, x, 2] = np.clip(b * inv // 0xFF + ban, 0, 255)
        return _tu_cong_rgb(zhen)

    def _shuaxin_guangpu(self):
        """重画色板 / 色条 / 透明度条 / 三个光标 / 预览（AEG 的 UpdateSpectrumDisplay）"""
        i = self.xia_moshi.currentIndex()
        if self._guangpu_zang or self.sepu._tu is None:
            self.sepu.shezhi_tu(self._sepu_tu())
            self._guangpu_zang = False
        self.setiao.shezhi_tu(self._tiao_tu())
        self.alpha_tiao.shezhi_tu(self._alpha_tu())
        if i in (0, 1, 2):
            self.setiao.shezhi_dian(0, self.shuru["rgb"][i].value())
            if i == 0:
                self.sepu.shezhi_dian(self.shuru["rgb"][2].value(),
                                      self.shuru["rgb"][1].value())
            elif i == 1:
                self.sepu.shezhi_dian(self.shuru["rgb"][2].value(),
                                      self.shuru["rgb"][0].value())
            else:
                self.sepu.shezhi_dian(self.shuru["rgb"][1].value(),
                                      self.shuru["rgb"][0].value())
        elif i == 3:
            self.setiao.shezhi_dian(0, self.shuru["hsl"][2].value())
            self.sepu.shezhi_dian(self.shuru["hsl"][1].value(),
                                  self.shuru["hsl"][0].value())
        else:
            self.setiao.shezhi_dian(0, self.shuru["hsv"][0].value())
            self.sepu.shezhi_dian(self.shuru["hsv"][1].value(),
                                  self.shuru["hsv"][2].value())
        self.alpha_tiao.shezhi_dian(0, self._se[3])
        self.yulan.shezhi_se(self._se if self._alpha else self._se[:3] + [0])

    # ---- 各个入口（照 AEG 的 UpdateFrom*）----
    def _cong_shuzi(self, jian):
        if self._tian:
            return
        if jian == "rgb":
            r, g, b = [x.value() for x in self.shuru["rgb"]]
        elif jian == "hsl":
            r, g, b = hsl_dao_rgb(*[x.value() for x in self.shuru["hsl"]])
        else:
            r, g, b = hsv_dao_rgb(*[x.value() for x in self.shuru["hsv"]])
        self._se[0], self._se[1], self._se[2] = r, g, b
        self._tian = True
        if jian != "rgb":
            for i, zhi in enumerate((r, g, b)):
                self.shuru["rgb"][i].setValue(zhi)
        if jian != "hsl":
            for i, zhi in enumerate(rgb_dao_hsl(r, g, b)):
                self.shuru["hsl"][i].setValue(zhi)
        if jian != "hsv":
            for i, zhi in enumerate(rgb_dao_hsv(r, g, b)):
                self.shuru["hsv"][i].setValue(zhi)
        self.shuru_ass.setText(se_dao_ass(r, g, b))
        self.shuru_html.setText(se_dao_html(r, g, b))
        self._tian = False
        self._guangpu_zang = self._guangpu_zang or jian != "rgb"
        self._shuaxin_guangpu()

    def _cong_sepu(self):
        if self._tian:
            return
        x, y = self.sepu.dian()
        i = self.xia_moshi.currentIndex()
        self._tian = True
        if i == 0:
            self.shuru["rgb"][2].setValue(x)
            self.shuru["rgb"][1].setValue(y)
        elif i == 1:
            self.shuru["rgb"][2].setValue(x)
            self.shuru["rgb"][0].setValue(y)
        elif i == 2:
            self.shuru["rgb"][1].setValue(x)
            self.shuru["rgb"][0].setValue(y)
        elif i == 3:
            self.shuru["hsl"][1].setValue(x)
            self.shuru["hsl"][0].setValue(y)
        else:
            self.shuru["hsv"][1].setValue(x)
            self.shuru["hsv"][2].setValue(y)
        self._tian = False
        self._cong_shuzi("hsl" if i == 3 else ("hsv" if i == 4 else "rgb"))

    def _cong_setiao(self):
        if self._tian:
            return
        y = self.setiao.dian()[1]
        i = self.xia_moshi.currentIndex()
        self._tian = True
        if i in (0, 1, 2):
            self.shuru["rgb"][i].setValue(y)
        elif i == 3:
            self.shuru["hsl"][2].setValue(y)
        else:
            self.shuru["hsv"][0].setValue(y)
        self._tian = False
        self._guangpu_zang = True
        self._cong_shuzi("hsl" if i == 3 else ("hsv" if i == 4 else "rgb"))

    def _cong_alpha_tiao(self):
        if self._tian:
            return
        self._se[3] = self.alpha_tiao.dian()[1]
        self._tian = True
        self.shuru_alpha.setValue(self._se[3])
        self._tian = False
        self._shuaxin_guangpu()

    def _cong_touming(self):
        if self._tian:
            return
        self._se[3] = self.shuru_alpha.value()
        self._shuaxin_guangpu()

    def _cong_ass(self):
        se = se_du(self.shuru_ass.text())
        if se is None:
            self.shuru_ass.setText(se_dao_ass(*self._se[:3]))
            return
        self._zhuang_se([se[0], se[1], se[2], self._se[3]])

    def _cong_html(self):
        se = se_du(self.shuru_html.text())
        if se is None:
            self.shuru_html.setText(se_dao_html(*self._se[:3]))
            return
        self._zhuang_se([se[0], se[1], se[2], self._se[3]])

    def _cong_zuijin(self, se):
        if se is not None:
            self._zhuang_se([se[0], se[1], se[2], self._se[3]])

    # ---- 取色器：照 AEG 的 OnDropperMouse（dialog_colorpicker.cpp:1122-1148）----
    def _quse_quan(self):
        """那把屏抓进小窗 —— AEG 那段 `if (HasCapture()) DropFromScreenXY(...)`"""
        if not self._quse_zhua:
            return
        dian = QtGui.QCursor.pos()
        self.qu_se_qi.drop_from_screen_xy(dian.x(), dian.y())

    def _quse_an(self, dian):
        """= AEG 的 LeftDown 那一段：换滴管光标、图标清空、抓住鼠标、记按点

        AEG 记的是 evt.GetPosition()（图标内坐标），比较时也用同一个；我们这边还有
        定时器兜底那条路只有全屏坐标，所以统一都换算成全屏坐标存、比。
        """
        try:
            self._quse_an_nei(dian)
        except Exception as cuowu:               # noqa
            _ji_ri_zhi("滴管按下", cuowu)

    def _quse_an_nei(self, dian):
        if self._quse_zhua:                 # AEG: if (LeftDown && !HasCapture())
            return
        self._quse_zhua = True
        self._quse_yijing = False
        self._quse_an_dian = self.di_guan.mapToGlobal(QtCore.QPoint(dian))
        self.di_guan.setIcon(QtGui.QIcon())          # SetBitmap(wxNullBitmap)
        try:
            self.di_guan.grabMouse()                 # CaptureMouse()
        except Exception:            # noqa
            pass
        self._quse_jishi.start()
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        except Exception:            # noqa
            pass
        self._quse_quan()

    def _quse_dong(self, dian):
        """= AEG OnDropperMouse 尾部：还抓着就把光标那块屏幕抓进小窗"""
        self._quse_quan()

    def _quse_fang(self, dian):
        """= AEG 的 LeftUp 那一段

        `if (is_grabbed || |位移| > 7) 放掉; else is_grabbed = true;`
        短按一下（≤7px）就是"抓住不放"：不按键也继续跟着光标走，直到再按一次。
        """
        if not self._quse_zhua:
            return
        if dian is None:
            dian_now = QtGui.QCursor.pos()
        else:
            dian_now = self.di_guan.mapToGlobal(QtCore.QPoint(dian))
        an = self._quse_an_dian or dian_now
        yuan = abs(dian_now.x() - an.x()) + abs(dian_now.y() - an.y())
        if self._quse_yijing or yuan > 7:
            self._quse_fang_shou()
        else:
            self._quse_yijing = True
        self._quse_quan()

    def _quse_fang_shou(self):
        """放掉：不再跟，图留着（AEG 的 ReleaseMouse，capture 一直留着）"""
        self._quse_zhua = False
        self._quse_jishi.stop()
        try:
            self.di_guan.releaseMouse()
        except Exception:            # noqa
            pass
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:            # noqa
            pass
        self.di_guan.setIcon(QtGui.QIcon(_diguan_tubiao()))     # SetBitmap(eyedropper_bitmap)
        self.qu_se_qi.update()

    def _quse_gen(self):
        """定时器兜底：还抓着就跟光标；左键松了就补一次 LeftUp（事件没收到的情况）"""
        if not self._quse_zhua:
            return
        self._quse_quan()
        if not (QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton):
            if not self._quse_yijing:
                self._quse_fang(None)

    def _cong_qu_se(self, se):
        """小窗报上来的颜色 —— AEG 里它和"最近用过"走同一个 OnRecentSelect：
        `new_color.a = cur_color.a; SetColor(new_color);`，也就是**透明度保持不变**

        顺手把"跟着光标走"停掉（AEG 里点击小窗那一下会转到图标上走 LeftUp，效果一样）：
        不然下一拍定时器又把这格图刷新了，就没法在小窗里来回点着挑颜色。
        """
        if se is None:
            return
        if self._quse_zhua:
            self._quse_fang_shou()
        self._zhuang_se([se[0], se[1], se[2], self._se[3]], zang=False)

    # ---- 收尾 ----
    def keyPressEvent(self, shi_jian):
        """Esc：正抓着（在跟光标走）就先放掉；小窗里那张图留着，还能接着点"""
        if shi_jian.key() == QtCore.Qt.Key_Escape and self._quse_zhua:
            self._quse_fang_shou()
            shi_jian.accept()
            return
        super().keyPressEvent(shi_jian)

    def accept(self):
        self.zuijin.jia(self._se)
        self._cun_dong()
        super().accept()

    @staticmethod
    def tiao(parent=None, yanse=None, alpha=True, biaoti="选择颜色"):
        """弹一次调色板 -> (QColor, 确定没确定)

        yanse 可以给 QColor / "#RRGGBB" / [r, g, b, 透明度] / ASS 写法 / 颜色名。
        """
        chuang = YsgColorDialog(parent, guiyi_yanse(yanse), alpha, biaoti)
        if chuang.exec_() == QtWidgets.QDialog.Accepted:
            return (chuang.yanse(), True)
        return (None, False)
