"""选字体：直接调 Windows 自己的字体对话框（Aegisub 就是这么干的）

为什么不走 Qt 的 QFontDialog：Qt 在 Windows 上只给文件框做了原生，字体框
是 Qt 自己画的 —— 界面永远是英文（项目只装了自己的 zh_CN.qm，没装 Qt 内置
那几个词），样例那块还是按点数值硬画，字号一大就从中间裁掉，看不出字形。

Aegisub 走的是系统 ChooseFont，中文系统上就是中文那一套（字体 / 字形 / 大小
/ 效果 / 示例 / 脚本 / 颜色 / 显示更多字体），示例比例也是系统自己排的。
这里照它来。

返回值是字体名、字号、粗体、斜体、下划线、删除线、颜色 —— 跟原来那条
不一样的才写进字幕（\\fs \\b \\i \\u \\s \\c），跟 AEG 的 fn 按钮一个做法。
AEG 那个框里颜色 / 下划线 / 删除线是不生效的，我们这边都通了。
"""

import ctypes
from ctypes import wintypes

LF_FACESIZE = 32
CF_SCREENFONTS = 0x00000001
CF_INITTOLOGFONTSTRUCT = 0x00000040
CF_EFFECTS = 0x00000100
DEFAULT_CHARSET = 1
LOGPIXELSY = 90


class LOGFONTW(ctypes.Structure):
    """Windows 的 LOGFONTW：告诉系统"现在用的是哪款字体"，回来也写在这"""

    _fields_ = [
        ("lfHeight", wintypes.LONG),
        ("lfWidth", wintypes.LONG),
        ("lfEscapement", wintypes.LONG),
        ("lfOrientation", wintypes.LONG),
        ("lfWeight", wintypes.LONG),
        ("lfItalic", wintypes.BYTE),
        ("lfUnderline", wintypes.BYTE),
        ("lfStrikeOut", wintypes.BYTE),
        ("lfCharSet", wintypes.BYTE),
        ("lfOutPrecision", wintypes.BYTE),
        ("lfClipPrecision", wintypes.BYTE),
        ("lfQuality", wintypes.BYTE),
        ("lfPitchAndFamily", wintypes.BYTE),
        ("lfFaceName", wintypes.WCHAR * LF_FACESIZE),
    ]


class CHOOSEFONTW(ctypes.Structure):
    """Windows 的 CHOOSEFONTW：ChooseFont 的参数结构"""

    _fields_ = [
        ("lStructSize", wintypes.DWORD),
        ("hwndOwner", wintypes.HWND),
        ("hDC", wintypes.HDC),
        ("lpLogFont", ctypes.POINTER(LOGFONTW)),
        ("iPointSize", wintypes.INT),        # 字号，单位是 1/10 点
        ("Flags", wintypes.DWORD),
        ("rgbColors", wintypes.DWORD),
        ("lCustData", wintypes.LPARAM),
        ("lpfnHook", ctypes.c_void_p),
        ("lpTemplateName", wintypes.LPCWSTR),
        ("hInstance", wintypes.HINSTANCE),
        ("lpszStyle", wintypes.LPWSTR),
        ("nFontType", wintypes.WORD),
        ("__peiqi", wintypes.WORD),         # 结构体对齐的填充，不许删
        ("nSizeMin", wintypes.INT),
        ("nSizeMax", wintypes.INT),
    ]


def _dpi():
    """屏幕一英寸多少像素（把点换算成像素高用，算不出来按 96）"""
    try:
        hdc = ctypes.windll.user32.GetDC(None)
        dpi = int(ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSY))
        ctypes.windll.user32.ReleaseDC(None, hdc)
        return dpi if dpi > 0 else 96
    except Exception:       # noqa
        return 96


def _se_dao_bgr(se):
    """「#RRGGBB」-> Windows 的 COLORREF（0x00BBGGRR，红在最低位）"""
    wen = str(se or "").strip().lstrip("#")
    if len(wen) != 6:
        return 0
    try:
        r = int(wen[0:2], 16)
        g = int(wen[2:4], 16)
        b = int(wen[4:6], 16)
    except ValueError:
        return 0
    return (r & 255) | ((g & 255) << 8) | ((b & 255) << 16)


def _bgr_dao_se(zhi):
    """COLORREF -> 「#RRGGBB」"""
    zhi = int(zhi or 0) & 0xFFFFFF
    return "#%02X%02X%02X" % (zhi & 255, (zhi >> 8) & 255, (zhi >> 16) & 255)


class YsgFontDialog:
    """弹系统字体对话框；要字体名、字号、字形、下划线/删除线、颜色"""

    @staticmethod
    def tiao(parent=None, ming="", daxiao=0, biaoti="字体",
             cu=False, xie=False, xia=False, shan=False, se=""):
        """弹一次 -> (字体名, 字号, 粗体, 斜体, 下划线, 删除线, 颜色, 确定没确定)

        取消时字体名给 None。传进来的都是打开时先摆上的当前值（这条字幕
        正在用的），系统框里「字形 / 效果 / 颜色」就按这些摆好。

        注意：AEG 那个框里颜色、下划线、删除线是坏的（10 年前就那样），
        它只认字体名和字号；我们这边是真生效的。
        """
        ming = str(ming or "").replace("@", "")
        try:
            daxiao = int(round(float(daxiao or 0)))
        except (TypeError, ValueError):
            daxiao = 0

        zi = LOGFONTW()
        zi.lfWeight = 700 if cu else 400    # FW_BOLD / FW_NORMAL
        zi.lfItalic = 1 if xie else 0
        zi.lfUnderline = 1 if xia else 0
        zi.lfStrikeOut = 1 if shan else 0
        zi.lfCharSet = DEFAULT_CHARSET
        zi.lfFaceName = ming[: LF_FACESIZE - 1]
        if daxiao > 0:
            zi.lfHeight = -int(round(daxiao * _dpi() / 72.0))

        can = CHOOSEFONTW()
        can.lStructSize = ctypes.sizeof(CHOOSEFONTW)
        ju = 0
        try:
            # 谁的窗口弹出来的，就挂在谁身上
            chuang = parent.window() if parent is not None else None
            if chuang is not None:
                ju = int(chuang.winId() or 0)
        except Exception:       # noqa
            ju = 0
        can.hwndOwner = ju or None
        can.lpLogFont = ctypes.pointer(zi)
        can.iPointSize = max(0, daxiao) * 10
        can.rgbColors = _se_dao_bgr(se)
        can.Flags = CF_SCREENFONTS | CF_INITTOLOGFONTSTRUCT | CF_EFFECTS
        if not ctypes.windll.comdlg32.ChooseFontW(ctypes.byref(can)):
            return (None, 0, False, False, False, False, "", False)
        ming = str(zi.lfFaceName or "").strip()
        # iPointSize 才是"点"；lfHeight 是像素高，还带着 DPI，别拿它当字号
        xin_daxiao = int(round(int(can.iPointSize) / 10.0)) if can.iPointSize else 0
        # 字形那一栏回来的是字重和斜体位：700 以上算粗体
        xin_cu = int(zi.lfWeight) >= 700
        xin_xie = bool(zi.lfItalic)
        xin_xia = bool(zi.lfUnderline)
        xin_shan = bool(zi.lfStrikeOut)
        xin_se = _bgr_dao_se(can.rgbColors)
        return (ming, xin_daxiao, xin_cu, xin_xie,
                xin_xia, xin_shan, xin_se, True)
