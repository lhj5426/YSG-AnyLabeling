"""
秒级转发器：EnumWindows + SendMessage(WM_COPYDATA)。
只依赖 ctypes + stdlib，无需 PyQt/conda/yaml。

使用 EnumWindows 确保在有多个子窗口（横向/垂直/瀑布流 viewer）
打开时也能精确找到主窗口 HWND，避免向错误的窗口发送 WM_COPYDATA。

匹配优先级：
1. xanylabeling_TITLE.ini 里的标题精确匹配；
2. 标题包含 "AnyLabeling" 的窗口；
3. 原始默认标题 APP_TITLE。
"""
import os
import sys
import ctypes

from app_info import __appname__

WM_COPYDATA = 0x004A
APP_TITLE = __appname__


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", ctypes.c_void_p),
        ("cbData", ctypes.c_uint32),
        ("lpData", ctypes.c_void_p),
    ]


def _load_active_titles():
    """读项目根目录下的 xanylabeling_TITLE.ini，每行一个标题。"""
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    title_file = os.path.join(app_dir, "xanylabeling_TITLE.ini")
    titles = set()
    if not os.path.isfile(title_file):
        return titles
    try:
        with open(title_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    titles.add(name)
        return titles
    except Exception:
        return titles


def _find_main_window():
    """使用 EnumWindows 按优先级查找主窗口 HWND。"""
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    GWL_STYLE = -16
    WS_VISIBLE = 0x10000000

    active_titles = _load_active_titles()

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
    )

    def _enum_match(match_fn):
        """遍历顶层窗口，返回第一个符合 match_fn 且是 Qt 主窗口的 HWND。"""
        found = []

        @WNDENUMPROC
        def enum_proc(hwnd, lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value or ""

            if not match_fn(title):
                return True

            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            if not (style & WS_VISIBLE):
                return True
            class_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buf, 256)
            class_name = class_buf.value or ""
            if not (class_name.startswith("Qt") and "Dialog" not in class_name):
                return True
            found.append(hwnd)
            return True

        user32.EnumWindows(enum_proc, 0)
        return found[0] if found else None

    # 优先级 1：配置文件精确匹配
    if active_titles:
        hwnd = _enum_match(lambda title: title in active_titles)
        if hwnd:
            return hwnd

    # 优先级 2：包含 AnyLabeling
    hwnd = _enum_match(lambda title: "AnyLabeling" in title)
    if hwnd:
        return hwnd

    # 优先级 3：原始默认标题
    return _enum_match(lambda title: title == APP_TITLE)


def _shi_qt_zhu_chuangkou(hwnd):
    """兜底认窗时用：必须是可见的 Qt 主窗口（类名以 Qt 开头、不是 Dialog）。

    FindWindowW 只比标题、根本不看窗口类，别的软件标题撞名就会中招
    （比如 SourceGit 把仓库文件夹名当标题，正好就叫 YSG-AnyLabeling）。
    """
    if not hwnd:
        return False
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    GWL_STYLE = -16
    WS_VISIBLE = 0x10000000
    if not (u32.GetWindowLongW(hwnd, GWL_STYLE) & WS_VISIBLE):
        return False
    lei_buf = ctypes.create_unicode_buffer(256)
    u32.GetClassNameW(hwnd, lei_buf, 256)
    lei_ming = lei_buf.value or ""
    return lei_ming.startswith("Qt") and "Dialog" not in lei_ming


if len(sys.argv) < 2:
    sys.exit(2)

user32 = ctypes.WinDLL("user32", use_last_error=True)

titles = _load_active_titles()

hwnd = _find_main_window()
if not hwnd:
    for t in titles:
        hou_xuan = user32.FindWindowW(None, t)
        if _shi_qt_zhu_chuangkou(hou_xuan):
            hwnd = hou_xuan
            break
if not hwnd:
    hou_xuan = user32.FindWindowW(None, APP_TITLE)
    if _shi_qt_zhu_chuangkou(hou_xuan):
        hwnd = hou_xuan
if not hwnd:
    sys.exit(2)

path = sys.argv[1].strip('"')
try:
    print(f"已发送路径到已有窗口: {path}")
except OSError:
    pass

data = path.encode("utf-8")
buf = ctypes.create_string_buffer(data)
cds = COPYDATASTRUCT(0, len(data), ctypes.cast(buf, ctypes.c_void_p))
user32.SendMessageW(hwnd, WM_COPYDATA, 0, ctypes.byref(cds))
sys.exit(0)
