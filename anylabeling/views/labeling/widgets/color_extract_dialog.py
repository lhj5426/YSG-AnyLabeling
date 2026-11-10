"""
颜色提取对话框 —— 纯算法版（移植自 Leptonica pixColorSegment）

对矩形框内的像素做无监督颜色聚类，判定「文字色」和「背景色」，
写入 shape.attributes["fg"] / ["bg"]，格式与 Manga-OCR 完全一致：
    attributes["fg"] = [r, g, b]    # 0-255 整数列表
    attributes["bg"] = [r, g, b]

不依赖任何模型、不加载权重、不训练。同样的图 + 同样的参数 → 结果永远一致。

算法照抄 leptonica/src/colorseg.c 的四阶段：
    Phase 1  贪心聚类（按光栅序，距离 <= maxdist 就归入已有簇，否则新建簇）
             超过 maxcolors 就把 maxdist 乘 1.3 重试，最多 20 次
    Phase 2  非贪心重分配（每个色归到最近的簇平均色）
    Phase 3  按像素数从多到少，逐色做形态学闭运算，吞掉杂色点
    Phase 4  只保留像素数前 finalcolors 的簇，其余像素按原始色重分配

与 C 版的唯一差别：在「唯一颜色」层面循环，而不是逐像素循环。
同色像素的归类结果必然相同，所以结果等价，但快几十倍。
"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import json
import os
import os.path as osp

# 新的取色引擎：看「谁把谁包在里面」来定谁是字、谁是描边，
# 不靠面积阈值，所以换画风不用重新调参数。
try:
    from .color_extract_engine import qu_zise as _qu_zise_xin
except ImportError:  # 万一被当脚本单独跑
    from color_extract_engine import qu_zise as _qu_zise_xin


# 判定阈值。界面上可调，默认值是照着几部漫画试出来的 ——
# 换画风差别大的作品时这几个数通常要重调。
TH = {
    "white_min": 0.38,      # 白占比下限。低于它，白是笔画而不是背景
    "hole_min": 0.015,      # 黑围出的洞至少要占这么多，才认黑是描边
    "chroma_min": 50,       # 彩色字门槛，低于它不算彩色
    "adj_black": 0.25,      # 彩色像素里紧挨着黑描边的比例
    "border_frac": 0.15,    # 颜色在框边出现超过这个比例，就算背景
}


# ---------------------------------------------------------------------------
# 形态学：纯 numpy 实现，不依赖 cv2 / scipy
# ---------------------------------------------------------------------------

def _dilate(mask, k):
    """二值膨胀，方形核 k×k（分离式：先横后竖）

    用「整体平移再按位或」而不是 sliding_window_view + any：
    后者要建 (H,W,k) 的跨步视图再做一次归约，比直接 OR 慢好几倍。
    """
    if k <= 1:
        return mask
    if k % 2 == 0:
        k += 1                      # 偶数核会让输出尺寸差一格，统一成奇数
    r = k // 2
    h, w = mask.shape

    p = np.zeros((h, w + 2 * r), dtype=bool)
    p[:, r : r + w] = mask
    out = p[:, 0:w].copy()
    for dx in range(1, k):
        out |= p[:, dx : dx + w]

    q = np.zeros((h + 2 * r, w), dtype=bool)
    q[r : r + h] = out
    res = q[0:h].copy()
    for dy in range(1, k):
        res |= q[dy : dy + h]
    return res


def _erode(mask, k):
    """二值腐蚀 = 反相膨胀再反相"""
    if k <= 1:
        return mask
    return ~_dilate(~mask, k)


def _close(mask, k):
    """闭运算 = 先膨胀后腐蚀（填小洞、连小缝）"""
    if k <= 1:
        return mask
    return _erode(_dilate(mask, k), k)


# ---------------------------------------------------------------------------
# pixColorSegment 的 numpy 移植
# ---------------------------------------------------------------------------

MAX_ALLOWED_ITERATIONS = 20     # 与 C 版一致
DIST_EXPAND_FACT = 1.3          # 与 C 版一致

# 单次聚类最多吃多少像素。超过就先抽样降采样。
# 别设太小：步长一大，细笔画的纯色会被抽样抽没（整图 424x867 步长 4 时红字就消失了）。
MAX_CLUSTER_PIXELS = 120000


def _pack_rgb(arr):
    """(N,3) uint8 → (N,) uint32，一维 unique 比三维快得多"""
    a = arr.astype(np.uint32)
    return (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]


def _unpack_rgb(packed):
    """(N,) uint32 → (N,3) uint8"""
    return np.stack(
        [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1
    ).astype(np.uint8)


def _downsample(rgb, max_pixels=MAX_CLUSTER_PIXELS):
    """大框先降采样，返回 (新图, 步长)

    用最近邻抽样，不用均值池化：均值会把细笔画和背景混成一种中间色
    （红字 + 白底 → 粉色），那种假颜色饱和度还不低，会被误判成文字色。
    """
    h, w = rgb.shape[:2]
    if h * w <= max_pixels:
        return rgb, 1
    step = int(np.ceil(np.sqrt(h * w / max_pixels)))
    return np.ascontiguousarray(rgb[::step, ::step]), step


def _try_cluster(rgb, maxdist, maxcolors):
    """Phase 1 + Phase 2。

    返回 (index_map, palette) 或 None（簇数超限，需要放大 maxdist 重试）
    """
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)

    # 唯一色 + 首次出现位置（按光栅序，与 C 版遍历顺序对齐）
    packed = _pack_rgb(flat)
    uniq_packed, first_pos, inverse, counts = np.unique(
        packed, return_index=True, return_inverse=True, return_counts=True
    )
    uniq = _unpack_rgb(uniq_packed)
    order = np.argsort(first_pos)

    # ---- Phase 1：贪心聚类 ----
    maxdist2 = maxdist * maxdist
    reps = []                                    # 各簇代表色
    color_to_cluster = np.full(len(uniq), -1, dtype=np.int32)

    for idx in order:
        c = uniq[idx].astype(np.int32)
        assigned = -1
        for k, rep in enumerate(reps):
            dr = int(c[0]) - int(rep[0])
            dg = int(c[1]) - int(rep[1])
            db = int(c[2]) - int(rep[2])
            if dr * dr + dg * dg + db * db <= maxdist2:
                assigned = k
                break
        if assigned < 0:
            if len(reps) >= maxcolors:
                return None                      # 超限，外层放大 maxdist
            reps.append(c)
            assigned = len(reps) - 1
        color_to_cluster[idx] = assigned

    ncl = len(reps)

    # 各簇代表色：取「簇内出现次数最多的那个真实颜色」，不取平均值。
    # 平均值会被抗锯齿过渡色拉脏（纯红字变成 (184,115,117) 脏粉色）；
    # 也不能取「最饱和」的，那样会挑到 JPEG 边缘的暗色噪点。
    # 众数色一定是最主要的那个色，且必然在原图里真实存在。
    palette = np.zeros((ncl, 3), dtype=np.uint8)
    order_by_cluster = np.lexsort((-counts, color_to_cluster))
    sorted_cluster = color_to_cluster[order_by_cluster]
    first = np.ones(len(sorted_cluster), dtype=bool)
    first[1:] = sorted_cluster[1:] != sorted_cluster[:-1]
    winner = order_by_cluster[first]
    palette[color_to_cluster[winner]] = uniq[winner]

    # ---- Phase 2：非贪心重分配，每个唯一色归到最近的簇平均色 ----
    d2 = ((uniq[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)) ** 2).sum(axis=-1)
    best = d2.argmin(axis=1).astype(np.int32)
    index_map = best[inverse].reshape(h, w)
    return index_map, palette


def pix_color_segment(rgb, maxdist=90, maxcolors=12, selsize=1, finalcolors=6):
    """完整四阶段颜色分割。

    Args:
        rgb: (H, W, 3) uint8，RGB 顺序
        maxdist: 欧氏距离阈值
        maxcolors: 第一阶段允许的最大簇数
        selsize: 闭运算核尺寸
        finalcolors: 最终保留的颜色数

    Returns:
        (index_map, palette)
        index_map: (H, W) int32，每个像素的颜色索引
        palette:   (N, 3) uint8
    """
    newmaxdist = int(maxdist)

    # 大框先降采样（均值池化），颜色统计不受影响，还顺带压掉 JPEG 噪点
    small, step = _downsample(rgb)
    selsize_eff = max(1, int(round(selsize / step))) if selsize > 1 else 1

    result = None
    for _ in range(MAX_ALLOWED_ITERATIONS):
        result = _try_cluster(small, newmaxdist, maxcolors)
        if result is not None:
            break
        newmaxdist = int(DIST_EXPAND_FACT * newmaxdist)
    if result is None:
        return None

    index_map, palette = result
    ncl = len(palette)

    # ---- Phase 3：按像素数从多到少，逐色闭运算 ----
    if selsize_eff > 1 and ncl > 0:
        pop = np.bincount(index_map.ravel(), minlength=ncl)
        for k in np.argsort(-pop):
            if pop[k] == 0:
                continue
            m = index_map == k
            added = _close(m, selsize_eff) & ~m
            if added.any():
                index_map[added] = k

    # ---- Phase 4：只留前 finalcolors 个簇 ----
    pop = np.bincount(index_map.ravel(), minlength=ncl)
    if ncl > finalcolors:
        order = np.argsort(-pop)
        keep = list(order[:finalcolors])

        # 保护：最终会被判成文字色的那个簇不能被裁掉，
        # 否则红字这种「像素少但醒目」的颜色会被灰白过渡色挤出去。
        bg_i = int(order[0])
        others = [i for i in range(ncl) if pop[i] > 0 and i != bg_i]
        if others:
            key_i = max(others, key=lambda i: _color_score(palette[i], palette[bg_i]))
            if key_i not in keep:
                keep[-1] = key_i
        keep = np.array(sorted(set(keep)))

        keep_mask = np.zeros(ncl, dtype=bool)
        keep_mask[keep] = True

        drop_pixels = ~keep_mask[index_map]          # 需要重分配的像素
        if drop_pixels.any():
            # 在唯一色层面找最近的保留色（用原始像素值）
            flat = small.reshape(-1, 3)
            packed = _pack_rgb(flat)
            uniq_packed, inverse = np.unique(packed, return_inverse=True)
            uniq = _unpack_rgb(uniq_packed)
            keep_pal = palette[keep].astype(np.int32)
            d2 = ((uniq[:, None, :].astype(np.int32) - keep_pal[None, :, :]) ** 2).sum(axis=-1)
            new_idx = d2.argmin(axis=1)[inverse].reshape(index_map.shape)

            remap = -np.ones(ncl, dtype=np.int32)
            remap[keep] = np.arange(len(keep), dtype=np.int32)
            index_map = np.where(drop_pixels, new_idx, remap[index_map])
        palette = palette[keep]

    return index_map, palette


def _chroma(c):
    """绝对饱和度 = 最大通道 - 最小通道"""
    return int(max(c)) - int(min(c))


def _redmean(a, b):
    """redmean 加权色差，比纯欧氏更接近人眼感受"""
    dr = int(a[0]) - int(b[0])
    dg = int(a[1]) - int(b[1])
    db = int(a[2]) - int(b[2])
    rm = (int(a[0]) + int(b[0])) / 2.0
    return (2 + rm / 256.0) * dr * dr + 4.0 * dg * dg + (2 + (255 - rm) / 256.0) * db * db


def _hue(c):
    """色相角 0-360；无彩色返回 -1"""
    r, g, b = [int(x) / 255.0 for x in c]
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < 0.02:
        return -1.0
    if mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60.0


def _luma(c):
    """亮度 0-255"""
    return 0.299 * int(c[0]) + 0.587 * int(c[1]) + 0.114 * int(c[2])


def _drop_dark_same_hue(cand, palette):
    """同色相的候选里只留最亮的那个。

    文字往往有深色阴影/描边，它和文字填充色相相同但更暗。
    不去掉的话，阴影因为「离背景更远」会被误判成文字色
    （红字 → 提取成 #672C0A 这种暗红）。
    """
    kept = []
    for i in cand:
        hi = _hue(palette[i])
        beaten = False
        for j in cand:
            if i == j:
                continue
            hj = _hue(palette[j])
            if hi >= 0 and hj >= 0 and abs(hi - hj) < 30:
                if _luma(palette[j]) > _luma(palette[i]):
                    beaten = True
                    break
        if not beaten:
            kept.append(i)
    return kept or cand


def _hole_regions(mask, max_side=192):
    """返回 mask 内部被完全包住的区域（布尔图，尺寸同 mask）。

    做法：先用「或池化」降采样（细描边不会被抽样抽断），
    再从图边框往里漫水，漫不到的地方就是被 mask 围住的空洞。
    """
    h, w = mask.shape
    step = int(np.ceil(max(h, w) / float(max_side)))
    hs, ws = h // step, w // step
    if hs < 4 or ws < 4:
        return np.zeros_like(mask, dtype=bool)
    m = mask[: hs * step, : ws * step].reshape(hs, step, ws, step).any(axis=(1, 3))

    border = np.zeros_like(m)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    reach = border & ~m
    # 漫水连「外部」都铺不开（mask 把边框封死了），空洞判断不可信
    if reach.sum() < 0.10 * (~m).sum():
        return np.zeros_like(mask, dtype=bool)
    while True:
        d = reach.copy()
        d[1:, :] |= reach[:-1, :]
        d[:-1, :] |= reach[1:, :]
        d[:, 1:] |= reach[:, :-1]
        d[:, :-1] |= reach[:, 1:]
        d &= ~m
        if np.array_equal(d, reach):
            break
        reach = d

    holes = (~m) & (~reach)
    full = np.repeat(np.repeat(holes, step, axis=0), step, axis=1)
    out = np.zeros((h, w), dtype=bool)
    out[: full.shape[0], : full.shape[1]] = full[:h, :w]
    return out


def _thinness(mask):
    """细线程度：3×3 腐蚀后还剩多少。

    描边/轮廓线只有 1-2 像素宽，腐蚀后接近 0；实心笔画明显大于 0。
    """
    n = int(mask.sum())
    if n == 0:
        return 1.0
    return float(_erode(mask, 3).sum()) / float(n)


def _channel_diff(a, b):
    """单通道最大差，用来判断两个颜色是不是「差不多」"""
    return max(
        abs(int(a[0]) - int(b[0])),
        abs(int(a[1]) - int(b[1])),
        abs(int(a[2]) - int(b[2])),
    )


def _border_ring(rgb, frac=20):
    """图边框一圈。文字框裁得再紧，边框上大部分也是底色"""
    h, w = rgb.shape[:2]
    t = max(2, min(h, w) // frac)
    mask = np.zeros((h, w), dtype=bool)
    mask[:t, :] = True
    mask[-t:, :] = True
    mask[:, :t] = True
    mask[:, -t:] = True
    return mask


def _modal_color(pixels):
    """一片像素的代表色：先按 16 级量化取众数桶，再取桶内中位数"""
    if len(pixels) == 0:
        return None
    q = pixels.astype(np.uint32) >> 4          # 每通道 16 级
    key = ((q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]).astype(np.intp)
    # bincount 直接出直方图，比 unique 少一次排序
    k = int(np.bincount(key, minlength=16384).argmax())
    sel = pixels[key == k]
    return tuple(int(v) for v in np.round(np.median(sel, axis=0)))


def _bg_excluding(rgb, fg, exclude=None):
    """底色 = 排掉白色和文字色之后剩下的主要颜色

    文字框裁得紧时边框上全是文字和描边，直接取边框颜色会把描边当底色。
    """
    # 白到蓝之间的抗锯齿过渡色也很亮，一并排掉
    amax, amin, px = _rgb_aux(rgb)
    keep = (amax < 240) & (amin < 210)
    if fg is not None:
        keep &= np.abs(px - np.array(fg, np.int16)).max(axis=1).reshape(keep.shape) > 40
    if exclude is not None:
        # 描边色也不是底色
        keep &= (
            np.abs(px - np.array(exclude, np.int16)).max(axis=1).reshape(keep.shape) > 45
        )
    if int(keep.sum()) < rgb.shape[0] * rgb.shape[1] * 0.05:
        return _modal_color(rgb[_border_ring(rgb)])
    return _modal_color(rgb[keep])


_AUX_RGB = None      # 上回算的是哪张裁图（持有引用，id 不会被复用）
_AUX_AMAX = None
_AUX_AMIN = None
_AUX_I16 = None


def _rgb_aux(rgb):
    """一张裁图的派生量只算一次：通道最大、通道最小、int16 展平视图

    这三个在取色流程里被十几处判据反复用到。每处各自重算一遍
    rgb.max(axis=2) 的话，单个框要多花一倍时间。
    """
    global _AUX_RGB, _AUX_AMAX, _AUX_AMIN, _AUX_I16
    if _AUX_RGB is not rgb:
        _AUX_RGB = rgb
        _AUX_AMAX = rgb.max(axis=2)
        _AUX_AMIN = rgb.min(axis=2)
        _AUX_I16 = rgb.reshape(-1, 3).astype(np.int16)
    return _AUX_AMAX, _AUX_AMIN, _AUX_I16


def _near_white(rgb):
    """近白掩膜。白色经常被聚类拆成好几个碎簇，直接按像素判更稳"""
    return _rgb_aux(rgb)[1] >= 235


def _near_black(rgb):
    """近黑掩膜。固定描边除了白色就是黑色"""
    return _rgb_aux(rgb)[0] <= 60


def _core_from_mask(rgb, mask, check_thickness=True):
    """描边掩膜围出的那片就是芯。取芯里饱和度最高的三成像素做众数。

    不能直接取众数：白描边旁边一圈抗锯齿过渡色也很亮，
    众数会落到过渡色上，取出来是一抹浅蓝。
    """
    share = float(mask.mean())
    if share < 0.02 or share > 0.85:
        return None
    # 厚度上限只对白色掩膜生效：白色铺满整张图基本就是底色。
    # 黑色铺满则很常见（粗体字的黑描边），不能一刀切。
    if check_thickness and _thinness(mask) >= 0.35:
        return None
    # JPEG 压缩会把细描边啃出 1px 缺口，围不成闭合环，先膨胀接上
    # 阈值放到 0.2%：细描边在 JPEG 下断口多，围出的洞往往很小
    holes = _hole_regions(_dilate(mask, 3))
    if int(holes.sum()) >= max(30, int(mask.size * 0.002)):
        core = _modal_high_chroma(rgb[holes])
        if core is not None:
            return core

    # 洞围不出来时兜底：白层厚说明白色是笔画外形，芯就在它外侧一圈里。
    # 细白描边不能走这条 —— 它外侧全是抗锯齿过渡色。
    if _thinness(mask) > 0.15:
        band = _dilate(mask, 5) & (~mask)
        px = rgb[band]
        keep = px.min(axis=1) < 225
        # 白描边的图里，深色底漏进来的都是近黑，按亮度掐掉
        ring_color = _modal_color(rgb[mask])
        if ring_color is not None and _luma(ring_color) > 150:
            luma = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
            keep &= luma > 45
        bg = _modal_color(rgb[_border_ring(rgb)])
        if bg is not None:
            keep &= np.abs(px.astype(np.int16) - np.array(bg, np.int16)).max(axis=1) > 55
        if int(keep.sum()) >= 20:
            return _modal_high_chroma(px[keep])
    return None


def _modal_high_chroma(px):
    """一片像素里饱和度最高的三成，取众数。

    抗锯齿过渡色亮度高但饱和度低，取众数会落到过渡色上，
    取出来的是一抹浅色而不是真正的芯色。
    """
    keep = px.min(axis=1) < 225
    if int(keep.sum()) < 20:
        return None
    px = px[keep]
    chroma = px.max(axis=1).astype(np.int16) - px.min(axis=1).astype(np.int16)
    sel = px[chroma >= np.percentile(chroma, 70)] if int(chroma.max()) > 30 else px
    return _modal_color(sel)


def _detect_core(rgb, wmask=None, bmask=None):
    """自动判定文字色和描边色。返回 (文字色, 描边色)。

    文字色 = 描边围住的那片
    描边色 = 围住它的那圈颜色
    白描边、黑描边都试一遍。
    """
    # 「白字黑边 / 黑字白边」先直接判：看白（或黑）像素外面紧贴的那圈
    # 是不是另一种极端色。这是最常见的排版，而且不会被气球那圈粉色抢走。
    # 厚白边 + 彩色字：白色可能占一半以上（白边很厚，甚至和气球连成一片），
    # 但里面裹着的那块是有彩色的，那就是文字。这条要排在黑白字判断之前，
    # 否则会被外面零星的黑像素抢成「黑字白边」。
    if wmask is None:
        wmask = _near_white(rgb)
    if bmask is None:
        bmask = _near_black(rgb)
    wshare = float(wmask.mean())
    if 0.25 <= wshare <= 0.85:
        # 只认「白边真的围出来的一块」，不用外侧环带兜底 ——
        # 兜底会把白边外面的墙/气球当成内芯
        holes = _hole_regions(_dilate(wmask, 3))
        if (
            int(holes.sum()) >= max(30, int(wmask.size * 0.002))
            and float(holes.mean()) <= 0.25
        ):
            core = _modal_high_chroma(rgb[holes])
            if core is not None and _chroma(core) >= 40 and _is_spread(rgb, core):
                ring = _modal_color(rgb[wmask])
                if ring is not None:
                    return core, (
                        (255, 255, 255) if min(ring) >= 235 else ring
                    )

    for near, m, want_dark in (
        (_near_white, wmask, True),
        (_near_black, bmask, False),
    ):
        share = float(m.mean())
        if share < 0.03 or share > 0.45:
            continue
        # 看 2~4 像素外的一圈：紧贴的 1 像素全是抗锯齿过渡色
        ring = _dilate(m, 9) & (~_dilate(m, 3))
        if int(ring.sum()) < 20:
            continue
        rpx = rgb[ring]
        if want_dark:
            # 白字外面是不是暗色
            hit = rpx.max(axis=1) <= 90
        else:
            # 黑字外面是不是亮色
            hit = rpx.min(axis=1) >= 200
        if float(hit.mean()) < 0.5:
            continue
        m_color = _modal_color(rgb[m])
        o_color = _modal_color(rpx[hit])
        if m_color is None or o_color is None:
            continue
        return m_color, o_color

    edge_bg = _modal_color(rgb[_border_ring(rgb)])
    cands = []
    for near, mask in ((_near_white, wmask), (_near_black, bmask)):
        core = _core_from_mask(rgb, mask, check_thickness=(near is _near_white))
        if core is None:
            continue
        # 中灰不是真颜色，是笔画和底之间的抗锯齿，丢掉换别的判据
        if _chroma(core) < 25 and 60 < _luma(core) < 200:
            continue
        thin = _thinness(mask)
        # 只对白掩膜做「围出来的是底色」判断：白气球围出的孔确实是底色。
        # 黑环围住的东西一定是文字，哪怕它和气球同色
        if (
            near is _near_white
            and edge_bg is not None
            and _channel_diff(core, edge_bg) <= 90
            and thin > 0.08
        ):
            continue
        ring = _modal_color(rgb[mask])
        if ring is None:
            ring = (0, 0, 0)
        cands.append((core, ring))
    if not cands:
        return None
    # 「白字黑边」「黑字白边」是最常见的排版，先认这两种，
    # 否则会被气球上那圈粉色抢走（粉色饱和度比白字高）
    for core, ring in cands:
        if (min(core) >= 215 and max(ring) <= 90) or (
            max(core) <= 90 and min(ring) >= 215
        ):
            return core, ring
    cands.sort(key=lambda cr: _chroma(cr[0]), reverse=True)
    for core, ring in cands:
        if _chroma(core) >= 25:
            return core, ring
    for core, ring in cands:
        if edge_bg is None or _channel_diff(core, edge_bg) > 60:
            return core, ring
    return cands[0]


def _detect_white_text(rgb, white=None):
    """彩描边包白芯 → 白色就是文字色。返回 (白, 底色)；不成立返回 None。

    第二类写法：描边是彩色（含荧光色），芯是白色。
    判据：白色笔画又厚、又不铺到图边。
      - 白色是底色（黑字压白底）时，白色会一直铺到图边
      - 白色是细描边（彩字白边）时，腐蚀一下白色就没了
    """
    if white is None:
        white = _near_white(rgb)
    share = float(white.mean())
    if share < 0.05 or share > 0.52:
        return None
    thin = _thinness(white)
    if thin < 0.18 or thin > 0.80:
        return None
    # 白色铺到图边就说明白色是底色，不是笔画
    if float(white[_border_ring(rgb)].mean()) > 0.25:
        return None
    fg = _modal_color(rgb[white])
    bg = _modal_color(rgb[~white])
    return fg, bg


def _color_score(color, bg):
    """文字色打分 = 与背景的色差 × 饱和度

    只比色差会选到文字的深色阴影（它离白底最远）；
    只比饱和度会选到背景里的鲜艳物体（橙色木桌比蓝字还艳）。
    两者相乘，纯文字色同时满足「鲜艳」和「离背景远」，才拿得到高分。

    饱和度取 max(chroma, 40)：黑字白字饱和度都是 0，
    直接乘会让黑字被背景里淡淡的木色抢走（黑字 → 提取成棕色）。
    """
    return _redmean(color, bg) * max(_chroma(color), 40)


def _bg_plain(rgb, fg, exclude=None):
    """底色 = 既不是文字色也不是描边色的主要颜色

    指定模式下一步就能算准，不需要 _bg_excluding 那套亮度过滤
    （那套会把白底排掉，只剩抗锯齿灰）。
    """
    keep = np.ones(rgb.shape[:2], dtype=bool)
    px = _rgb_aux(rgb)[2]
    if fg is not None:
        keep &= np.abs(px - np.array(fg, np.int16)).max(axis=1).reshape(keep.shape) > 40
    if exclude is not None:
        keep &= (
            np.abs(px - np.array(exclude, np.int16)).max(axis=1).reshape(keep.shape) > 45
        )
    if int(keep.sum()) < rgb.shape[0] * rgb.shape[1] * 0.05:
        return None
    return _modal_color(rgb[keep])


def _snap_achromatic(rgb, fg, bg):
    """文字色不该落在中灰上

    中灰几乎都是「黑字/白字」与其背景之间的抗锯齿，不是真颜色。
    真出现中灰时，往更近的纯黑或纯白靠；如果和底色撞了，底色换成另一端。
    """
    if fg is None or _chroma(fg) >= 30:
        return fg, bg
    luma = _luma(fg)
    if luma <= 55 or luma >= 195:
        return fg, bg
    blk = _near_black(rgb)
    wht = _near_white(rgb)
    black = _modal_color(rgb[blk]) if blk.any() else None
    white = _modal_color(rgb[wht]) if wht.any() else None
    target = black if luma < 128 else white
    other = white if luma < 128 else black
    if target is None:
        return fg, bg
    if bg is not None and _channel_diff(bg, target) <= 45 and other is not None:
        bg = other
    return target, bg


def _bg_hint(rgb):
    """用图边框的颜色猜底色

    文字框裁得紧时，字很粗会占掉一半以上的像素，按「像素最多 = 底色」
    就会把文字本身当成底色，文字色退化成抗锯齿灰。边框一圈基本还是底色。
    边框上底色占比不到四成时返回 None，交回原来的判断。
    """
    ring = _border_ring(rgb)
    px = rgb[ring]
    bg = _modal_color(px)
    if bg is None:
        return None
    near = (
        np.abs(px.astype(np.int16) - np.array(bg, np.int16)).max(axis=1) <= 45
    ).mean()
    return bg if near >= 0.40 else None


def _detect_flat_text(rgb):
    """没有描边的彩色字压在纯色底上

    这种字没有边可围，聚类又会被抗锯齿过渡色拉散，取出来一会深一会浅。
    直接取「离底色远、饱和度最高」的那一小批像素当文字色 —— 那是笔画芯，
    每张裁图都落在同一档，不会飘。描边色就是底色。
    """
    bg = _bg_hint(rgb)
    if bg is None:
        return None
    px = _rgb_aux(rgb)[2]
    if int(px.shape[0]) < 200:
        return None
    far = np.abs(px - np.array(bg, np.int16)).max(axis=1) > 40
    if float(far.mean()) < 0.02:
        return None
    chroma = px.max(axis=1) - px.min(axis=1)
    thresh = np.percentile(chroma[far], 90)
    sel = px[far & (chroma >= thresh)]
    if len(sel) < 20:
        return None
    fg = _modal_color(sel)
    if fg is None or _channel_diff(fg, bg) <= 25:
        return None

    # 描边 = 紧贴文字外面那一圈。有白边/黑边时它和裁图边框的底色不同，
    # 不能直接拿边框色当描边（粉底黑字带粗白边，描边是白不是粉）
    fgm = (np.abs(px - np.array(fg, np.int16)).max(axis=1) <= 45).reshape(rgb.shape[:2])
    ring = _dilate(fgm, 3) & (~fgm)
    outline = bg
    if int(ring.sum()) >= 20:
        # 只认「白边 / 黑边」这两种真描边，
        # 抗锯齿过渡色（粉字外那圈浅粉）不算，否则会把底色误当成描边
        rpx = rgb[ring]
        n_white = float((rpx.min(axis=1) >= 235).mean())
        n_black = float((rpx.max(axis=1) <= 60).mean())
        # 门槛放到 0.10：框裁得紧时白边只剩一条缝
        if n_white >= 0.10:
            outline = _modal_color(rpx[rpx.min(axis=1) >= 235])
        elif n_black >= 0.10:
            outline = _modal_color(rpx[rpx.max(axis=1) <= 60])
        if outline is None:
            outline = bg
    return fg, outline


def _pick_fg_bg_ex(index_map, palette, bg_hint=None):
    """返回 (文字色, 背景色, 文字簇占比)"""
    n = len(palette)
    if n == 0:
        return None, None, 0.0

    pop = np.bincount(index_map.ravel(), minlength=n)
    valid = np.where(pop > 0)[0]
    if len(valid) < 2:
        # 只有一个有效色：说明框内是纯色（没有文字），不写 fg
        return None, None, 0.0

    total = pop.sum()
    bg_i = int(valid[np.argmax(pop[valid])])
    if bg_hint is not None:
        # 用边框颜色做底色，前提是它确实在调色板里
        near_i = min(
            (i for i in valid),
            key=lambda i: _channel_diff(palette[i], bg_hint),
        )
        if _channel_diff(palette[near_i], bg_hint) <= 45:
            bg_i = near_i
    bg = palette[bg_i]

    cand = [
        i for i in valid
        if i != bg_i and pop[i] >= 20 and pop[i] / total >= 0.001
    ]
    if not cand:
        cand = [i for i in valid if i != bg_i]
    if not cand:
        return None, None, 0.0

    # 去掉同色相的暗色版本（文字阴影/描边）
    cand = _drop_dark_same_hue(cand, palette)

    best_i = max(cand, key=lambda i: _color_score(palette[i], bg))
    return (
        tuple(int(v) for v in palette[best_i]),
        tuple(int(v) for v in bg),
        float(pop[best_i]) / float(total),
    )


def pick_fg_bg(index_map, palette):
    """从聚类结果里判定「文字色」和「背景色」。

    参数顺序与 pix_color_segment 的返回值一致：(index_map, palette)。

    判定规则：
      1. 像素数最多的簇 = 背景
      2. 候选 = 除背景外、像素占比 >= 0.1% 的簇（滤掉噪点）
      3. 去掉同色相的暗色版本（文字阴影/描边）
      4. 候选里按「色差 × 饱和度」打分，取最高分
    """
    fg, bg, _ = _pick_fg_bg_ex(index_map, palette)
    return fg, bg


def _white_text_color(rgb, seg):
    """白字彩边：笔画是白色，外面套一圈彩色或荧光描边。

    这类字白色笔画又厚又不贴图边：
      - 白色是底色（黑字压白底）时白色会一直铺到图边
      - 白色是细描边（彩字白边）时白色腐蚀一下就没了
    返回 (白色, 背景色, None)；条件不成立返回 None。
    """
    h, w = rgb.shape[:2]
    near = _rgb_aux(rgb)[1] >= 235
    share = float(near.mean())
    if share < 0.05 or share > 0.52:
        return None
    if _thinness(near) < 0.18:
        return None
    b = max(2, min(h, w) // 20)
    edge = np.zeros((h, w), dtype=bool)
    edge[:b, :] = True
    edge[-b:, :] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    if float(near[edge].mean()) > 0.06:
        return None

    idx, pal = seg
    pop = np.bincount(idx.ravel(), minlength=len(pal))
    best_bg = None
    for i in range(len(pal)):
        if pop[i] == 0:
            continue
        if int(pal[i].min()) >= 235:
            continue
        if best_bg is None or pop[i] > best_bg[0]:
            best_bg = (int(pop[i]), i)
    bg = tuple(int(v) for v in pal[best_bg[1]]) if best_bg else (255, 255, 255)

    white = np.median(rgb[near], axis=0)
    fg = tuple(int(v) for v in np.round(white))
    return fg, bg, None


def _mask_near(rgb, color, tol=70):
    """和指定颜色足够接近的像素"""
    diff = np.abs(_rgb_aux(rgb)[2] - np.array(color, np.int16)).max(axis=1)
    return diff.reshape(rgb.shape[:2]) <= tol


def _fill_share(rgb, color, tol=40):
    """这个颜色占了整块的多少像素。铺得太满的多半是底色，不是字"""
    px = _rgb_aux(rgb)[2]
    hit = np.abs(px - np.array(color, np.int16)).max(axis=1) <= tol
    return float(hit.mean())


def _on_border(rgb, color, tol=40, frac=None):
    """这个颜色是不是铺到图边框上

    铺到边框的通常是背景（墙、木板、气球外沿），不是文字。
    """
    if frac is None:
        frac = TH["border_frac"]
    px = rgb[_border_ring(rgb)]
    if len(px) == 0:
        return False
    hit = np.abs(px.astype(np.int16) - np.array(color, np.int16)).max(axis=1) <= tol
    return float(hit.mean()) > frac


def _extract_text_by_outline(rgb, outline_color):
    """模式1：描边颜色固定（白/黑），文字就是描边以外那批笔画像素

    不靠「洞」找芯。竖排小字的描边只有两三像素宽，围不成闭合环，
    _hole_regions 在这种窄框上会整体失效 —— 一整页「黑字白边」
    就会全部判成白字。改成按距离取：
      有彩色芯 → 取最艳那档（但要铺得开、不贴图边，否则取到的是背景）
      没有彩色芯 → 取最暗那档（旁白 / 心理描写的纯黑字）
    """
    oc = tuple(int(v) for v in outline_color)
    if min(oc) >= 235 and _chroma(oc) < 32:
        mask = _near_white(rgb)
    elif max(oc) <= 60:
        mask = _near_black(rgb)
    else:
        mask = _mask_near(rgb, oc, tol=55)

    # 指定的描边色在这张图里几乎不存在（比如填了黑色，但这页描边其实是紫的）
    # → 这个模式不适用，交回自动判定，不要硬套
    if float(mask.mean()) < 0.01:
        return None

    px = rgb[~mask]
    if len(px) < 30:
        return None
    px16 = px.astype(np.int16)
    ch = px16.max(axis=1) - px16.min(axis=1)

    # 1) 有彩色芯：从最艳的一档往下试，跳过贴在边框上的背景色
    if int(ch.max()) >= 55:
        for pct in (95, 85, 70, 50):
            thr = np.percentile(ch, pct)
            if thr < 45:
                continue
            c = _modal_color(px[ch >= thr])
            if c is None or _channel_diff(c, oc) <= 30:
                continue
            if _on_border(rgb, c):
                continue
            if _fill_share(rgb, c) > 0.45:
                continue
            if _is_spread(rgb, c):
                return c

    # 2) 没有彩色芯 → 旁白那种纯黑字。
    # 这一步不排底色：黑字本来就比深色背景更黑，直接取最暗那档最稳。
    luma = 0.299 * px16[:, 0] + 0.587 * px16[:, 1] + 0.114 * px16[:, 2]
    dark = px[luma <= np.percentile(luma, 15)]
    c = _modal_color(dark)
    if c is not None and _channel_diff(c, oc) > 30:
        return _snap_extreme(c)
    return None


def _extract_outline_by_core(rgb, core_color):
    """模式2：文字芯颜色固定（白/黑），要的是紧贴着芯的那圈描边色。

    描边一定是紧贴文字的，所以只取芯外侧一个像素的一圈，
    再往外就是气泡底色了。
    """
    mask = _mask_near(rgb, core_color)
    share = float(mask.mean())
    if share < 0.01 or share > 0.85:
        return None
    ring = _dilate(mask, 3) & (~mask)
    if int(ring.sum()) < 30:
        return None
    rp = rgb[ring]
    keep = np.abs(rp.astype(np.int16) - np.array(core_color, np.int16)).max(axis=1) > 50
    if int(keep.sum()) < 20:
        return None
    return _modal_color(rp[keep])


def _top_chroma_color(rgb, pct=97):
    """整块里饱和度最高的那一档颜色"""
    px = rgb.reshape(-1, 3).astype(np.int16)
    ch = px.max(axis=1) - px.min(axis=1)
    sel = px[ch >= np.percentile(ch, pct)]
    if len(sel) < 30:
        return None
    return _modal_color(sel)


def _rescue_chromatic(rgb, fg, outline):
    """结果是黑/白，但整块里明明有一档很艳的颜色 → 那才是文字色

    厚白边 + 深蓝芯压在黑底上最容易这样：白和黑各自成片，
    芯只占很小一块，被黑白判据抢先了。
    """
    if fg is not None and _chroma(fg) >= 20:
        return fg, outline
    strong = _top_chroma_color(rgb)
    if strong is None or _chroma(strong) < 70:
        return fg, outline
    if outline is not None and _channel_diff(strong, outline) <= 30:
        return fg, outline
    # 这档颜色如果铺到图边框上，说明它是背景（墙、气球），不是文字
    near = (
        np.abs(_rgb_aux(rgb)[2] - np.array(strong, np.int16)).max(axis=1)
        .reshape(rgb.shape[:2])
        <= 40
    )
    if float(near[_border_ring(rgb)].mean()) > 0.02:
        return fg, outline
    return strong, outline


def _prefer_strong_chroma(rgb, fg):
    """文字色如果不够艳，而整块里还有明显更艳的一档，就用更艳的那档

    白描边围出来的区域里常混进别的浅色（背景的缎带、反光），
    它比文字本身更"白一点"，会被当成芯。真正写字的那档饱和度更高。
    """
    if fg is None or _chroma(fg) < 25:
        return fg
    px = rgb.reshape(-1, 3).astype(np.int16)
    ch = px.max(axis=1) - px.min(axis=1)
    sel = px[ch >= np.percentile(ch, 97)]
    if len(sel) < 50:
        return fg
    strong = _modal_color(sel)
    if strong is None or not _is_spread(rgb, strong):
        return fg
    # 更艳的那档如果铺在框边，那是木纹/墙一类的背景，换过去就把字色带到背景上了
    if _on_border(rgb, strong) or _fill_share(rgb, strong) > 0.45:
        return fg
    if _chroma(strong) < _chroma(fg) * 1.3 or _channel_diff(strong, fg) <= 30:
        return fg
    # 只有色相不同才换。同一个色相只是深浅差别 —— 那是浅粉字被画面冲淡了，
    # 换成更艳的同色相会把 F5AFC9 变成 FF7CB8。
    hs = _hue(strong)
    hf = _hue(fg)
    if hs >= 0 and hf >= 0 and min(abs(hs - hf), 360 - abs(hs - hf)) <= 40:
        return fg
    return strong


def _is_spread(rgb, color, min_span=0.30):
    """这个颜色是不是铺开在整个框里

    ❤ ★ 这类符号颜色很艳但只占一小块，拿它当文字色会把整句白字带偏
    """
    if color is None:
        return False
    px = _rgb_aux(rgb)[2]
    near = np.abs(px - np.array(color, np.int16)).max(axis=1) <= 40
    rows = near.reshape(rgb.shape[:2])
    ys = np.flatnonzero(rows.any(axis=1))
    if len(ys) == 0:
        return False
    xs = np.flatnonzero(rows.any(axis=0))
    h, w = rgb.shape[:2]
    span = float((ys[-1] - ys[0] + 1) * (xs[-1] - xs[0] + 1)) / (h * w)
    return span >= min_span


def _fix_outline_by_ring(rgb, fg, outline):
    """用文字芯周围的局部环带复核描边，避免把远处背景当成描边。

    关键变化：
      1. 不再只处理高饱和彩色文字，黑字、白字、浅色字也参与复核。
      2. 先估计图框边缘背景，再从文字周围环带中排除背景色。
      3. 只有紧贴文字、占环带有足够比例、且明显不同于背景的颜色，
         才能推翻原描边结果；抗锯齿过渡色不会轻易成为描边。
    """
    if fg is None:
        return outline

    px = _rgb_aux(rgb)[2]
    fg16 = np.asarray(fg, np.int16)
    # 用较窄容差建立文字芯掩膜，避免把整圈抗锯齿都算进文字本身。
    m = (np.abs(px - fg16).max(axis=1) <= 32).reshape(rgb.shape[:2])
    if int(m.sum()) < 12:
        return outline

    ring = _dilate(m, 3) & (~m)
    if int(ring.sum()) < 12:
        return outline

    rpx = rgb[ring].astype(np.int16)
    # 框边背景是最容易被误认成描边的颜色。先从局部环带中剔除它。
    bg = _bg_hint(rgb)
    keep = np.ones(len(rpx), dtype=bool)
    if bg is not None and _channel_diff(bg, fg) > 25:
        keep &= (np.abs(rpx - np.asarray(bg, np.int16)).max(axis=1) > 32)

    # 去掉落在「文字色—背景色」混色线上的抗锯齿像素。
    if bg is not None:
        f = np.asarray(fg, np.float32)
        b = np.asarray(bg, np.float32)
        v = b - f
        vv = float(np.dot(v, v))
        if vv > 1.0:
            q = rpx.astype(np.float32) - f
            t = (q @ v) / vv
            proj = f + np.clip(t, 0.0, 1.0)[:, None] * v
            dist = np.linalg.norm(rpx.astype(np.float32) - proj, axis=1)
            keep &= ~((t > 0.04) & (t < 0.96) & (dist < 18.0))

    cand_px = rpx[keep]
    if len(cand_px) < 8:
        return outline

    candidate = _modal_color(cand_px)
    if candidate is None or _channel_diff(candidate, fg) <= 28:
        return outline

    # 候选色必须在近距离环带里形成真实色块，而不是零散噪点。
    hit = np.abs(rpx - np.asarray(candidate, np.int16)).max(axis=1) <= 28
    coverage = float(hit.sum()) / float(max(1, len(rpx)))
    if coverage < 0.16:
        return outline

    # 黑/白描边是漫画中最常见的两类，允许较低覆盖率以保留细描边。
    # 必须在排除背景后的 cand_px 上统计；否则白底上的蓝字会把白底误判成白描边。
    if len(cand_px) >= 8 and float((cand_px.min(axis=1) >= 235).mean()) >= 0.18:
        return (255, 255, 255)
    if len(cand_px) >= 8 and float((cand_px.max(axis=1) <= 60).mean()) >= 0.18:
        return (0, 0, 0)

    # 彩色描边必须同时满足：局部环带占比足够，且不是边框背景色。
    if bg is not None and _channel_diff(candidate, bg) <= 32:
        return outline
    return candidate


def _snap_extreme(color):
    """接近纯白/纯黑的直接归到 255/0

    JPEG 会把 255,255,255 压成 255,237,251 这种带色的浅色，
    同一页不同裁图还会给出不同的偏差值，看着像"深浅不一"。
    描边是白就是白，统一成纯白更好比对。
    """
    if color is None:
        return None
    c = tuple(int(v) for v in color)
    if min(c) >= 235:
        return (255, 255, 255)
    if max(c) <= 45:
        return (0, 0, 0)
    return c


def _too_close_to_bg(fg, bg):
    """文字色和底色几乎一样 → 这次取色是失败的

    取成「和背景一个颜色」等于没取，写进标注只会误导下游。
    """
    if fg is None or bg is None:
        return True
    return _channel_diff(fg, bg) <= 25


def pick_color_by_mode(rgb, mode, fixed_color, maxdist_base=90, maxcolors=12,
                       selsize=1, finalcolors=6):
    """取文字色和文字描边色。返回 (文字色, 描边色, 实际用到的颜色距离)

    属性只有两个：文字颜色 fg、文字描边颜色 bg。三种上色模式：

      0 自动判断   —— 两个都从图里提
      1 描边固定   —— 描边写死成设定色，只提文字色
      2 字色固定   —— 文字写死成设定色，只提描边色
    """
    if mode == 1:
        # 描边固定：不管图里描边是什么颜色，一律写设定的那个值
        fg = _extract_text_by_outline(rgb, fixed_color)
        if fg is None:
            fg = pick_color_robust(
                rgb,
                maxdist_base=maxdist_base,
                maxcolors=maxcolors,
                selsize=selsize,
                finalcolors=finalcolors,
            )[0]
        if fg is not None:
            return _snap_extreme(fg), tuple(int(v) for v in fixed_color), maxdist_base
    elif mode == 2:
        # 字色固定：不管图里文字是什么颜色，一律写设定的那个值
        outline = _extract_outline_by_core(rgb, fixed_color)
        if outline is None:
            outline = pick_color_robust(
                rgb,
                maxdist_base=maxdist_base,
                maxcolors=maxcolors,
                selsize=selsize,
                finalcolors=finalcolors,
            )[1]
        if outline is not None:
            return (
                tuple(int(v) for v in fixed_color),
                _snap_extreme(outline),
                maxdist_base,
            )

    fg, outline, md = pick_color_robust(
        rgb,
        maxdist_base=maxdist_base,
        maxcolors=maxcolors,
        selsize=selsize,
        finalcolors=finalcolors,
    )
    fg, outline = _snap_achromatic(rgb, fg, outline)
    fg, outline = _snap_extreme(fg), _snap_extreme(outline)
    return fg, outline, md


def _border_frac(mask, rgb):
    """掩膜落在图边框一圈里的比例，取四条全宽边（含角）"""
    h, w = mask.shape
    t = max(2, min(h, w) // 10)
    ring = np.zeros((h, w), dtype=bool)
    ring[:t, :] = True
    ring[-t:, :] = True
    ring[:, :t] = True
    ring[:, -t:] = True
    tot = int(ring.sum())
    return float((mask & ring).sum()) / float(tot) if tot else 0.0


def _white_is_outline(rgb):
    """白色是文字的描边，还是文字本身

    两个量实测能把白字黑底 / 白字黑边 / 黑字白边 / 粉字白边分开：
      白占比 < 20%             → 白是笔画（白字黑底那种小字）
      白铺到框边（>50%）        → 白是笔画（真白字的裁框一定切在笔画上）

    剩下的「白占比高、又不碰框边」有两种，要看黑在哪：
      黑不贴边（笔画在框中间） → 黑是字，白是描边
      黑只出现在框边（描边）    → 白是字，黑是描边
      压根没有真黑             → 白是字（白字压在木纹/深背景上）
    """
    wm = _near_white(rgb)
    if float(wm.mean()) < 0.20:
        return False
    if _border_frac(wm, rgb) > 0.50:
        return False
    return True


def _black_rings_core(rgb):
    """黑描边把字芯围在中间 → 返回 (黑围住的那片的众数色, 黑) 或 None

    看黑掩膜自己围出来的洞有多大 —— 黑是描边时洞就是字芯，
    实测洞占比 0.03~0.08；「黑字白边」的黑是笔画，洞只有 0.006~0.019。
    字芯不一定是白的，浅粉 / 浅青 / 浅紫都有，所以颜色从洞里取而不是写死。
    """
    wm = _near_white(rgb)
    bm = _near_black(rgb)
    if float(wm.mean()) < TH["white_min"] or float(bm.mean()) < 0.05 or float(bm.mean()) > 0.40:
        return None
    holes = _hole_regions(_dilate(bm, 3))
    if float(holes.mean()) < TH["hole_min"]:
        return None
    core = _modal_color(rgb[holes])
    if core is None:
        return None
    # 洞里取到灰蒙蒙的过渡色说明这个「洞」不是字芯，别用
    if _chroma(core) < 25 and 80 < _luma(core) < 220:
        return None
    return core


def _color_core_with_dark_outline(rgb):
    """彩色字芯 + 黑描边 → 返回字芯颜色，或 None

    竖排字的黑描边逐个字包，围不成闭合环（洞判据在这里是 0），
    只能反过来找：整块里最鲜艳、铺得开、又不贴框边的那档彩色就是字芯。
    黑字白边那种整块只有黑白，最艳的一档 chroma 低于 70，进不来。
    """
    wm = _near_white(rgb)
    bm = _near_black(rgb)
    sw = float(wm.mean())
    sb = float(bm.mean())
    if sw < TH["white_min"] or sw > 0.68 or sb < 0.05 or sb > 0.22:
        return None
    c = _top_chroma_color(rgb)
    if c is None or _chroma(c) < TH["chroma_min"]:
        return None
    if _on_border(rgb, c) or not _is_spread(rgb, c):
        return None
    # 字芯一定紧挨着黑描边；桌子、墙那种彩色不挨黑
    mask = (
        np.abs(_rgb_aux(rgb)[2] - np.array(c, np.int16)).max(axis=1) <= 40
    ).reshape(rgb.shape[:2])
    n = int(mask.sum())
    if n == 0:
        return None
    if float((_dilate(mask, 3) & bm).sum()) / float(n) < TH["adj_black"]:
        return None
    return c


def pick_color_robust(rgb, maxdist_base=90, maxcolors=12, selsize=1, finalcolors=6):
    """自动判定：返回 (文字色, 文字描边色, 距离)

    先走新引擎 color_extract_engine：把框里的纯色找出来，再按“谁把谁包在里面”
    定角色 —— 被包住的那块是字，包着它的那圈是描边。

    新引擎说不出个所以然（框里只有一种颜色之类）时，才落到下面这套老办法兜底。
    """
    try:
        _fg, _outline = _qu_zise_xin(rgb)
    except Exception:
        _fg, _outline = None, None
    if _fg is not None:
        return _fg, _outline, maxdist_base

    wmask = _near_white(rgb)
    bmask = _near_black(rgb)

    # 黑描边围着浅色字芯（浅粉/浅青/浅紫，小图上看着像白）先判
    core = _black_rings_core(rgb)
    if core is not None:
        return _snap_extreme(core), (0, 0, 0), maxdist_base

    # 竖排字那种黑描边闭合不了 → 反过来取最鲜艳、铺得开、不贴边的彩色字芯。
    # 只在 _detect_core 给不出彩色时才用它：_detect_core 判成彩色说明字芯本来就
    # 找到了，再拿「最艳的一档」去替换只会把浅粉换成艳粉。
    core = _color_core_with_dark_outline(rgb)
    if core is not None:
        _dc = _detect_core(rgb, wmask, bmask)
        if _dc is None or _dc[0] is None or _chroma(_dc[0]) < 40:
            return _snap_extreme(core), (0, 0, 0), maxdist_base

    wmask = _near_white(rgb)
    bmask = _near_black(rgb)

    got = _detect_core(rgb, wmask, bmask)
    if got is not None and got[0] is not None:
        fg0, bg0 = got[0], got[1] or (0, 0, 0)
        # 判成「彩色字 + 白描边」，但那个字色铺到了框边 —— 它是木纹/墙一类的背景，
        # 白才是字。木纹底上压白字最容易被判反。
        if _luma(bg0) >= 200 and _chroma(bg0) < 40 and _on_border(rgb, fg0):
            got_w = _detect_white_text(rgb, wmask)
            if got_w is not None and got_w[0] is not None:
                fg0, bg0 = got_w[0], got_w[1] or (0, 0, 0)
        # 取到的就是边框那一圈的颜色 —— 那是背景（木板、墙、气球外沿），不是字。
        # 暖色背景特别容易被各种判据抢成字色，这条兜一下。
        hint = _bg_hint(rgb)
        if hint is not None and _channel_diff(fg0, hint) <= 35:
            core = _extract_text_by_outline(rgb, bg0)
            # 只在重判结果是「真墨色」（够黑或够艳）时才采纳，
            # 否则换成一个灰蒙蒙的过渡色，还不如原来的白
            if core is not None and (
                _channel_diff(core, hint) > 35
                and (_luma(core) < 80 or _chroma(core) >= 40)
            ):
                fg0 = core
        # 判成「白字」，但白色其实是描边（白占比高、又不铺框边）→ 按描边取芯重算。
        # 整页黑字白边就是栽在这里：白外面一圈是深色背景，被当成「白字深边」。
        if (
            _luma(fg0) >= 200
            and _chroma(fg0) < 40
            and float(wmask.mean()) >= 0.20
            and _white_is_outline(rgb)
        ):
            core = _extract_text_by_outline(rgb, (255, 255, 255))
            if core is not None and _luma(core) < 200:
                return _snap_extreme(core), (255, 255, 255), maxdist_base
        # 取到的是暗浊色（饱和度不低但发黑），基本是背景或描边，不是字色。
        # 这时如果白占了不小比例，按「白字 + 深色描边」重判一次。
        # 白本身就是描边时不重判 —— 否则刚取对的黑字会被翻回白色。
        if (
            _luma(fg0) < 110
            and 0.15 <= float(wmask.mean()) <= 0.52
            and not _white_is_outline(rgb)
        ):
            got_w = _detect_white_text(rgb, wmask)
            if got_w is not None and got_w[0] is not None:
                fg0, bg0 = got_w[0], got_w[1] or (0, 0, 0)
        fg0, bg0 = _rescue_chromatic(rgb, fg0, bg0)
        fg0 = _prefer_strong_chroma(rgb, fg0)
        return fg0, _fix_outline_by_ring(rgb, fg0, bg0), maxdist_base

    # _detect_core 什么都没给出来，但这确实是「白描边」排版 → 按描边取芯
    if _white_is_outline(rgb):
        core = _extract_text_by_outline(rgb, (255, 255, 255))
        if core is not None and _luma(core) < 200:
            return _snap_extreme(core), (255, 255, 255), maxdist_base

    got = _detect_white_text(rgb, wmask)
    if got is not None and got[0] is not None:
        return got[0], got[1] or (0, 0, 0), maxdist_base

    # 没有描边的彩色字压纯色底
    got = _detect_flat_text(rgb)
    if got is not None:
        return got[0], got[1], maxdist_base

    bg_hint = _bg_hint(rgb)
    best = None
    for factor in (1.0, 0.6, 0.8, 1.3):
        md = max(5, int(maxdist_base * factor))
        res = pix_color_segment(
            rgb,
            maxdist=md,
            maxcolors=maxcolors,
            selsize=selsize,
            finalcolors=finalcolors,
        )
        if res is None:
            continue
        idx, pal = res
        fg, bg, _ = _pick_fg_bg_ex(idx, pal, bg_hint)
        if fg is None:
            continue
        # 尺度之间用统一打分比较，不能只比饱和度 ——
        # 黑字饱和度 5 会输给抗锯齿灰字的 8，黑字就被提取成灰。
        score = _color_score(fg, bg)
        if best is None or score > best[0]:
            best = (score, fg, bg, md)
        if _chroma(fg) >= 120:
            break
    if best is None:
        return None, None, None
    fg, outline = _snap_achromatic(rgb, best[1], best[2])
    return fg, outline, best[3]


def _load_exclude_labels(parent):
    """从主配置 xanylabeling_config.ini 里读排除标签"""
    cfg = getattr(parent, "_config", None)
    if isinstance(cfg, dict):
        return str(cfg.get("color_extract_exclude_labels", "") or "")
    return ""


def _save_exclude_labels(parent, text):
    """写回主配置，和软件其它设置同一个文件、同一套保存逻辑"""
    cfg = getattr(parent, "_config", None)
    if not isinstance(cfg, dict):
        return
    cfg["color_extract_exclude_labels"] = text
    try:
        from anylabeling.config import save_config

        save_config(cfg)
    except Exception:
        pass


def _tight_btn(btn, text=None):
    """按钮等分撑满整行

    不手算宽度（CJK 取整误差会把字挤出去），也不用固定宽度
    （会留下右边一大片空白）。同一行的按钮均分宽度，看起来是整齐的一排。
    """
    btn.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
    )


def _hex(c):
    """(r,g,b) → '#RRGGBB'"""
    return "#%02X%02X%02X" % (int(c[0]), int(c[1]), int(c[2]))


def qimage_to_rgb(img):
    """QImage → (H, W, 3) uint8 RGB 数组"""
    if img is None or img.isNull():
        return None
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = img.width(), img.height()
    if w <= 0 or h <= 0:
        return None
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, img.bytesPerLine()))
    return buf[:, : w * 3].reshape((h, w, 3)).copy()


def _points_bbox(points, img_w, img_h):
    """[[x, y], ...] → 轴对齐外接矩形，裁到图像范围内"""
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    x0, y0 = np.floor(pts[:, :2].min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts[:, :2].max(axis=0)).astype(int)
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def shape_bbox(shape, img_w, img_h):
    """取 shape 的轴对齐外接矩形（旋转框也能用），并裁到图像范围内"""
    pts = np.array([[p.x(), p.y()] for p in shape.points], dtype=np.float64)
    if len(pts) < 2:
        return None
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# 对话框
# ---------------------------------------------------------------------------

class ColorExtractDialog(QtWidgets.QDialog):
    """颜色提取：对矩形框内的文字取色，写入 attributes["fg"] / ["bg"]"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._pending_rows = []     # 待写入表格的结果
        self._batching = False      # 批量执行中：表格延后到 flush 再画
        self.setWindowTitle(self.tr("颜色提取"))
        self.setModal(False)
        self.resize(430, 540)
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.init_ui()
        self.fit_to_content()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ---- 范围 ----
        scope_group = QtWidgets.QGroupBox(self.tr("处理范围"))
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        scope_layout.setContentsMargins(6, 4, 6, 4)
        scope_layout.setSpacing(3)

        scope_row = QtWidgets.QHBoxLayout()
        scope_row.setSpacing(4)
        self.scope_btn_group = QtWidgets.QButtonGroup(self)
        self.scope_btns = []
        for idx, text in enumerate(
            [
                self.tr("提取选中"),
                self.tr("提取本页"),
                self.tr("提取全部"),
            ]
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            _tight_btn(btn, text)
            self.scope_btn_group.addButton(btn, idx)
            self.scope_btns.append(btn)
            scope_row.addWidget(btn)
        self.scope_btns[1].setChecked(True)
        self.scope_btns[0].setToolTip(self.tr("本页已选中的框"))
        self.scope_btns[1].setToolTip(self.tr("本页所有框"))
        self.scope_btns[2].setToolTip(self.tr("任务内所有图片"))
        scope_layout.addLayout(scope_row)

        # ---- 页码范围 ----
        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(4)
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setPrefix(self.tr("从: "))
        self.start_spin.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.end_spin = QtWidgets.QSpinBox()
        self.end_spin.setPrefix(self.tr("到: "))
        self.end_spin.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.range_btn = QtWidgets.QPushButton(self.tr("执行范围"))
        self.range_btn.setMinimumHeight(26)
        _tight_btn(self.range_btn, self.tr("执行范围"))
        self.range_btn.setToolTip(self.tr("按页码区间批量处理"))
        self.range_btn.clicked.connect(self.run_range)
        range_row.addWidget(self.start_spin)
        range_row.addWidget(self.end_spin)
        range_row.addWidget(self.range_btn)
        scope_layout.addLayout(range_row)

        self.exclude_input = QtWidgets.QLineEdit()
        self.exclude_input.setPlaceholderText(self.tr("多个标签用逗号分隔，例如 other, bubble"))
        self.exclude_input.setToolTip(self.tr("这些标签的框跳过"))
        # 排除标签是整部漫画固定的，存进配置文件，重启不用重填
        self.exclude_input.setText(_load_exclude_labels(self.parent))
        self.exclude_input.editingFinished.connect(self.save_exclude_labels)
        exclude_row = QtWidgets.QHBoxLayout()
        exclude_row.setSpacing(4)
        exclude_row.addWidget(QtWidgets.QLabel(self.tr("排除标签:")))
        self.exclude_input.setFixedWidth(150)
        exclude_row.addWidget(self.exclude_input)
        exclude_row.addStretch()
        scope_layout.addLayout(exclude_row)

        self.only_empty_check = QtWidgets.QCheckBox(self.tr("只处理还没有文字颜色的框"))
        self.only_empty_check.setChecked(False)
        scope_layout.addWidget(self.only_empty_check)

        layout.addWidget(scope_group)

        # ---- 提取模式 ----
        mode_group = QtWidgets.QGroupBox(self.tr("提取模式"))
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(6, 4, 6, 4)
        mode_layout.setSpacing(3)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_btn_group = QtWidgets.QButtonGroup(self)
        self.mode_btns = []
        for idx, text in enumerate(
            [
                self.tr("自动判断"),
                self.tr("描边固定"),
                self.tr("字色固定"),
            ]
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            _tight_btn(btn, text)
            self.mode_btn_group.addButton(btn, idx)
            self.mode_btns.append(btn)
            mode_row.addWidget(btn)
        self.mode_btns[0].setChecked(True)
        self.mode_btns[0].setToolTip(
            self.tr("两类描边都自动试一遍，都不成立再退回聚类打分")
        )
        self.mode_btns[1].setToolTip(
            self.tr(
                "描边颜色固定（白或黑），文字是描边包住的那片彩色芯。\n"
                "下面「描边色」用这个颜色。"
            )
        )
        self.mode_btns[2].setToolTip(
            self.tr(
                "文字颜色固定（白或黑），外面套一圈彩色描边。\n"
                "下面「字色」用这个颜色。"
            )
        )
        mode_layout.addLayout(mode_row)

        self._outline_color = (255, 255, 255)
        self._text_color = (255, 255, 255)

        fixed_row = QtWidgets.QHBoxLayout()
        fixed_row.setSpacing(4)
        fixed_row.addWidget(QtWidgets.QLabel(self.tr("描边色")))
        self.outline_color_btn = QtWidgets.QPushButton()
        self.outline_color_btn.setFixedSize(QtCore.QSize(76, 26))
        self.outline_color_btn.setToolTip(self.tr("「描边固定」模式下用这个颜色"))
        self.outline_color_btn.clicked.connect(self.pick_outline_color)
        fixed_row.addWidget(self.outline_color_btn)

        fixed_row.addSpacing(10)
        fixed_row.addWidget(QtWidgets.QLabel(self.tr("字色")))
        self.text_color_btn = QtWidgets.QPushButton()
        self.text_color_btn.setFixedSize(QtCore.QSize(76, 26))
        self.text_color_btn.setToolTip(self.tr("「字色固定」模式下用这个颜色"))
        self.text_color_btn.clicked.connect(self.pick_text_color)
        fixed_row.addWidget(self.text_color_btn)

        fixed_row.addStretch()
        mode_layout.addLayout(fixed_row)
        self.refresh_fixed_color_btns()

        layout.addWidget(mode_group)

        # 参数和阈值都不再需要人工调 —— 取色改为按「谁包住谁」判定，
        # 跟颜色占比一点关系都没有。这里留一句说明，省得以后有人找不到。

        # ---- 按钮 ----
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.run_btn = QtWidgets.QPushButton(self.tr("开始提取"))
        self.run_btn.setAutoDefault(False)
        self.run_btn.setDefault(False)
        self.run_btn.setMinimumHeight(28)
        _tight_btn(self.run_btn, self.tr("开始提取"))
        self.run_btn.clicked.connect(self.run_extract)
        btn_row.addWidget(self.run_btn)

        self.clear_btn = QtWidgets.QPushButton(self.tr("清除颜色"))
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setDefault(False)
        self.clear_btn.setMinimumHeight(28)
        _tight_btn(self.clear_btn, self.tr("清除颜色"))
        self.clear_btn.clicked.connect(self.clear_selected_colors)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        # ---- 日志 ----
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            [self.tr("项目"), self.tr("颜色"), self.tr("色值")]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(130)
        self.table.setMinimumWidth(300)
        self.table.setShowGrid(True)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        head.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(1, 56)
        layout.addWidget(self.table)

    def _paint_color_btn(self, btn, color):
        fg_text = "#000000" if _luma(color) > 140 else "#FFFFFF"
        btn.setText(_hex(color))
        btn.setStyleSheet(
            f"QPushButton{{background:{_hex(color)};color:{fg_text};"
            f"border:1px solid #888;}}"
        )

    def refresh_fixed_color_btns(self):
        self._paint_color_btn(self.outline_color_btn, self._outline_color)
        self._paint_color_btn(self.text_color_btn, self._text_color)

    def pick_outline_color(self):
        c = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self._outline_color), self, self.tr("选择描边颜色")
        )
        if c.isValid():
            self._outline_color = (c.red(), c.green(), c.blue())
            self.refresh_fixed_color_btns()

    def pick_text_color(self):
        c = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self._text_color), self, self.tr("选择文字颜色")
        )
        if c.isValid():
            self._text_color = (c.red(), c.green(), c.blue())
            self.refresh_fixed_color_btns()

    # ------------------------------------------------------------------
    # 取框
    # ------------------------------------------------------------------

    def _exclude_labels(self):
        """解析「排除标签」输入框，返回标签名集合"""
        text = self.exclude_input.text().strip()
        if not text:
            return set()
        for ch in ("，", "、", "；", ";", "\n", "\t"):
            text = text.replace(ch, ",")
        return {x.strip() for x in text.split(",") if x.strip()}

    def fit_to_content(self):
        """窗口宽度按分组内容算，日志框不参与（它默认 hint 256 会把窗口撑宽）"""
        lay = self.layout()
        if lay is None:
            return
        lay.activate()
        width = 0
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.widget() is not None and not isinstance(
                item.widget(), QtWidgets.QTextEdit
            ):
                width = max(width, item.widget().sizeHint().width())
        m = lay.contentsMargins()
        self.resize(width + m.left() + m.right(), self.height())

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_page_range()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.save_exclude_labels()
        self._clear_output()

    def save_exclude_labels(self):
        """排除标签写进配置文件，下次打开自动带出来"""
        _save_exclude_labels(self.parent, self.exclude_input.text())

    def _scope(self):
        return self.scope_btn_group.checkedId()

    def _mode(self):
        return self.mode_btn_group.checkedId()

    def _mode_text(self):
        btn = self.mode_btn_group.checkedButton()
        return btn.text() if btn is not None else ""

    def _collect_shapes(self):
        """按当前范围收集要处理的框。

        不按 shape_type 过滤 —— 这个项目里框可能是 rectangle，
        也可能是四点 polygon 或旋转框，按类型过滤会把它们全漏掉。
        只要求能算出外接矩形即可。
        """
        canvas = getattr(self.parent, "canvas", None)
        if canvas is None:
            return []
        if self._scope() == 0:
            shapes = list(getattr(canvas, "selected_shapes", []) or [])
        else:
            shapes = list(canvas.shapes)

        exclude = self._exclude_labels()
        if exclude:
            shapes = [s for s in shapes if s.label not in exclude]

        if self.only_empty_check.isChecked():
            shapes = [s for s in shapes if not (s.attributes or {}).get("fg")]

        return [s for s in shapes if len(getattr(s, "points", None) or []) >= 2]

    def _label_path_for(self, image_path):
        """按项目的规则从图片路径推出 json 路径（要照顾 output_dir）"""
        parent = self.parent
        output_dir = getattr(parent, "output_dir", None)
        if output_dir:
            base = getattr(parent, "last_open_dir", None)
            if base:
                rel = osp.relpath(image_path, base)
                return osp.join(output_dir, osp.splitext(rel)[0] + ".json")
            return osp.join(
                output_dir, osp.basename(osp.splitext(image_path)[0]) + ".json"
            )
        return osp.splitext(image_path)[0] + ".json"

    def _fixed_color_for_mode(self):
        """描边固定用描边色，字色固定用文字色"""
        return self._outline_color if self._mode() == 1 else self._text_color

    def _parse_params(self):
        # 这几个数只剩老的兜底逻辑在用，界面上已经没有对应控件了，固定成默认值。
        return (
            self._mode(),
            self._fixed_color_for_mode(),
            90,
            12,
            1,
            6,
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run_extract(self):
        if self._scope() == 2:
            self.run_all_pages()
            return

        if getattr(self.parent, "image", None) is None or self.parent.image.isNull():
            self._log(self.tr("没有加载图像"))
            return

        shapes = self._collect_shapes()
        if not shapes:
            self._log(
                self.tr(
                    f"本页没有可处理的框（画布上共 {len(getattr(self.parent.canvas, 'shapes', []) or [])} 个）。"
                    "「只处理本页选中」要先在画布上选中框"
                )
            )
            return

        rgb = qimage_to_rgb(self.parent.image)
        if rgb is None:
            self._log(self.tr("图像读取失败"))
            return

        h, w = rgb.shape[:2]
        mode, fixed_color, maxdist, maxcolors, selsize, finalcolors = (
            self._parse_params()
        )

        self._clear_output()
        self.run_btn.setEnabled(False)
        self._log(
            self.tr(
                f"开始处理 {len(shapes)} 个框　模式 {self._mode_text()}"
            )
        )

        done = 0
        skipped = 0
        failed = 0
        self._batching = True
        for i, shape in enumerate(shapes):
            bbox = shape_bbox(shape, w, h)
            if bbox is None:
                skipped += 1
                continue

            x0, y0, x1, y1 = bbox
            region = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            fg, bg, used_md = pick_color_by_mode(
                region,
                mode,
                fixed_color,
                maxdist_base=maxdist,
                maxcolors=maxcolors,
                selsize=selsize,
                finalcolors=finalcolors,
            )
            if fg is None:
                skipped += 1
                self._log(self.tr(f"  [{i}] 框内只有一种颜色，跳过"))
                continue

            if shape.attributes is None:
                shape.attributes = {}
            shape.attributes["fg"] = [int(c) for c in fg]
            if bg is not None:
                shape.attributes["bg"] = [int(c) for c in bg]
            done += 1

            self._add_color_row(i, shape.label, (x0, y0, x1, y1), fg, bg)
            if len(self._pending_rows) >= 150:
                self._flush_rows()
        self._batching = False
        self._flush_rows()

        if done:
            self.parent.canvas.update()
            self.parent.set_dirty(mark_as_manually_edited=False)

        self.run_btn.setEnabled(True)
        self._log(
            self.tr(
                f"完成：成功 {done} 个，跳过 {skipped} 个，失败 {failed} 个。"
            )
        )

    def _pages(self):
        parent = self.parent
        images = list(getattr(parent, "image_list", None) or [])
        if not images and getattr(parent, "filename", None):
            images = [parent.filename]
        return images

    def refresh_page_range(self):
        """把页码范围的上限对齐到任务里的图片总数"""
        total = len(self._pages())
        if total <= 0:
            for w in (self.start_spin, self.end_spin):
                w.setRange(0, 0)
            return
        for w in (self.start_spin, self.end_spin):
            if w.maximum() != total:
                w.setRange(1, total)
        if self.end_spin.value() in (0, self.end_spin.maximum()):
            self.end_spin.setValue(total)
        if self.start_spin.value() == 0:
            self.start_spin.setValue(1)

    def _open_progress(self, title, total):
        """批量处理才开进度窗口

        样式照抄边距扩缩工具：普通 QProgressDialog，不加自定义样式表，
        标签写「正在处理: N/M」，默认带百分比和取消按钮。
        """
        dlg = QtWidgets.QProgressDialog(
            self.tr("正在处理: 0/{total}").format(total=total),
            self.tr("取消"),
            0,
            max(1, total),
            self,
        )
        dlg.setWindowTitle(title)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        QtWidgets.QApplication.processEvents()
        return dlg

    def _set_progress(self, dlg, done, total, text=None):
        if dlg is None:
            return
        dlg.setMaximum(max(1, total))
        dlg.setValue(done)
        dlg.setLabelText(
            text if text else self.tr("正在处理: {c}/{t}").format(c=done, t=total)
        )
        QtWidgets.QApplication.processEvents()

    def _close_progress(self, dlg):
        if dlg is None:
            return
        if not dlg.wasCanceled():
            dlg.setValue(dlg.maximum())
        dlg.close()

    def run_range(self):
        start = self.start_spin.value() - 1
        end = self.end_spin.value() - 1
        if start < 0 or end < start:
            self._log(self.tr("页码范围无效"))
            return
        self.run_all_pages(start, end)

    def run_all_pages(self, start=None, end=None):
        """批量处理：直接读写每张图对应的 json，不依赖画布。

        不走画布是因为逐张 load_file 太慢，而且会打断当前编辑状态。
        只在原 json 上改 shapes[i]["attributes"]，其余字段原样保留。
        """
        parent = self.parent
        images = self._pages()
        if not images:
            self._log(self.tr("没有可处理的任务图片"))
            return
        if start is not None or end is not None:
            start = 0 if start is None else start
            end = len(images) - 1 if end is None else end
            if start > end or start >= len(images):
                self._log(self.tr("页码范围无效"))
                return
            images = images[start : end + 1]

        exclude = self._exclude_labels()
        only_empty = self.only_empty_check.isChecked()
        mode, fixed_color, maxdist, maxcolors, selsize, finalcolors = (
            self._parse_params()
        )

        self._clear_output()
        self.run_btn.setEnabled(False)
        self.range_btn.setEnabled(False)
        self._log(
            self.tr(
                f"开始处理 {len(images)} 张图　"
                f"模式 {self._mode_text()}"
                + (
                    f"｜固定色 {_hex(fixed_color)}"
                    if self._mode() != 0
                    else ""
                )
            )
        )
        progress = self._open_progress(self.tr("提取颜色"), len(images))

        pages_done = 0
        boxes_done = 0
        pages_skipped = 0
        pages_failed = 0
        current = getattr(parent, "filename", None)

        cancelled = False
        self._batching = True
        for n, image_path in enumerate(images):
            if progress is not None and progress.wasCanceled():
                cancelled = True
                break
            self._set_progress(
                progress,
                n,
                len(images),
                self.tr("正在处理: {c}/{t}  {name}").format(
                    c=n + 1, t=len(images), name=osp.basename(image_path)
                ),
            )
            label_path = self._label_path_for(image_path)
            if not osp.exists(label_path):
                pages_skipped += 1
                continue
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pages_failed += 1
                continue

            rgb = qimage_to_rgb(QtGui.QImage(image_path))
            if rgb is None:
                pages_failed += 1
                continue
            h, w = rgb.shape[:2]

            changed = 0
            for sh in data.get("shapes") or []:
                if exclude and sh.get("label") in exclude:
                    continue
                attrs = sh.get("attributes") or {}
                if only_empty and attrs.get("fg"):
                    continue
                bbox = _points_bbox(sh.get("points") or [], w, h)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                fg, bg, _ = pick_color_by_mode(
                    np.ascontiguousarray(rgb[y0:y1, x0:x1]),
                    mode,
                    fixed_color,
                    maxdist_base=maxdist,
                    maxcolors=maxcolors,
                    selsize=selsize,
                    finalcolors=finalcolors,
                )
                if fg is None:
                    continue
                attrs["fg"] = [int(c) for c in fg]
                if bg is not None:
                    attrs["bg"] = [int(c) for c in bg]
                sh["attributes"] = attrs
                changed += 1
                self._add_color_row(
                    changed,
                    f"{osp.basename(image_path)}  {sh.get('label', '')}",
                    bbox,
                    tuple(int(v) for v in fg),
                    tuple(int(v) for v in bg) if bg is not None else None,
                )
                if len(self._pending_rows) >= 150:
                    self._flush_rows()

            if changed:
                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pages_failed += 1
                    continue
                pages_done += 1
                boxes_done += changed

        self._batching = False
        self._flush_rows()
        self._close_progress(progress)

        # 当前页的结果也写回文件了，重新载入一次让画布显示最新颜色
        if pages_done and current:
            try:
                parent.load_file(current)
            except Exception:
                pass

        self.run_btn.setEnabled(True)
        self.range_btn.setEnabled(True)
        if cancelled:
            self._log(
                self.tr(
                    f"已取消：处理到 {pages_done} 张 / {boxes_done} 个框，"
                    f"后面的没有再执行。"
                )
            )
        else:
            self._log(
                self.tr(
                    f"完成：{pages_done} 张 / {boxes_done} 个框已写入，"
                    f"跳过 {pages_skipped} 张（没有 json），失败 {pages_failed} 张。"
                )
            )

    def clear_selected_colors(self):
        """清除颜色属性。跟着范围按钮走：选中 / 本页 / 全部页面"""
        if self._scope() == 2:
            self.clear_all_pages()
            return

        canvas = getattr(self.parent, "canvas", None)
        if canvas is None:
            return
        exclude = self._exclude_labels()
        if self._scope() == 0:
            shapes = list(getattr(canvas, "selected_shapes", []) or [])
        else:
            shapes = list(canvas.shapes)
        if exclude:
            shapes = [s for s in shapes if s.label not in exclude]
        if not shapes:
            self._log(self.tr("没有可清除的框"))
            return

        self._clear_output()
        n = 0
        for shape in shapes:
            attrs = shape.attributes or {}
            if "fg" in attrs or "bg" in attrs:
                attrs.pop("fg", None)
                attrs.pop("bg", None)
                shape.attributes = attrs
                n += 1
        if n:
            canvas.update()
            self.parent.set_dirty(mark_as_manually_edited=False)
        self._log(self.tr(f"已清除 {n} 个框的颜色属性"))

    def clear_all_pages(self):
        """清除全部页面的颜色属性，直接改各自的 json"""
        images = self._pages()
        if not images:
            self._log(self.tr("没有可处理的任务图片"))
            return
        exclude = self._exclude_labels()
        got = getattr(self.parent, "filename", None)

        self._clear_output()
        pages = 0
        boxes = 0
        progress = self._open_progress(self.tr("提取颜色"), len(images))
        cancelled = False
        for n, image_path in enumerate(images):
            if progress is not None and progress.wasCanceled():
                cancelled = True
                break
            self._set_progress(
                progress,
                n,
                len(images),
                self.tr("正在处理: {c}/{t}").format(c=n + 1, t=len(images)),
            )
            label_path = self._label_path_for(image_path)
            if not osp.exists(label_path):
                continue
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            changed = 0
            for sh in data.get("shapes") or []:
                if exclude and sh.get("label") in exclude:
                    continue
                attrs = sh.get("attributes") or {}
                if "fg" in attrs or "bg" in attrs:
                    attrs.pop("fg", None)
                    attrs.pop("bg", None)
                    sh["attributes"] = attrs
                    changed += 1
            if changed:
                try:
                    with open(label_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    continue
                pages += 1
                boxes += changed

        self._close_progress(progress)
        if pages and got:
            try:
                self.parent.load_file(got)
            except Exception:
                pass
        self._log(
            self.tr(
                ("已取消：" if cancelled else "全部页面清除完成：")
                + f"{pages} 张 / {boxes} 个框已清空颜色"
            )
        )

    def _log(self, msg):
        """状态提示写状态栏，表格只放每个框的取色结果"""
        self.status_label.setText(str(msg))

    def _clear_output(self):
        self._pending_rows = []
        self.table.setRowCount(0)
        self.status_label.setText("")

    def _add_color_row(self, index, label, bbox, fg, outline):
        """每个框占 3 行：标题行 / 文字色 / 描边色

        先攒进队列再一次性写入。逐行 insertRow + scrollToBottom 时
        每一次都要重排整张表，一千个框能拖到十秒以上，全耗在界面上。
        """
        self._pending_rows.append((index, label, bbox, fg, outline))
        if not self._batching:
            self._flush_rows()

    def _flush_rows(self):
        """把攒下的结果一次性写进表格（关掉刷新 + 预先定好行数）"""
        rows = self._pending_rows
        if not rows:
            return
        self._pending_rows = []

        table = self.table
        table.setUpdatesEnabled(False)
        first = table.rowCount()
        table.setRowCount(first + 3 * len(rows))

        for n, (index, label, bbox, fg, outline) in enumerate(rows):
            base = first + 3 * n
            x0, y0, x1, y1 = bbox
            head = QtWidgets.QTableWidgetItem(f"[{index}] {label}")
            head.setToolTip(f"({x0},{y0})-({x1},{y1})　{x1 - x0} × {y1 - y0}")
            table.setItem(base, 0, head)
            table.setSpan(base, 0, 1, 3)

            for offset, name, color in (
                (1, self.tr("文字"), fg),
                (2, self.tr("描边"), outline),
            ):
                table.setItem(
                    base + offset, 0, QtWidgets.QTableWidgetItem(name)
                )
                # 用 QLabel 画色块而不是单元格底色：纯白色块没有边框
                # 时会和白底表格融为一体，看着像空单元格
                chip = QtWidgets.QLabel()
                chip.setStyleSheet(
                    f"background:{_hex(color)}; border:1px solid #999;"
                    if color is not None
                    else "background:transparent; border:1px dashed #bbb;"
                )
                if color is not None:
                    chip.setToolTip(_hex(color))
                wrapper = QtWidgets.QWidget()
                wrap = QtWidgets.QHBoxLayout(wrapper)
                wrap.setContentsMargins(2, 1, 2, 1)
                wrap.addWidget(chip)
                table.setCellWidget(base + offset, 1, wrapper)
                text = (
                    f"{color[0]}, {color[1]}, {color[2]}"
                    if color is not None
                    else "—"
                )
                table.setItem(
                    base + offset, 2, QtWidgets.QTableWidgetItem(text)
                )

        table.setUpdatesEnabled(True)
        table.scrollToBottom()
