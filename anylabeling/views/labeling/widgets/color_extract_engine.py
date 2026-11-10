# -*- coding: utf-8 -*-
"""
漫画文字取色引擎
================

只用一个 numpy，不加载模型、不训练、不依赖任何新库。

输入一块文字框的 RGB 图，输出 (文字色, 描边色)。

道理就三句：

  1. 框里的“纯色”只有几种，其余全是抗锯齿混出来的过渡色。过渡色又薄又碎，
     在颜色密度图里堆不出峰，所以找峰就能把纯色挑出来。到底几种纯色，
     由图自己决定，人不指定。

  2. 给每个像素算“从框外走进来，最少要跨过几道颜色边界”。数值越大，
     说明它被包得越深。这一步区分的是“里外”，不是“面积多少”。

  3. 包得最深的那一片就是文字芯；紧贴它的外一层就是描边。普通没有描边的
     文字，深浅只差一层，自动认出来。

换什么画风都不用调参数，因为上面三步里没有一个是“照着某部漫画量出来的数”。
"""

import numpy as np

# 项目本来就依赖 scipy，用它的连通域（C 实现）比逐像素推快一个数量级。
# 万一哪天环境里没有，下面的纯 numpy 版本照样能跑，只是慢一点。
try:
    from scipy import ndimage as _nd
except Exception:  # pragma: no cover
    _nd = None


# ============================================================
# 只有下面三个数可能用得着改，而且都不是按某部漫画凑出来的：
#   第一个是“多小的色块算噪点”，第二个是“差多少算两种颜色”，
#   第三个是“最多认几种颜色”。正常情况下一个都不用动。
# ============================================================
CHUN_SE_ZUIXIAO = 0.004   # 一种颜色至少占这么大比例才当真（0.4%，再少就是噪点）
CHUN_SE_JULI = 12.0       # 两种颜色在感知空间差这么多才算两种颜色（12≈肉眼明显不同）
CHUN_SE_ZUIDUO = 12       # 最多认这么多种颜色，多出来的当噪声

# 一个框超过这么多像素才缩小处理。纯粹是速度考虑，不影响结果。
# 按像素总数而不是边长判断，是因为细长条形的字幕框（比如 340x60）
# 按边长会被误缩，一缩就把一像素宽的描边整条跳掉了。
ZUI_DA_XIANG_SU = 120000


# ------------------------------------------------------------
# 颜色空间：sRGB → CIELAB
# 用 Lab 是因为它按“人眼觉得差多少”算距离，比直接比 RGB 靠谱得多。
# ------------------------------------------------------------
def _rgb_to_lab(rgb):
    """(H,W,3) 0-255 → (H,W,3) Lab，float32"""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = c.reshape(-1, 3) @ m.T
    xyz = xyz.reshape(rgb.shape) / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _lab_of_one(color):
    """单个颜色 → Lab"""
    a = np.asarray(color, np.float32).reshape(1, 1, 3)
    return _rgb_to_lab(a).reshape(3)


# ------------------------------------------------------------
# 形状工具
# ------------------------------------------------------------
def _peng_zhang(mask, k=1):
    """方形膨胀 k 圈（不改原图）"""
    if k <= 0:
        return mask.copy()
    m = mask
    for _ in range(k):
        p = np.pad(m, 1, constant_values=False)
        m = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return m


def _tian_dong(m):
    """
    把 mask 围出来的洞填满。

    洞 = 没有和框外连通的那部分。一圈描边围住字，这里的洞就是字。
    这是判断「谁在谁里面」最硬的一条证据，跟面积比例没关系。
    """
    if _nd is not None:
        return _nd.binary_fill_holes(m)
    h, w = m.shape
    wai = np.zeros((h, w), bool)
    wai[0, :] = ~m[0, :]
    wai[-1, :] = ~m[-1, :]
    wai[:, 0] = ~m[:, 0]
    wai[:, -1] = ~m[:, -1]
    for _ in range(max(h, w) + 2):
        xin = _peng_zhang(wai) & ~m
        if np.array_equal(xin, wai):
            break
        wai = xin
    return ~wai


def _suo_fang(rgb, zui_da_xiang_su=ZUI_DA_XIANG_SU):
    """像素太多才等比缩小（取整点采样，不做插值以免混色）"""
    h, w = rgb.shape[:2]
    if h * w <= zui_da_xiang_su:
        return rgb
    step = int(np.ceil(np.sqrt(h * w / float(zui_da_xiang_su))))
    step = max(1, step)
    return np.ascontiguousarray(rgb[::step, ::step])


# ------------------------------------------------------------
# 第一步：找纯色（颜色密度图的峰）
# ------------------------------------------------------------
def _zhao_chun_se(lab, rgb):
    """返回 (纯色 RGB 列表, 纯色 Lab 列表, 每种颜色的像素数)"""
    h, w = lab.shape[:2]
    zong = float(h * w)

    # 把 Lab 切成很细的格子，统计每个格子落了多少像素
    li = np.clip((lab[..., 0] / 6.0).astype(np.int32), 0, 16)
    ai = np.clip(((lab[..., 1] + 128.0) / 8.0).astype(np.int32), 0, 31)
    bi = np.clip(((lab[..., 2] + 128.0) / 8.0).astype(np.int32), 0, 31)
    ge_zi = (li * 32 + ai) * 32 + bi

    zong_ge = 17 * 32 * 32
    ji_shu = np.bincount(ge_zi.ravel(), minlength=zong_ge).astype(np.float32)
    ti_ji = ji_shu.reshape(17, 32, 32)

    # 每格加上周围 26 格 = 这个颜色周围一共堆了多少像素。
    # 抗锯齿的过渡色同样是均匀摊开的，加完也堆不高；
    # 真纯色即使被噪声打散到相邻几格，加完还是明显一片。
    p = np.pad(ti_ji, 1, mode="edge")
    ping = np.zeros_like(ti_ji)
    for dl in (-1, 0, 1):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                ping += p[
                    1 + dl : 1 + dl + 17, 1 + da : 1 + da + 32, 1 + db : 1 + db + 32
                ]

    # 找局部最大。注意这里比的是「这一格本身有多少像素」，
    # 不是上面那个加过邻域的值 —— 加过邻域的数会让人眼觉得差很远的两种颜色
    # 互相压过去（比如 白 会被 浅灰底 压没），那就把真正的纯色弄丢了。
    pp = np.pad(ti_ji, 1, mode="edge")
    lin_max = np.zeros_like(ti_ji)
    for dl in (-1, 0, 1):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                lin_max = np.maximum(
                    lin_max,
                    pp[
                        1 + dl : 1 + dl + 17,
                        1 + da : 1 + da + 32,
                        1 + db : 1 + db + 32,
                    ],
                )
    feng = (ti_ji >= lin_max) & (ti_ji > 0)
    if not feng.any():
        return [], [], []

    zui_shao = max(4.0, CHUN_SE_ZUIXIAO * zong)

    hang, lie, ceng = np.nonzero(feng)
    # 排序还是用加过邻域的量：它代表「这一片总共有多大」，
    # 噪声把一种颜色打散成好几格时，靠它挑出主力那一格。
    zhi = ping[hang, lie, ceng]
    shun_xu = np.argsort(-zhi)

    rgb_ping = rgb.reshape(-1, 3)
    li_ping = li.ravel()
    ai_ping = ai.ravel()
    bi_ping = bi.ravel()

    yan_se_rgb = []
    yan_se_lab = []
    shu_liang = []

    for i in shun_xu:
        if float(ping[hang[i], lie[i], ceng[i]]) < zui_shao:
            continue
        gi, gj, gk = int(hang[i]), int(lie[i]), int(ceng[i])

        # 这一格附近大约是什么颜色（只用来判重，够用）
        lm = np.array(
            [gi * 6.0 + 3.0, gj * 8.0 - 125.0, gk * 8.0 - 125.0], np.float32
        )
        chong = False
        for c in yan_se_lab:
            if float(np.linalg.norm(lm - c)) < CHUN_SE_JULI:
                chong = True
                break
        if chong:
            continue

        # 真正取颜色：把这一格和紧邻的 26 格里的像素全拿出来取中位色。
        # 不能只取峰所在那一格 —— 平滑会把峰推到一个像素很少的边缘格上。
        xuan = (
            (np.abs(li_ping - gi) <= 1)
            & (np.abs(ai_ping - gj) <= 1)
            & (np.abs(bi_ping - gk) <= 1)
        )
        n = int(xuan.sum())
        if n < 4:
            continue

        zhong = np.median(rgb_ping[xuan], axis=0)
        zhong = np.clip(zhong, 0, 255).astype(np.float32)
        lab1 = _lab_of_one(zhong).astype(np.float32)

        # 合并过之后可能和已选的颜色重合，再验一次
        chong = False
        for c in yan_se_lab:
            if float(np.linalg.norm(lab1 - c)) < CHUN_SE_JULI:
                chong = True
                break
        if chong:
            continue

        yan_se_rgb.append(zhong)
        yan_se_lab.append(lab1)
        shu_liang.append(n)
        if len(yan_se_rgb) >= CHUN_SE_ZUIDUO:
            break

    if not yan_se_rgb:
        return [], [], []
    return yan_se_rgb, yan_se_lab, shu_liang


def _fen_lei(lab, yan_se_lab):
    """每个像素归到最近的纯色"""
    pal = np.stack(yan_se_lab, axis=0)  # (K,3)
    d = lab[..., None, :] - pal[None, None, :, :]
    d = np.einsum("hwkc,hwkc->hwk", d, d)
    return d.argmin(axis=-1).astype(np.int32)


def _qu_sui_kuai(label, K):
    """
    把零碎的小块并到周围的大块里去。

    漫画的网点纸、扫描噪点会把画面切成成千上万个三像素的小格子，它们会
    让“隔了几层”这个计数一直往上翻，字的层次就被淹没了。字和描边都是一整
    片连着的，所以小块一律当纹理处理。

    只有小块数量很多（说明是网点/噪声）时才动，否则不碰 —— 免得把细笔画抹了。
    """
    if _nd is None:
        return label
    h, w = label.shape
    CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    zong = label.size
    zui_da = max(20, int(0.002 * zong))

    quan = np.zeros((h, w), bool)
    ge_shu = 0
    for k in range(K):
        m = label == k
        if not m.any():
            continue
        lb, n = _nd.label(m, structure=CROSS)
        if n == 0:
            continue
        ji = np.bincount(lb.ravel())
        ji[0] = 0
        xiao = np.flatnonzero(ji <= zui_da)
        if xiao.size == 0:
            continue
        ge_shu += int(xiao.size)
        quan |= np.isin(lb, xiao)

    # 零星几点不算网点，不动
    if ge_shu < 30 or not quan.any() or quan.all():
        return label

    ind = _nd.distance_transform_edt(
        quan, return_distances=False, return_indices=True
    )
    out = label.copy()
    out[quan] = label[tuple(ind[:, quan])]
    return out


def _shi_bai(c, zui_shao=235):
    """这个颜色算不算“白字”"""
    return all(float(v) >= zui_shao for v in c)


def _cai_du(c):
    """彩度：RGB 三分量的最大差，越大越“彩”"""
    return float(max(float(v) for v in c) - min(float(v) for v in c))


def _gui_bai(c, rong_cha=24.0):
    """离纯白很近的颜色，归成纯白

    一像素宽的白描边被 JPEG 咬着，量出来常带一点旁边的杂色
    （255,223,255 这种）。看着是“淡粉边”，其实就是白边。
    真正的淡彩（淡紫、淡粉）离纯白都在 34 以上，不会被误吞。
    """
    lab = _lab_of_one(np.array(c, np.float32))
    bai = _lab_of_one(np.array([255.0, 255.0, 255.0], np.float32))
    if float(np.linalg.norm(lab - bai)) < rong_cha:
        return (255, 255, 255)
    return tuple(int(round(v)) for v in c)


def _zai_xian_duan_shang(c, a, b, rong_cha):
    """c 是不是正好落在 a 到 b 的连线上（也就是 a、b 混出来的过渡色）"""
    ab = np.asarray(b, np.float32) - np.asarray(a, np.float32)
    chang2 = float(np.dot(ab, ab))
    if chang2 < 1e-6:
        return False
    t = float(np.dot(np.asarray(c, np.float32) - np.asarray(a, np.float32), ab)) / chang2
    if t <= 0.05 or t >= 0.95:
        return False
    dian = np.asarray(a, np.float32) + t * ab
    return float(np.linalg.norm(np.asarray(c, np.float32) - dian)) < rong_cha


def _ti_chun(rgb, lab, yan_se, can_kao_lab, bi_li=0.30):
    """
    在一大片同色像素里，挑「离参考色最远」的那一部分再取中位色。

    描边常常只有一两个像素宽，边上全是抗锯齿混出来的过渡色，直接平均会发灰、
    发淡；离底色越远的那一截越接近印刷时真正用的颜色。字芯同理。
    """
    if not yan_se.any():
        return None
    pix = lab[yan_se]
    if pix.shape[0] < 6:
        zhong = np.median(rgb[yan_se], axis=0)
        return np.clip(zhong, 0, 255).astype(np.float32)
    d = pix - np.asarray(can_kao_lab, np.float32)
    d = np.einsum("nc,nc->n", d, d)
    if float(d.max()) < 1e-3:
        # 这一片和参考色根本就是一个颜色，没什么可挑的
        zhong = np.median(rgb[yan_se], axis=0)
        return np.clip(zhong, 0, 255).astype(np.float32)
    nhuo = max(3, int(pix.shape[0] * bi_li))
    shun = np.argsort(-d)[:nhuo]
    zhong = np.median(rgb[yan_se][shun], axis=0)
    return np.clip(zhong, 0, 255).astype(np.float32)


# ------------------------------------------------------------
# 第二步：分层 —— 从框外走进来要跨几道颜色边界
# ------------------------------------------------------------
def _ding_ceng(label, bei_jing, kuang_bian):
    """
    每一块像素离框外“隔了几层”。
    框外面那层是 0，跨过一个颜色就是 1，再往里就是 2。
    数字越大 = 被包得越深 = 越像字的芯。

    同一颜色在里、外各有一块时会被正确分开 —— 这是关键：
    白底、白字、中间隔一圈黑边，里外两块白必须算成不同层。
    所以这里按“连通的块”分层，不是按颜色分层。
    """
    if _nd is not None:
        return _ding_ceng_lian_tong(label, bei_jing, kuang_bian)
    return _ding_ceng_zhu_xiang(label, bei_jing, kuang_bian)


def _ding_ceng_lian_tong(label, bei_jing, kuang_bian):
    """按连通块分层（scipy 版，快）"""
    from collections import deque

    h, w = label.shape
    K = int(label.max()) + 1
    CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

    # 每个颜色的每一块单独编号，合起来得到“块号图”
    kuai = np.zeros((h, w), np.int32)
    pian = 0
    for k in range(K):
        m = label == k
        if not m.any():
            continue
        lab_k, nk = _nd.label(m, structure=CROSS)
        kuai[m] = lab_k[m] + pian
        pian += nk
    if pian == 0:
        return np.full((h, w), 900, np.int16)

    # 相邻的块之间连一条边
    bian = []
    for a, b in ((kuai[:, :-1].ravel(), kuai[:, 1:].ravel()),
                 (kuai[:-1, :].ravel(), kuai[1:, :].ravel())):
        m = a != b
        if not m.any():
            continue
        u = np.minimum(a[m], b[m]).astype(np.int64)
        v = np.maximum(a[m], b[m]).astype(np.int64)
        bian.append(np.unique(u * (pian + 1) + v))
    if bian:
        bian = np.unique(np.concatenate(bian))
        eu = bian // (pian + 1)
        ev = bian % (pian + 1)
    else:
        eu = ev = np.zeros(0, np.int64)

    # 起点：贴着框最外一圈的那些颜色。
    # 不能只认一种背景色 —— 渐变底、带纹理的底会被拆成好几种颜色，
    # 只认一种的话，另一种就会被误当成“第一层”，把字挤掉。
    # 但也不能把什么都算成外面：只在框边占了一点点的（比如框切到了字的边缘），
    # 不算外面。
    qi = np.zeros((h, w), bool)
    qi[:kuang_bian, :] = True
    qi[-kuang_bian:, :] = True
    qi[:, :kuang_bian] = True
    qi[:, -kuang_bian:] = True
    bian_zong = max(1, int(qi.sum()))
    ge_se = np.bincount(label[qi].ravel(), minlength=K).astype(np.float64) / bian_zong
    qi_se = ge_se >= 0.05
    if not qi_se.any():
        qi_se[bei_jing] = True
    qi &= qi_se[label]
    if not qi.any():
        qi = label == bei_jing
    if not qi.any():
        return np.full((h, w), 900, np.int16)

    adj = [[] for _ in range(pian + 1)]
    for u, v in zip(eu.tolist(), ev.tolist()):
        adj[u].append(v)
        adj[v].append(u)

    kuai_cu = np.full(pian + 1, -1, np.int16)
    dq = deque()
    for c in np.unique(kuai[qi]).tolist():
        kuai_cu[c] = 0
        dq.append(c)
    while dq:
        c = dq.popleft()
        d = kuai_cu[c] + 1
        for nb in adj[c]:
            if kuai_cu[nb] < 0:
                kuai_cu[nb] = d
                dq.append(nb)

    cu = np.full((h, w), 900, np.int16)
    you = kuai_cu[kuai] >= 0
    cu[you] = kuai_cu[kuai[you]]
    return cu


def _ding_ceng_zhu_xiang(label, bei_jing, kuang_bian):
    """
    纯 numpy 版：0-1 最短路。同色走一步不花钱，跨色走一步花 1。
    和上面连通块版结果一样，只是逐像素推进，慢一些。
    """
    h, w = label.shape
    INF = 900
    cu = np.full((h, w), INF, np.int16)
    K = int(label.max()) + 1

    qi_dian = np.zeros((h, w), bool)
    qi_dian[:kuang_bian, :] = True
    qi_dian[-kuang_bian:, :] = True
    qi_dian[:, :kuang_bian] = True
    qi_dian[:, -kuang_bian:] = True
    bian_zong = max(1, int(qi_dian.sum()))
    ge_se = np.bincount(label[qi_dian].ravel(), minlength=K).astype(np.float64) / bian_zong
    qi_se = ge_se >= 0.05
    if not qi_se.any():
        qi_se[bei_jing] = True
    qi_dian &= qi_se[label]
    if not qi_dian.any():
        qi_dian = label == bei_jing
    if not qi_dian.any():
        return cu
    cu[qi_dian] = 0

    for _ in range(max(h, w) + 2):
        p = np.pad(cu, 1, constant_values=INF)
        pl = np.pad(label, 1, constant_values=-99)
        xin = cu.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            lin = p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            lin_lab = pl[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            hou_xuan = lin + (lin_lab != label).astype(np.int16)
            np.minimum(xin, hou_xuan, out=xin)
        if np.array_equal(xin, cu):
            break
        cu = xin
    return cu


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def qu_zise(rgb, xiang_xi=False):
    """
    给一块文字框的图，算出 (文字色, 描边色)。

    rgb      : (H,W,3) uint8
    xiang_xi : 为 True 时额外返回过程信息（调试用）

    返回 (文字色, 描边色)，都是 0-255 的整数三元组。
    只有一个颜色、实在分不出字的时候返回 (None, None)。
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return (None, None, None) if xiang_xi else (None, None)
    rgb = rgb[..., :3]
    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        return (None, None, None) if xiang_xi else (None, None)

    # 统一缩到合适大小算，快，而且不影响颜色本身
    xiao = _suo_fang(np.ascontiguousarray(rgb))
    lab = _rgb_to_lab(xiao)

    yan_se_rgb, yan_se_lab, shu_liang = _zhao_chun_se(lab, xiao)
    if len(yan_se_rgb) == 0:
        return (None, None, None) if xiang_xi else (None, None)

    K = len(yan_se_rgb)
    label = _fen_lei(lab, yan_se_lab)
    label = _qu_sui_kuai(label, K)
    mian_ji = np.bincount(label.ravel(), minlength=K)

    # 只有一种颜色：无法区分文字、描边和背景，必须返回失败。
    # 不能把同一个颜色同时伪装成文字色和描边色，否则上层会认为识别成功。
    if K == 1:
        if xiang_xi:
            return None, None, {"zu_shu": 1, "reason": "single_color"}
        return None, None

    # ---- 背景：框最外一圈里占比最大的那种颜色 ----
    kuang_bian = max(1, min(xiao.shape[0], xiao.shape[1]) // 30)
    wai = np.zeros(label.shape, bool)
    wai[:kuang_bian, :] = True
    wai[-kuang_bian:, :] = True
    wai[:, :kuang_bian] = True
    wai[:, -kuang_bian:] = True
    bian_ji = np.bincount(label[wai].ravel(), minlength=K).astype(np.float32)

    # 背景优先由「框边占比」决定，而不是要求它必须占整框 10%。
    # 紧裁剪文字框里，背景可能只占整框很小一部分，但仍然是边缘的主色；
    # 原来的 10% 总面积门槛会把它排除，随后误把文字/描边当成背景。
    edge_total = max(1, int(wai.sum()))
    hou_xuan_bg = bian_ji.astype(np.float32).copy()
    min_edge_pixels = max(3.0, 0.015 * edge_total)
    hou_xuan_bg[hou_xuan_bg < min_edge_pixels] = -1
    # 仅在边框没有足够稳定的颜色时，才退回整框最大簇。
    if hou_xuan_bg.max() < 0:
        hou_xuan_bg = mian_ji.astype(np.float32)
    bei_jing = int(hou_xuan_bg.argmax())

    # ---- 白芯 / 白描边被并进浅色底时，把它拆出来单独算一类 ----
    # 底色压得浅（#D9EDEB 这种），纯白离它太近会被当成同一种颜色：
    # 于是「白字被黑边围住」变成「底色被黑边围住」，字色和描边整个反过来。
    if float(min(yan_se_rgb[bei_jing])) < 225:
        bai = (label == bei_jing) & (xiao.min(axis=2) >= 240)
        n_bai = int(bai.sum())
        if n_bai >= max(8, int(0.02 * label.size)):
            bai_rgb = np.clip(
                np.median(xiao[bai].astype(np.float32), axis=0), 0, 255
            ).astype(np.float32)
            yan_se_rgb.append(bai_rgb)
            yan_se_lab.append(_lab_of_one(bai_rgb).astype(np.float32))
            shu_liang.append(n_bai)
            label[bai] = K
            K += 1
            mian_ji = np.bincount(label.ravel(), minlength=K)
            bian_ji = np.bincount(label[wai].ravel(), minlength=K).astype(np.float32)

    # ---- 找「谁把谁包在里面」 ----
    # 字就是被描边圈起来的那一块。判断办法：把某个颜色的像素全当成墙，
    # 把墙围出来的洞填上，看洞里主要是什么颜色 —— 那就是它包住的东西。
    #
    # 这个判断跟「面积占多少」「在第几层」都没关系，所以：
    #   彩色字 + 细黑轮廓、白字 + 黑边、黑字白底，都是同一套逻辑，
    #   不会出现“轮廓线断了一小段就把字认反”这种毛病。
    zui_shao_dong = max(8, int(0.03 * label.size))

    # 先挑出「另外两种颜色混出来的过渡色」。这种颜色只是抗锯齿的副产品，
    # 不能拿来当描边 —— 不然细描边旁边那一圈灰，总会显得“它把字包住了”。
    guo_du = np.zeros(K, bool)
    for i in range(K):
        for j in range(i + 1, K):
            for k in range(K):
                if k == i or k == j or guo_du[k]:
                    continue
                if mian_ji[k] >= min(int(mian_ji[i]), int(mian_ji[j])):
                    continue
                if _zai_xian_duan_shang(
                    yan_se_lab[k], yan_se_lab[i], yan_se_lab[j], CHUN_SE_JULI * 0.6
                ):
                    guo_du[k] = True

    guan_xi = []
    for c in range(K):
        if guo_du[c]:
            continue
        if mian_ji[c] < max(8, int(0.02 * label.size)):
            continue
        m = label == c
        # 先往外长一圈再填洞：描边线常常只有一像素宽，抗锯齿还会把它咬出小缺口，
        # 直接填洞会漏。长一圈把缺口接上，洞只是小了一点点，不影响判断。
        m_bi = _peng_zhang(m, 1)
        dong = _tian_dong(m_bi) & ~m_bi
        dm = int(dong.sum())
        if dm < zui_shao_dong:
            continue
        c_tong = np.bincount(label[dong].ravel(), minlength=K)
        nei = int(c_tong.argmax())
        if nei == c:
            continue
        # 外层就是底色 —— 这不算「字被描边包住」，只是「底色里有个洞」。
        # 描边色不许等于背景色：一旦拿背景当描边，背景里的木纹、墙、天空
        # 就会原样写进 json（淡紫字配一块木板色）。真正的情况是洞里那圈浅色
        # 才是描边，字在它里面 —— 交给下面的「没有描边」分支按紧贴字的一圈去取。
        if c == bei_jing:
            continue
        # 洞里的东西就是底色本身 —— 这是「一个框套住一片纸」，不是字。
        # 真字的洞永远不是纸色（白芯被并进浅色底的情况，白芯已经单独成类了）。
        if nei == bei_jing:
            continue
        if float(c_tong[nei]) / dm < 0.5:
            continue
        # 洞里装的就是底色 —— 那是字自带的孔（「日」「間」中间那块），不是字。
        # 判据看「洞里这一整片是不是都跟底色一个样」：
        # 纯纸是一片都像；白字芯被压得再淡，也总有一截明显比纸白。
        dong_lab = lab[dong] - yan_se_lab[bei_jing]
        dong_lab = np.sqrt(np.einsum("nc,nc->n", dong_lab, dong_lab))
        if float(np.percentile(dong_lab, 80)) < CHUN_SE_JULI:
            # 洞里跟底色很像。两种可能：
            #   纸从字自带的孔里透进来（「日」「間」）→ 跳过；
            #   白芯被并进浅色底、其实比底色白得多 → 那是字，留下。
            dong_zhong = np.median(xiao[dong].astype(np.float32), axis=0)
            if not (
                float(dong_zhong.min()) >= 235
                and float(min(yan_se_rgb[bei_jing])) < 235
            ):
                continue
        guan_xi.append((dm, c, nei))

    zi_se = None
    miao_bian = None
    zi_dong = None
    if guan_xi:
        # 洞最小的那层最贴身，正好是「字芯 + 包着它的描边」
        guan_xi.sort(key=lambda t: t[0])
        _, miao_bian, zi_se = guan_xi[0]
        # 字芯就用「洞里那一片」本身来取色，不要拿整个颜色分类去取 ——
        # 白字和米白纸常常被归到同一类里，混着取会把字色取成纸色。
        m_bi = _peng_zhang(label == miao_bian, 1)
        zi_dong = _tian_dong(m_bi) & ~m_bi

    if zi_se is None:
        # 没有任何一圈把别的东西圈起来：字就是直接压在底上的。
        # 这时挑「离底色最远」的那一档，而不是面积最大的那一档 ——
        # 白描边、抗锯齿的灰、气泡的阴影都夹在字和底色之间，永远比字芯浅；
        # 面积最大的常常就是这圈浅色，不是字。
        hou = mian_ji.astype(np.float64).copy()
        hou[bei_jing] = -1.0
        if hou.max() <= 0:
            c = tuple(int(round(v)) for v in yan_se_rgb[0])
            return (c, c, {"zu_shu": K}) if xiang_xi else (c, c)
        yuan = np.zeros(K, np.float64)
        for k in range(K):
            if k == bei_jing or mian_ji[k] < max(8, int(0.02 * label.size)):
                yuan[k] = -1.0
                continue
            if float(bian_ji[k]) > 0.2 * float(wai.sum()):
                yuan[k] = -1.0
                continue
            yuan[k] = float(
                np.linalg.norm(yan_se_lab[k] - yan_se_lab[bei_jing])
            )
        if yuan.max() > 0:
            # 有彩色可选就优先彩色：白底黑字、描边的阴影都算“离得远”，
            # 但真正印上去的字一般是有颜色的那一档。
            # 两个门槛都得过：够彩，而且紧贴着的那一圈是浅色（白描边/白对话框）。
            # 照片里的木纹、墙、皮肤颜色够彩，但边上不是浅色，进不来。
            cai = []
            for k in range(K):
                if k == bei_jing:
                    continue
                if _cai_du(yan_se_rgb[k]) < 85:
                    continue
                # 笔画很细的时候，字本身的面积可能只有 1.5%，比 2% 的门槛还小
                if mian_ji[k] < max(8, int(0.01 * label.size)):
                    continue
                if float(bian_ji[k]) > 0.2 * float(wai.sum()):
                    continue
                ring = _peng_zhang(label == k, 3) & ~(label == k)
                if not ring.any():
                    continue
                rpx = xiao[ring]
                if float((rpx.min(axis=1) >= 225).mean()) < 0.55:
                    continue
                cai.append((
                    float(np.linalg.norm(yan_se_lab[k] - yan_se_lab[bei_jing])),
                    k,
                ))
            if cai:
                zi_se = max(cai)[1]
            else:
                zi_se = int(yuan.argmax())
        else:
            zi_se = int(hou.argmax())
        miao_bian = None

        # 白占了大半框、旁边还有一块明显的彩色 —— 那白是对话框或白描边，
        # 彩色才是字。白字压在深色底上时，白只占字的面积，不会过半框。
        if _shi_bai(yan_se_rgb[zi_se]) and mian_ji[zi_se] > 0.5 * label.size:
            cai = [
                (int(mian_ji[k]), k)
                for k in range(K)
                if k != bei_jing
                and k != zi_se
                and mian_ji[k] >= max(8, int(0.02 * label.size))
                and float(bian_ji[k]) <= 0.15 * float(wai.sum())
                and _cai_du(yan_se_rgb[k]) >= 60
            ]
            if cai:
                miao_bian = zi_se
                zi_se = max(cai)[1]

        # 黑字压在白底上：笔画细，抗锯齿那一圈灰反而比黑芯铺得开，
        # 面积最大的成了灰。同一族里明显更黑、面积也够的那档才是字。
        # 只在当前这一档本身就是暗色时才往下找 —— 白字压深底时，
        # 旁边那一圈灰也“更黑”，照这个规则找会把白字换成灰。
        ming = float(np.mean(yan_se_rgb[zi_se]))
        if _cai_du(yan_se_rgb[zi_se]) < 30 and ming <= 200:
            hei = [
                (float(np.mean(yan_se_rgb[k])), k)
                for k in range(K)
                if k != bei_jing
                and k != zi_se
                and mian_ji[k] >= max(8, int(0.3 * float(mian_ji[zi_se])))
                and float(np.mean(yan_se_rgb[k])) <= ming - 25.0
                and _cai_du(yan_se_rgb[k]) < 30
            ]
            if hei:
                zi_se = min(hei)[1]

    # 认出来的“描边”如果正好落在字色和底色的连线上，那它不是描边，
    # 是抗锯齿混出来的那一圈过渡色。丢掉它，按“没有描边”处理。
    if miao_bian is not None and miao_bian != bei_jing:
        if _zai_xian_duan_shang(
            yan_se_lab[miao_bian], yan_se_lab[zi_se], yan_se_lab[bei_jing], CHUN_SE_JULI
        ):
            miao_bian = None

    zi_pian = label == zi_se

    zi_jing = _ti_chun(
        xiao, lab, zi_dong if zi_dong is not None else zi_pian, yan_se_lab[bei_jing]
    )
    if zi_jing is None:
        zi_jing = yan_se_rgb[zi_se]
    zi_rgb = tuple(int(round(v)) for v in zi_jing)

    if miao_bian is None or miao_bian == bei_jing:
        # 没有描边：字是直接压在底上的，那“外面那一圈”就是底色本身
        bian_rgb = tuple(int(round(v)) for v in yan_se_rgb[bei_jing])
        # 白字压在深色底上（半透明黑框、暗背景画），本身没画描边。
        # 这种底色随背后的画面变，每张都不一样，拿去用是脏的；
        # 深底白字要立得住，描边就该是纯黑。
        if _shi_bai(zi_rgb):
            bian_rgb = (0, 0, 0)
        elif float(min(yan_se_rgb[bei_jing])) < 235:
            # 黑字紧挨着一圈白 —— 那是白描边，只是被并进浅色底里了。
            # 看字外面这一圈：过半是近白，就是白边。
            ring = _peng_zhang(label == zi_se, 3) & ~(label == zi_se)
            if ring.any():
                rpx = xiao[ring]
                if float((rpx.min(axis=1) >= 235).mean()) >= 0.5:
                    bian_rgb = (255, 255, 255)
    else:
        bian_jing = _ti_chun(
            xiao, lab, label == miao_bian, yan_se_lab[bei_jing], bi_li=0.25
        )
        if bian_jing is None:
            bian_jing = yan_se_rgb[miao_bian]
        bian_rgb = tuple(int(round(v)) for v in bian_jing)

    # 白色描边量出来常带一点杂色（255,223,255 这种），归成纯白。
    # 字色只归「几乎就是纯白」的那一点 —— 淡青、淡粉的字离纯白也就十几，
    # 一律归白会把「淡彩字 + 黑边」整个吞成「白字 + 黑边」。
    # 两边不能撞成一个色，撞了就把字色还原。
    z0 = zi_rgb
    zi_rgb = _gui_bai(zi_rgb, rong_cha=8.0)
    bian_rgb = _gui_bai(bian_rgb)
    if zi_rgb == bian_rgb and zi_rgb != z0:
        zi_rgb = z0

    # 字色和描边色都是白的 = 这个组合没有信息量（白字压白边，等于没描边）。
    # 真实情况多半是「白字 + 一两像素的深色细边」，细边太窄被并进了别的一档。
    # 那就看紧贴字的那一圈：挑最暗、且和字色有明显反差的那一档当描边。
    if min(zi_rgb) >= 235 and bian_rgb is not None and min(bian_rgb) >= 235:
        ring = _peng_zhang(label == zi_se, 2) & ~(label == zi_se)
        if ring.any():
            an = []
            for k in range(K):
                if k == zi_se:
                    continue
                if not ((label == k) & ring).any():
                    continue
                d = float(np.linalg.norm(yan_se_lab[k] - yan_se_lab[zi_se]))
                if d < 25.0:
                    continue
                an.append((float(np.mean(yan_se_rgb[k])), k))
            if an:
                bian_rgb = _gui_bai(
                    tuple(int(round(v)) for v in yan_se_rgb[min(an)[1]])
                )

    if xiang_xi:
        xin_xi = {
            "zu_shu": K,
            "yan_se": [tuple(int(round(v)) for v in c) for c in yan_se_rgb],
            "shu_liang": list(shu_liang),
            "bei_jing": bei_jing,
            "zi_se": zi_se,
            "miao_bian": miao_bian,
            "bao_wei": list(guan_xi),
        }
        return zi_rgb, bian_rgb, xin_xi
    return zi_rgb, bian_rgb
