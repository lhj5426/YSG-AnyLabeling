# -*- coding: utf-8 -*-
"""样式库（照 AEG 的 ass_style_storage.cpp 抄）

一个库 = 一个 .sty 文件，一行一条 ASS 样式原文（Style: 名字,Arial,48,...），
跟 gongzuotai.ini 摆在同一个文件夹里。库是全局的：打开哪份字幕都能用，
也能把库里的样式搬进字幕、把字幕里的样式存回库。

格式跟 AEG 一样：UTF-8（带 BOM），一行一条，认不出来的行直接跳过；
没有这个库文件就当空库（不报错）。一个库都没有时自动建一个默认库，
里面只放一条 Default（照 AEG 的 DialogStyleManager::LoadCatalog）。
"""

import os
import os.path as osp

MOREN_KU = "Default"            # 默认库的名字
KU_HOU_ZHUI = ".sty"            # 库文件的后缀

# 默认库里那一条 Default（跟编辑器里「新建样式」摆的是同一套：Arial / 48 / 白字）
YANGSHI_MOREN_HANG = (
    "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
)


def ku_wenjian_jia():
    """样式库放在哪个文件夹：跟 gongzuotai.ini 一个地方（软件根目录）

    本文件在 <软件根>/anylabeling/views/labeling/widgets/ 下面，往上走五层
    就是软件根目录 —— 跟 video_infer_panel.buju_peizhi_lu() 算的是同一个地方。
    """
    return osp.dirname(
        osp.dirname(
            osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        )
    )


def ku_ming_gan_jing(ming):
    """库名洗一洗：文件名里不能用的字符换成下划线（照 AEG 的 OnCatalogNew）"""
    ming = str(ming or "").strip()
    for ch in '\\/:*?"<>|':
        ming = ming.replace(ch, "_")
    return ming.strip()


def ku_lu(ming):
    """一个库对应的文件路径"""
    return osp.join(ku_wenjian_jia(), ku_ming_gan_jing(ming) + KU_HOU_ZHUI)


def ku_zai(ming):
    """这个库现在存不存在"""
    ming = ku_ming_gan_jing(ming)
    return bool(ming) and osp.isfile(ku_lu(ming))


def lie_ku():
    """现在有哪些库（默认库排最前，其余按名字排）"""
    ming = []
    try:
        for x in os.listdir(ku_wenjian_jia()):
            if x.lower().endswith(KU_HOU_ZHUI):
                ming.append(osp.splitext(x)[0])
    except OSError:
        return []
    qi = [x for x in ming if x.lower() == MOREN_KU.lower()]
    qi += sorted(x for x in ming if x.lower() != MOREN_KU.lower())
    return qi


def du_ku(ming):
    """读一个库：返回里面的样式原文行（[str, ...]）

    库文件不在 / 读不动就当空库（照 AEG 的 AssStyleStorage::Load）。
    """
    hang = []
    try:
        with open(ku_lu(ming), "r", encoding="utf-8-sig") as wj:
            for xian in wj:
                x = xian.strip()
                if x and x.lower().startswith("style:"):
                    hang.append(x)
    except OSError:
        return []
    return hang


def xie_ku(ming, hang_men):
    """写一个库：一行一条（照 AEG 的 AssStyleStorage::Save，带 BOM）"""
    ming = ku_ming_gan_jing(ming)
    if not ming:
        return False
    try:
        with open(ku_lu(ming), "w", encoding="utf-8-sig", newline="\n") as wj:
            for x in hang_men or []:
                x = str(x).strip()
                if x:
                    wj.write(x + "\n")
    except OSError:
        return False
    return True


def xin_ku(ming):
    """新建一个空库；名字洗没了 / 已经有同名的，返回空串"""
    ming = ku_ming_gan_jing(ming)
    if not ming or ku_zai(ming):
        return ""
    if not xie_ku(ming, []):
        return ""
    return ming


def shan_ku(ming):
    """删掉一个库（连带里面的样式）"""
    try:
        os.remove(ku_lu(ming))
    except OSError:
        return False
    return True


def que_bao_moren_ku():
    """一个库都没有时，建一个默认库、里面只放一条 Default（照 AEG）"""
    if lie_ku():
        return
    xie_ku(MOREN_KU, [YANGSHI_MOREN_HANG])
