def align_to_vits_symbols(phones: list) -> list:
    """
    将 pyopenjtalk / espnet 提取的原始音素和韵律符号数组，
    清洗并映射为完全兼容 symbols2.py 词表的格式。
    
    参数:
        phones (list): 原始音素数组，例如 ['^', 'k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']
        
    返回:
        list: 对齐后的音素数组，例如 ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']
    """
    
    # 1. 标点符号映射表 (提取自原项目 post_replace_ph)
    # 作用：将全角标点或非常规标点，降维到 symbols2.py 中仅有的 ["!", "?", "…", ",", "."]
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
    }

    # 2. 冗余符号黑名单
    # 作用：剔除 espnet 输出但 symbols2.py 未定义的边界符号
    # ^: 句首符号 | $: 句末符号 | #: 重音短语边界
    # 注：espnet 输出的 pau 已经被转换成了 "_"，对应 symbols2 的 pad = "_"，故保留。
    ignored_symbols = {"^", "$", "#"}

    aligned_phones = []

    for ph in phones:
        # 第一步：标点强转映射
        # 如果 ph 在字典里，替换为映射值；否则保持原样
        mapped_ph = rep_map.get(ph, ph)

        # 第二步：剔除词表外符号
        if mapped_ph in ignored_symbols:
            continue
            
        aligned_phones.append(mapped_ph)

    return aligned_phones

# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟 espnet 处理 "こんにちは。" 后的输出
    raw_output = ['^', 'k', 'o', '#', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '。', '$']
    
    final_output = align_to_vits_symbols(raw_output)
    
    print(f"原始输出: {raw_output}")
    print(f"对齐输出: {final_output}")
    # 预期输出: ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '.']