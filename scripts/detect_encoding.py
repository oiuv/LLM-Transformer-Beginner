"""
扫描 data/ 下所有 txt 文件,识别编码并输出报告。
- 并行版: 用 ProcessPoolExecutor 多进程
- 减小样本: 严格解码用 256KB,容错比对只用 64KB(够识别)
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 相对路径:相对于项目根目录
# 脚本位于 scripts/,项目根 = scripts/ 的上一级
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "03-practice" / "chinese-gpt" / "data"
SAMPLE_STRICT = 256 * 1024
SAMPLE_LENIENT = 64 * 1024


def detect_with_bom(raw: bytes):
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "BOM"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le", "BOM"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be", "BOM"
    return None


def detect(path_str: str):
    fp = Path(path_str)
    try:
        raw = fp.read_bytes()
    except Exception as e:
        return (fp, "ERROR", "", str(e), 0)

    size = len(raw)
    sample_full = raw[:SAMPLE_STRICT]
    sample_small = raw[:SAMPLE_LENIENT]

    bom = detect_with_bom(sample_full)
    if bom:
        enc, method = bom
    else:
        try:
            sample_full.decode("utf-8")
            enc, method = "utf-8", "strict-utf8"
        except UnicodeDecodeError:
            # 容错比对
            su_text = sample_small.decode("utf-8", errors="replace")
            sg_text = sample_small.decode("gb18030", errors="replace")
            ru = su_text.count("\ufffd")
            rg = sg_text.count("\ufffd")
            n = len(su_text)
            pu = ru / max(n, 1)
            pg = rg / max(n, 1)
            TH = 0.005
            if pu < TH and pu <= pg:
                enc, method = "utf-8", f"lenient-utf8({pu:.2%})"
            elif pg < TH and pg < pu:
                enc, method = "gbk", f"gb18030({pg:.2%})"
            elif pu < TH and pg < TH:
                enc, method = "utf-8", f"ambiguous-utf8(u={pu:.2%}/g={pg:.2%})"
            else:
                enc, method = "unknown", f"u={pu:.2%}/g={pg:.2%}"

    return (fp, enc, method, "", size)


def classify(enc: str) -> str:
    if enc == "gbk":
        return "GBK (需转换)"
    if enc in ("utf-8", "utf-8-sig"):
        return "UTF-8 (已就绪)"
    if enc.startswith("utf-16"):
        return f"{enc.upper()} (可选)"
    if enc == "unknown":
        return "未知 (需检查)"
    if enc == "ERROR":
        return "ERROR"
    return f"{enc} (其他)"


def main():
    files = sorted(DATA_DIR.rglob("*.txt"))
    print(f"扫描目录: {DATA_DIR}")
    print(f"找到 {len(files)} 个 txt 文件,并行处理中...\n", flush=True)

    stats = {}
    rows = []
    cpu = max(2, min(8, os.cpu_count() or 4))
    print(f"使用 {cpu} 个进程", flush=True)

    with ProcessPoolExecutor(max_workers=cpu) as ex:
        futures = {ex.submit(detect, str(p)): p for p in files}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            fp, enc, method, err, size = r
            label = classify(enc)
            stats[label] = stats.get(label, 0) + 1
            rel = fp.relative_to(DATA_DIR)
            rows.append((rel, label, enc, method, size))
            done += 1
            if done % 500 == 0:
                print(f"  已处理 {done}/{len(files)}", flush=True)

    print("\n" + "=" * 60)
    print("编码分布统计:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v} 个")
    print("=" * 60)

    # 详细列表
    print("\n详细列表(按编码类别分组):")
    print(f"{'文件':<58} {'编码':<22} {'识别方式':<24} {'大小'}")
    print("-" * 118)

    gbk_rows = [r for r in rows if r[1] == "GBK (需转换)"]
    other_rows = [r for r in rows if r[1] != "GBK (需转换)"]

    def print_row(rel, label, enc, method, size):
        rel_s = str(rel)
        if len(rel_s) > 56:
            rel_s = "…" + rel_s[-55:]
        size_s = f"{size/1024/1024:.2f} MB" if size > 1024 * 1024 else f"{size/1024:.1f} KB"
        print(f"{rel_s:<58} {label:<22} {method:<24} {size_s}", flush=True)

    if gbk_rows:
        print(f"\n--- GBK 文件 (共 {len(gbk_rows)} 个,显示前 30) ---")
        for r in gbk_rows[:30]:
            print_row(*r)
        if len(gbk_rows) > 30:
            print(f"... 省略其余 {len(gbk_rows)-30} 个 GBK 文件")

    if other_rows:
        print(f"\n--- 非 GBK 文件 (共 {len(other_rows)} 个,显示前 30) ---")
        for r in other_rows[:30]:
            print_row(*r)
        if len(other_rows) > 30:
            print(f"... 省略其余 {len(other_rows)-30} 个")

    # 保存完整报告
    out = DATA_DIR.parent / "encoding_report.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"扫描目录: {DATA_DIR}\n文件总数: {len(files)}\n\n")
        f.write("编码分布:\n")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            f.write(f"  {k}: {v}\n")
        f.write("\n--- GBK 文件 ---\n")
        for rel, label, enc, method, size in gbk_rows:
            f.write(f"{rel}\t{method}\n")
        f.write("\n--- 非 GBK 文件 ---\n")
        for rel, label, enc, method, size in other_rows:
            f.write(f"{rel}\t{label}\t{method}\n")
    print(f"\n完整报告已保存到: {out}")


if __name__ == "__main__":
    main()