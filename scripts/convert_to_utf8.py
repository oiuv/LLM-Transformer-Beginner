"""
把 data/ 下所有 GBK 编码的 txt 原地转换为 UTF-8(无 BOM)。
- 跳过 UTF-8 / UTF-16 文件(已就绪)
- 解码失败的文件记录到 encoding_report.txt 末尾,不动它
- 并行执行
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import time

# 相对路径:相对于项目根目录
# 脚本位于 scripts/,项目根 = scripts/ 的上一级
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "03-practice" / "chinese-gpt" / "data"
REPORT_FILE = DATA_DIR.parent / "encoding_report.txt"


def convert_one(path_str: str):
    """单文件转换任务,返回 (path, status, message)"""
    fp = Path(path_str)
    try:
        raw = fp.read_bytes()
    except Exception as e:
        return (fp, "READ_FAIL", str(e))

    # BOM 头判断
    if raw.startswith(b"\xef\xbb\xbf") or raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return (fp, "SKIP_BOM", "已有 BOM")

    # 严格 UTF-8:跳过
    try:
        raw.decode("utf-8")
        return (fp, "SKIP_UTF8", "已是 UTF-8")
    except UnicodeDecodeError:
        pass

    # GB18030 解码(Gbk / GB2312 / GBK 都覆盖)
    try:
        text = raw.decode("gb18030")
    except UnicodeDecodeError as e:
        return (fp, "DECODE_FAIL", f"GB18030 失败: {e}")

    # 写回为 UTF-8(无 BOM)
    try:
        fp.write_bytes(text.encode("utf-8"))
    except Exception as e:
        return (fp, "WRITE_FAIL", str(e))

    return (fp, "OK", "")


def main():
    # 只对 GBK 文件转(从报告里读)
    if not REPORT_FILE.exists():
        print(f"找不到报告: {REPORT_FILE},请先运行 detect_encoding.py")
        sys.exit(1)

    gbk_files = []
    with REPORT_FILE.open("r", encoding="utf-8") as f:
        in_gbk_section = False
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("--- GBK"):
                in_gbk_section = True
                continue
            if line.startswith("--- 非 GBK"):
                in_gbk_section = False
                continue
            if in_gbk_section and "\t" in line:
                rel = line.split("\t")[0]
                if rel:
                    gbk_files.append(DATA_DIR / rel)

    print(f"待转换 GBK 文件数: {len(gbk_files)}", flush=True)

    cpu = max(2, min(8, os.cpu_count() or 4))
    print(f"使用 {cpu} 个进程", flush=True)

    stats = {"OK": 0, "SKIP_UTF8": 0, "SKIP_BOM": 0, "READ_FAIL": 0,
             "DECODE_FAIL": 0, "WRITE_FAIL": 0}
    failed = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=cpu) as ex:
        futures = {ex.submit(convert_one, str(p)): p for p in gbk_files}
        done = 0
        for fut in as_completed(futures):
            fp, status, msg = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if status not in ("OK", "SKIP_UTF8", "SKIP_BOM"):
                failed.append((fp.relative_to(DATA_DIR), status, msg))
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(gbk_files) - done) / rate if rate > 0 else 0
                print(f"  已处理 {done}/{len(gbk_files)}  速度 {rate:.1f} 文件/秒  剩余 ~{eta:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.1f}s")
    print("\n结果统计:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k:<14} {v}")

    if failed:
        print(f"\n失败文件 {len(failed)} 个,记录到 encoding_report.txt 末尾:")
        with REPORT_FILE.open("a", encoding="utf-8") as f:
            f.write("\n\n--- 转换失败列表 ---\n")
            for rel, status, msg in failed:
                f.write(f"{rel}\t{status}\t{msg}\n")
                print(f"  {rel}  [{status}] {msg}")
    else:
        print("\n✅ 全部转换成功,无失败")


if __name__ == "__main__":
    main()