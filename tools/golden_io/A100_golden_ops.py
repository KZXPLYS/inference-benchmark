#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
import subprocess
import sys


MINIO_ROOT = "minio/toolchain/golden_A100_op"
DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/A100-Golden-Ops")

# 直接 tar.gz 的普通算子
DIRECT_TAR_OPS = {
    "add",
    "cast",
    "embedding",
    "gather",
    "gelu",
    "layer_norm",
    "mul",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "rms_norm",
    "silu",
    "softmax",
    "sub",
}

# 单层分片算子
PART_OPS = {
    "index_add",
    "index_select",
    "scatter",
    "topk",
}


def run_cmd(cmd, cwd=None, capture_output=False):
    print(f"[CMD] {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )
    if result.returncode != 0:
        if capture_output:
            raise RuntimeError(
                f"Command failed: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout if capture_output else None


def check_mc():
    mc_path = shutil.which("mc")
    if mc_path is None:
        raise RuntimeError(
            "mc not found in PATH.\n"
            "请先配置MinIO客户端:\n\n"
            "  mkdir -p ~/.local/bin\n"
            "  wget https://dl.minio.org.cn/client/mc/release/linux-amd64/mc -O ~/.local/bin/mc\n"
            "  chmod +x ~/.local/bin/mc\n"
            "  export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "  mc alias set minio http://minio-api.evas.ai:9000 cicd abc@1234\n"
        )
    print(f"[INFO] Found mc: {mc_path}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_remove_file(path):
    if os.path.isfile(path):
        print(f"[CLEAN] Remove file: {path}")
        os.remove(path)


def safe_remove_dir(path):
    if os.path.isdir(path):
        print(f"[CLEAN] Remove dir : {path}")
        shutil.rmtree(path)


def detect_strip_components(tar_path):
    """
    自动探测 tar 包需要 strip 几层。
    逻辑：
    - 读取 tar 顶部若干文件路径
    - 找到第一个包含 .bin / .json 的成员
    - 如果路径像 home/fiery.fang/xpu-perf-golden-IO/cast/...，则 strip 到真正算子目录前
    """
    output = run_cmd(f'tar -tzf "{tar_path}" | head -50', capture_output=True)
    lines = [x.strip() for x in output.splitlines() if x.strip()]

    if not lines:
        print("[WARN] Empty tar listing, fallback strip_components=0")
        return 0

    for line in lines:
        parts = [p for p in line.split("/") if p]
        if not parts:
            continue

        if any(
            token.endswith(".bin") or token.endswith(".json")
            for token in parts
        ):
            # 例如:
            # home/fiery.fang/xpu-perf-golden-IO/cast/1x1024_fp32/input1.bin
            # 希望最终保留 cast/...
            if "xpu-perf-golden-IO" in parts:
                idx = parts.index("xpu-perf-golden-IO")
                # strip 到 xpu-perf-golden-IO 前，再额外去掉它本身
                strip_n = idx + 1
                print(f"[INFO] Auto detected strip-components={strip_n} from {line}")
                return strip_n

            # 如果是类似 cast/1x1024_fp32/input1.bin
            # 一般不需要 strip
            print(f"[INFO] Auto detected strip-components=0 from {line}")
            return 0

    # 如果没找到 bin/json，再看第一条
    first = lines[0]
    parts = [p for p in first.split("/") if p]
    if "xpu-perf-golden-IO" in parts:
        idx = parts.index("xpu-perf-golden-IO")
        strip_n = idx + 1
        print(f"[INFO] Auto detected strip-components={strip_n} from first entry")
        return strip_n

    print("[INFO] Auto detected strip-components=0")
    return 0


def untar_file_auto(tar_path, output_dir):
    ensure_dir(output_dir)
    strip_components = detect_strip_components(tar_path)
    if strip_components > 0:
        run_cmd(
            f'tar -xzvf "{tar_path}" --strip-components={strip_components} -C "{output_dir}"'
        )
    else:
        run_cmd(f'tar -xzvf "{tar_path}" -C "{output_dir}"')


def fetch_direct_tar(op, output_root, cleanup):
    """
    普通算子：
    minio/toolchain/golden_A100_op/add.tar.gz
    """
    ensure_dir(output_root)

    remote_tar = f"{MINIO_ROOT}/{op}.tar.gz"
    local_tar = os.path.join(output_root, f"{op}.tar.gz")
    extract_dir = output_root

    print(f"[INFO] Fetching direct tar op: {op}")
    run_cmd(f'mc cp "{remote_tar}" "{local_tar}"')

    print(f"[INFO] Extracting to: {extract_dir}")
    untar_file_auto(local_tar, extract_dir)

    if cleanup:
        safe_remove_file(local_tar)

    print(f"[DONE] op={op}")
    print(f"[DIR ] extracted: {extract_dir}")


def fetch_part_op(op, output_root, cleanup):
    """
    单层分片算子：
    minio/toolchain/golden_A100_op/index_add/index_add.part_000 ...
    """
    ensure_dir(output_root)

    remote_dir = f"{MINIO_ROOT}/{op}"
    local_parts_dir = os.path.join(output_root, f"{op}_parts")
    local_tar = os.path.join(local_parts_dir, f"{op}.tar.gz")
    extract_dir = output_root

    ensure_dir(local_parts_dir)

    print(f"[INFO] Fetching part op: {op}")
    run_cmd(f'mc cp --recursive "{remote_dir}/" "{local_parts_dir}/"')

    print(f"[INFO] Reconstructing tar.gz: {local_tar}")
    run_cmd(f'cat "{local_parts_dir}/{op}.part_"* > "{local_tar}"')

    print(f"[INFO] Extracting to: {extract_dir}")
    untar_file_auto(local_tar, extract_dir)

    if cleanup:
        safe_remove_file(local_tar)
        safe_remove_dir(local_parts_dir)

    print(f"[DONE] op={op}")
    print(f"[DIR ] extracted: {extract_dir}")


def fetch_gemm(dtype, shape, output_root, cleanup):
    """
    GEMM:
    minio/toolchain/golden_A100_op/gemm/fp16/kn_1024_8192/kn_1024_8192_fp16.part_000
    """
    if not dtype:
        raise ValueError("gemm requires --dtype")
    if not shape:
        raise ValueError("gemm requires --shape")

    ensure_dir(output_root)

    remote_dir = f"{MINIO_ROOT}/gemm/{dtype}/{shape}"
    local_parts_dir = os.path.join(output_root, f"{shape}_{dtype}_parts")
    local_tar_name = f"{shape}_{dtype}.tar.gz"
    local_tar = os.path.join(local_parts_dir, local_tar_name)
    extract_dir = output_root

    ensure_dir(local_parts_dir)

    print(f"[INFO] Fetching GEMM: dtype={dtype}, shape={shape}")
    run_cmd(f'mc cp --recursive "{remote_dir}/" "{local_parts_dir}/"')

    prefix = f"{shape}_{dtype}.part_"
    matched_parts = sorted(glob.glob(os.path.join(local_parts_dir, f"{prefix}*")))
    if not matched_parts:
        raise FileNotFoundError(
            f"No part files found under {local_parts_dir} with prefix {prefix}"
        )

    print(f"[INFO] Found {len(matched_parts)} part files")
    print(f"[INFO] Reconstructing tar.gz: {local_tar}")
    run_cmd(f'cat "{local_parts_dir}/{prefix}"* > "{local_tar}"')

    print(f"[INFO] Extracting to: {extract_dir}")
    untar_file_auto(local_tar, extract_dir)

    if cleanup:
        safe_remove_file(local_tar)
        safe_remove_dir(local_parts_dir)

    print(f"[DONE] op=gemm dtype={dtype} shape={shape}")
    print(f"[DIR ] extracted: {extract_dir}")


def list_supported_ops():
    print("Supported direct tar ops:")
    for op in sorted(DIRECT_TAR_OPS):
        print(f"  - {op}")
    print("\nSupported part ops:")
    for op in sorted(PART_OPS):
        print(f"  - {op}")
    print("\nSupported special op:")
    print("  - gemm (requires --dtype and --shape)")


def main():
    parser = argparse.ArgumentParser(
        description="One-click fetch/unpack xpu-perf A100 Golden IO from MinIO"
    )
    parser.add_argument(
        "--op",
        help="Operator name, e.g. add / index_add / gemm",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Required for gemm, e.g. fp16 / fp32 / bf16",
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="Required for gemm, e.g. kn_1024_8192",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Local output root, default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported ops and exit",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate tar.gz / parts directory",
    )

    args = parser.parse_args()

    if args.list:
        list_supported_ops()
        return

    if not args.op:
        parser.error("--op is required unless --list is used")

    cleanup = not args.keep_temp

    check_mc()
    ensure_dir(args.output_root)

    op = args.op.strip()

    try:
        if op in DIRECT_TAR_OPS:
            fetch_direct_tar(op, args.output_root, cleanup)
        elif op in PART_OPS:
            fetch_part_op(op, args.output_root, cleanup)
        elif op == "gemm":
            fetch_gemm(args.dtype, args.shape, args.output_root, cleanup)
        else:
            raise ValueError(
                f"Unsupported op: {op}\n"
                f"Use --list to see supported ops."
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()