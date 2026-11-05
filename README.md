# inference-benchmark

Utilities for preparing operator-level A100 golden input/output data for downstream inference correctness checks. The current implementation downloads and unpacks existing reference artifacts from MinIO; it does not generate golden outputs, execute GPU kernels, or measure inference performance.

## A100 golden IO workflow

1. Select an operator with `--op`; GEMM also requires `--dtype` and `--shape`.
2. Fetch its archive or archive parts from `minio/toolchain/golden_A100_op` using the MinIO client (`mc`).
3. Concatenate split parts when needed, then inspect the first 50 archive entries to infer whether the prefix through `xpu-perf-golden-IO/` should be stripped.
4. Extract into `data/golden_io/` under the repository by default.
5. Delete downloaded archives and parts after successful extraction, unless `--keep-temp` is set.

The extracted data can be used by an external correctness checker. Such a checker is not included here.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tools/golden_io/A100_golden_ops.py` | Download, reconstruct, and extract golden IO archives |
| `data/golden_io/` | Default local output; created at runtime, not included in the repository |
| `.gitignore` | Excludes `data/` from version control |
| `README.md` | Setup, usage, and implementation boundaries |

## Requirements

- Python 3 with f-string support; the script uses only the Python standard library.
- A Unix-like shell with `tar` (including `--strip-components`), `cat`, and `head`.
- MinIO Client `mc` on `PATH`, an alias named `minio`, and authorized access to the source bucket.
- Local disk space for both downloaded archives/parts and extracted data.

No GPU, CUDA, or PyTorch installation is required for this download utility. The remote dataset and credentials are not distributed with the repository.

Configure the alias using credentials supplied by your storage administrator; replace the placeholders:

```bash
mc alias set minio <MINIO_ENDPOINT> <ACCESS_KEY> <SECRET_KEY>
```

## Run

From the repository root:

```bash
# Inspect the CLI without storage access or mc.
python3 tools/golden_io/A100_golden_ops.py --help
python3 tools/golden_io/A100_golden_ops.py --list

# Download and extract a direct archive.
python3 tools/golden_io/A100_golden_ops.py --op add

# Download, concatenate, and extract a split archive.
python3 tools/golden_io/A100_golden_ops.py --op index_add

# Select a GEMM dataset using its remote directory names.
python3 tools/golden_io/A100_golden_ops.py --op gemm --dtype fp16 --shape kn_1024_8192

# Override the destination and retain intermediate files.
python3 tools/golden_io/A100_golden_ops.py --op topk --output-root ./data/golden_io_debug --keep-temp
```

The GEMM example matches the naming convention in the source; remote availability is not verified. `--dtype` and `--shape` are used for GEMM only, and do not filter other operators.

### Supported archive layouts

All remote paths below are relative to `minio/toolchain/golden_A100_op/`.

| Layout | Operators | Remote artifact pattern |
| --- | --- | --- |
| Direct archive | add, cast, embedding, gather, gelu, layer_norm, mul, reduce_max, reduce_min, reduce_sum, rms_norm, silu, softmax, sub | `<op>.tar.gz` |
| Split archive | index_add, index_select, scatter, topk | `<op>/<op>.part_*` |
| GEMM split archive | gemm | `gemm/<dtype>/<shape>/<shape>_<dtype>.part_*` |

## Inputs and outputs

**Inputs:** operator selection, optional GEMM dtype/shape, output path, and existing remote archives. `fp16`, `fp32`, and `bf16` are CLI examples, not a verified coverage matrix.

**Outputs:** extracted archive contents beneath the output root, plus command/progress logs. The source refers to `.bin` tensor files and `.json` files, but does not define or validate their schema, tensor shapes, dtypes, or expected output names. The precise directory structure depends on the archive.

With `--keep-temp`, direct downloads retain `<op>.tar.gz`; split downloads retain `<op>_parts/`, or `<shape>_<dtype>_parts/` for GEMM, including the reconstructed archive. Custom output locations outside `data/` are not covered by the repository's data ignore rule.

A `[DONE]` message means the download/extraction path completed, not that numerical correctness passed.

## Current limitations and validation boundary

- This repository contains a data preparation utility, not a complete benchmark harness. It includes no latency, throughput, accuracy, or speedup results, and no numerical comparison tolerances or pass/fail report.
- Remote downloads require access to the configured MinIO service. No public sample dataset, pinned dataset version, or checksum manifest is included.
- Archive prefix detection is heuristic. Part ordering relies on shell filename expansion; part completeness and artifact integrity are not explicitly checked.
- Downloads/extraction are not transactional: reruns can overwrite files, and failures may leave partial data or temporary files.
- The script executes shell commands and extracts archives directly. Use trusted arguments and archives.
- Dependency versions and supported platform combinations are not pinned. The repository includes no automated test suite or CI configuration.
- The README examples describe the checked-in implementation. CLI help/list can be checked offline; successful remote retrieval and A100 numerical correctness require separate validation.
