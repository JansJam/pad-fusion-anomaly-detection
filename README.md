# Screen presentation attack detection

A PatchCore-inspired, one-class PAD prototype combining whole-image context with native-resolution texture features. It reads a folder of face images and writes one continuous **attack score** per image. Higher scores indicate greater deviation from bona fide reference features. Scores are **not probabilities** and may be negative or greater than one.

The submission includes the frozen model weights, reference banks, score-normalisation parameters and decision threshold. Inference requires no training data, labels, Google Drive access or model downloads.

## Quick start: Docker CPU inference

Run the commands from the repository root, alongside `Dockerfile`. Docker must be installed and running. Internet access is required during the build; inference runs offline. No GPU is required.

Build the image:

```sh
docker build -t pad-fusion:1 .
```

Create an `images` directory and place your input images inside it. Input images and generated outputs are not included in the submission. Subfolders are supported: for example, `images/Bonafide/00962.png` and `images/Screens/00962.png` remain separate inputs. Folder names do not affect predictions.

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force images, outputs
docker run --rm --network none --mount "type=bind,source=$($PWD.Path)\images,target=/input,readonly" --mount "type=bind,source=$($PWD.Path)\outputs,target=/output" pad-fusion:1 --input /input --output /output/scores.csv --recursive
```

**Linux/macOS shell**

```bash
mkdir -p images outputs
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD/images,target=/input,readonly" \
  --mount "type=bind,source=$PWD/outputs,target=/output" \
  pad-fusion:1 --input /input --output /output/scores.csv --recursive
```

Results are written to `outputs/scores.csv`. To use other directories, replace the host paths in the two `source=` arguments. The input directory must exist and the output directory must be writable. Add `--overwrite` to replace an existing CSV.

**Input requirement: both image dimensions must be at least 384 pixels after EXIF orientation correction.** Smaller images produce explicit error rows; they are not automatically enlarged.

## Input and command-line options

Supported extensions: JPG/JPEG, PNG, BMP, TIF/TIFF and WebP. Only single-frame images are supported. 

```sh
docker run --rm --network none pad-fusion:1 --help
```

| Option | Purpose |
|---|---|
| `--input` | Required input directory |
| `--output` | Required output CSV path |
| `--recursive` | Include images in subdirectories |
| `--overwrite` | Allow replacement of an existing output CSV |
| `--threads` | CPU thread count; default `4` |
| `--device` | `cpu` (default), `cuda` or `auto`; the supplied Docker image supports CPU only |
| `--model` | Checkpoint path; defaults to the bundled `artifacts/fusion_model.pt` |

## CSV output

The principal evaluation columns are `image_id` and `attack_score`. Additional columns expose the fixed decision rule and branch diagnostics. Rows are sorted by relative image path.

| Column | Meaning |
|---|---|
| `image_id` | Image path relative to the input directory, preserving subfolders |
| `attack_score` | Final score: `max(z_whole, z_native)` |
| `predicted_attack` | `1` if `attack_score > threshold`; otherwise `0` |
| `threshold` | Frozen fusion threshold, approximately `1.678816` |
| `score_whole`, `score_native` | Raw anomaly scores from the whole-image and native-crop branches |
| `z_whole`, `z_native` | Each raw score minus its calibration median, divided by its calibration IQR |
| `dominant_branch` | Branch supplying the maximum normalised score, or `tie`; not a causal explanation |
| `width`, `height` | Pixel dimensions after EXIF orientation correction, before resizing/cropping |
| `status`, `error` | `ok` with an empty error on success; `error` with an explanation on failure |

Unreadable, multi-frame and undersized images receive blank score/prediction fields and an error status. A failed input is **not a bona fide prediction**. Unsupported extensions are skipped. An empty input folder is an error.

Exit codes: `0` when all processed images succeed, `2` when one or more per-image errors occur (the CSV is still written), and `1` for setup/global failures. The final CSV is written atomically after processing finishes.

## Method

1. Correct EXIF orientation and convert to RGB. Alpha channels are discarded.
2. **Whole-image branch:** bilinearly resize the full image to 256×256. This retains broad context but distorts non-square aspect ratios.
3. **Native-crop branch:** extract nine contiguous central 128×128 crops covering a 384×384 region, without resizing. Features are computed separately for each crop.
4. Extract frozen ImageNet ResNet-18 `layer2` and `layer3` features. Apply 3×3 average pooling, bilinear feature-map alignment and per-layer L2 normalisation; concatenate and L2-normalise again to produce 384-dimensional local features.
5. Find exact nearest Euclidean distances to each branch's bona fide reference bank. Each branch score is the mean of the largest `ceil(0.10 × N)` local distances. The native branch pools distances from all nine crops before aggregation.
6. Normalise each branch score using its genuine-calibration median and interquartile range (IQR), take their maximum, and apply the saved threshold.

The two branches share frozen backbone weights but use separate reference banks and normalisation parameters. Banks were reduced using seeded random-priority sampling, rather than PatchCore's greedy coreset selection; this is therefore a **PatchCore-inspired** implementation. Inference does not fit or update any parameters.

## Calibration, results and limitations

The split used 650 bona fide reference images, 200 bona fide calibration images, 150 bona fide test images and 23 screen attacks. Each attack's matching bona fide source was reserved for testing to prevent source-pair leakage into fitting or calibration.

The 200 calibration images were divided into 100 for score normalisation and 100 for threshold selection. A nominal 5% bona fide rejection target selected the 96th ascending calibration score using a finite-sample order-statistic rule. Attack scores were not used to fit these parameters. The target does not guarantee a 5% rejection rate on new data.

| Metric | Exploratory test result |
|---|---:|
| ROC AUC | 0.949275 |
| Attacks detected | 13 / 23 |
| Attacks missed (APCER) | 10 / 23 (43.5%) |
| Bona fide images rejected (BPCER) | 2 / 150 (1.3%) |

These results are exploratory: the same small attack set informed repeated development decisions, and correlated captures reduce the effective sample size. Source grouping prevents direct pair leakage but does not remove development-time selection bias.

Resolution is a major confound: all bona fide images were 1024×1024, whereas attacks were larger. Native crops therefore cover different proportions of faces; fusion does not resolve this issue. Resizing or compression can suppress screen texture, and off-centre faces or different capture conditions may reduce performance. The observed AUC does not imply reliable attack rejection at the selected threshold.

## Submission contents and dependencies

| Path | Purpose |
|---|---|
| `infer.py`, `pad/` | Command-line entry point and inference implementation |
| `README.md` | Setup, inference, output format and method summary |
| `Dockerfile` | CPU runtime and checkpoint-loading build check |
| `.dockerignore` | Excludes unrelated files from the Docker build context |
| `requirements.txt`, `requirements-torch.txt`, `constraints.txt` | Pinned direct and transitive inference dependencies |
| `artifacts/fusion_model.pt` | Authoritative weights, banks, normalisation parameters and threshold |
| `artifacts/fusion_parameters.json` | Human-readable inference settings |
| `artifacts/model_info.json` | Export provenance and runtime metadata |
| `tests/` | Software contract tests using generated fixtures, not PAD accuracy tests |

Direct inference dependencies are PyTorch, torchvision, NumPy and Pillow. Notebooks, pandas and scikit-learn are not required for inference. The Dockerfile pins the Python version and the requirements pin package versions; small numerical differences can still occur across hardware and execution backends.

## Alternative: local Python on Linux CPU

Use the Python version recorded in `artifacts/model_info.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt -c constraints.txt
python -m pip check
python infer.py --input /path/to/images --output /path/to/scores.csv --recursive
```

## Software validation

The export workflow requires comparison against all 173 saved notebook test rows, checking raw branch scores, normalised scores, fusion scores and decisions. Its default absolute score tolerance is `1e-4`; decisions must match exactly.

The corrected Docker image has also been built and run on Windows with Docker Desktop. An offline smoke run processed 23 images without reported errors. CSV fusion calculations and threshold decisions were consistent; the eight attack images comparable with earlier rounded result tables had raw branch-score differences below `0.000026`. This is a smoke check, not a full 173-image Docker parity verification. Full Docker parity remains pending.

The optional `tests/` folder checks software behaviour without the original dataset. After installing the local Python dependencies above, run from the repository root:

```sh
python -m unittest discover -s tests -v
```

The tests use generated fixtures and do not measure PAD performance. They are not required for inference and are not included in the inference Docker image.
