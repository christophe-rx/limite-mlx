#!/usr/bin/env bash
# Set up a Limite model for LM Studio on a Mac, end to end:
# download -> convert to MLX format -> install -> make LM Studio able to load it.
#
#   ./install.sh                       # limite-1b-violetto (maths reasoning)
#   ./install.sh limite-1b-base        # the general base model
#
# Re-run after LM Studio updates its MLX runtime (see the note at the end).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="paradigma-inc/${1:-limite-1b-violetto}"
SHORT="${REPO##*/}"
SRC="models/$SHORT"
OUT="models/$SHORT-mlx"
DEST="$HOME/.lmstudio/models/paradigma-inc/$SHORT-mlx"
VENV="${MLX_VENV:-.venv-mlx}"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

# ---------------------------------------------------------------- interpreter
# mlx-lm >= 0.31 is required. It needs transformers>=5, which needs python>=3.10;
# on an older python the resolver quietly picks a pre-0.31 mlx-lm instead.
mlx_ver () {
  "$1" -c 'import importlib.metadata as md
try:
    import mlx_lm; print(getattr(mlx_lm,"__version__",None) or md.version("mlx-lm"))
except Exception: pass' 2>/dev/null | tail -1
}
ok_ver () { "$PY" -c 'import sys
try: p=tuple(int(x) for x in sys.argv[1].split(".")[:2])
except Exception: p=(0,0)
sys.exit(0 if p>=(0,31) else 1)' "$1" 2>/dev/null; }

MLXPY=""; MLXVER=""
for c in "$PY" "$VENV/bin/python"; do
  v=$(mlx_ver "$c" || true)
  if [ -n "$v" ] && ok_ver "$v"; then MLXPY="$c"; MLXVER="$v"; break; fi
done
if [ -z "$MLXPY" ]; then
  command -v uv >/dev/null 2>&1 || {
    echo "error: need mlx-lm >= 0.31 (or uv, to create an environment for it)."
    echo "  pip install 'mlx-lm>=0.31'  then re-run, or:"
    echo "  PYTHON=/path/to/python ./install.sh"
    exit 1; }
  echo "==> creating $VENV (python 3.12, mlx-lm>=0.31)"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" "mlx-lm>=0.31"
  MLXPY="$VENV/bin/python"; MLXVER=$(mlx_ver "$MLXPY" || true)
  { [ -n "$MLXVER" ] && ok_ver "$MLXVER"; } || {
    echo "error: $VENV has no usable mlx-lm (got '${MLXVER:-none}')"; exit 1; }
fi
echo "==> using $MLXPY (mlx-lm $MLXVER)"

# ------------------------------------------------------------------- download
if [ -f "$SRC/config.json" ]; then
  echo "==> $SRC already present, skipping download"
else
  echo "==> downloading $REPO (~2 GB)"
  mkdir -p "$SRC"
  "$MLXPY" - "$REPO" "$SRC" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
PYEOF
fi

# -------------------------------------------------------------------- convert
# LM Studio picks a runtime from the safetensors metadata. A PyTorch-saved
# checkpoint says {"format": "pt"} and is rejected as `torchSafetensors`;
# mlx_lm.convert re-saves it as {"format": "mlx"}.
fmt=$("$PY" - "$SRC" <<'PYEOF'
import glob, json, struct, sys
f = sorted(glob.glob(sys.argv[1] + "/model*.safetensors"))
if not f: print("none"); raise SystemExit
with open(f[0], "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    print((json.loads(fh.read(n)).get("__metadata__") or {}).get("format", "unknown"))
PYEOF
)
if [ "$fmt" = "mlx" ]; then
  echo "==> $SRC is already MLX format"
  OUT="$SRC"
else
  echo "==> converting to MLX format -> $OUT"
  cp -f limite_mlx.py "$SRC/limite.py"
  "$PY" - "$SRC" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"
c = json.loads(p.read_text()); c["model_file"] = "limite.py"
p.write_text(json.dumps(c, indent=2))
PYEOF
  rm -rf "$OUT"
  "$MLXPY" -m mlx_lm convert --hf-path "$SRC" --mlx-path "$OUT"
fi

# -------------------------------------------------------------------- install
echo "==> installing -> $DEST"
mkdir -p "$DEST"
cp "$OUT"/*.json "$OUT"/*.jinja "$DEST"/ 2>/dev/null || true
cp "$OUT"/*.safetensors "$DEST"/

# ------------------------------------------------- teach LM Studio the arch
# LM Studio calls mlx_lm.utils.load() without trust_remote_code, so it will not
# execute a limite.py shipped next to the weights. Putting the module where
# mlx-lm keeps built-in architectures needs no such permission.
echo "==> installing the architecture into LM Studio's MLX runtime"
found=0
while IFS= read -r d; do
  [ -d "$d" ] || continue
  cp -f limite_mlx.py "$d/limite.py"
  echo "    $d/limite.py"
  found=$((found + 1))
done < <(find "$HOME/.lmstudio" -type d -path '*/mlx_lm/models' 2>/dev/null | sort)

if [ "$found" -eq 0 ]; then
  echo "    no MLX runtime found under ~/.lmstudio"
  echo "    Install LM Studio's MLX runtime first (it ships with the app, or"
  echo "    run: lms runtime ls), then re-run this script."
  exit 1
fi

# with the architecture built in, the model must not request custom code
"$PY" - "$DEST" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"
c = json.loads(p.read_text())
if c.pop("model_file", None) is not None:
    p.write_text(json.dumps(c, indent=2))
PYEOF
rm -f "$DEST/limite.py"

cat <<MSG

Done. Restart LM Studio, then load:

    paradigma-inc/$SHORT-mlx

Note: LM Studio bundles its own copy of mlx-lm, so a runtime update will
remove the architecture file. If the model stops loading with
"Model type limite not supported", just re-run this script.
MSG
