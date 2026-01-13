#!/bin/sh
set -e

if [ $# -lt 1 ]; then
  echo "usage: verify_beam.sh <path_to_erl> [function] [arity]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$1
FUNC=${2:-add}
ARITY=${3:-2}
BASE=$(basename "$SRC" .erl)

OUTDIR="$ROOT/tmp/erlc_out"
mkdir -p "$OUTDIR"

erlc -o "$OUTDIR" "$SRC"
erlc_beam="$OUTDIR/$BASE.beam"

cargo -q run --manifest-path "$ROOT/Cargo.toml" -- build "$SRC"
otp_beam="${SRC%.erl}.beam"

"$ROOT/scripts/beam_fun_compare.escript" "$otp_beam" "$erlc_beam" "$FUNC" "$ARITY"
