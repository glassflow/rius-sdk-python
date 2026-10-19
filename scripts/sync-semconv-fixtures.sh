#!/usr/bin/env bash
# Copy the shared recorded spans from rius-semconv into this SDK's tests, at a
# pinned commit.
#
#   scripts/sync-semconv-fixtures.sh                     # ../rius-semconv @ origin/main
#   RIUS_SEMCONV=<checkout> REF=<commit|branch> scripts/sync-semconv-fixtures.sh
#
# rius-semconv holds every instrumentor's recording and its expected
# normalized result; tests/test_conformance.py replays them through this SDK's
# normalizer. Only the data is copied (manifests, recordings, expected results
# and the format README), never the recorders, plus the two Python modules that
# decode and encode a recording, vendored unchanged under tests/conformance/,
# and registry/media_parts.yaml, whose golden examples tests/test_media.py runs
# through the raw-payload strip. SOURCE records the commit the copy came from.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SRC="${RIUS_SEMCONV:-$ROOT/../rius-semconv}"
REF="${REF:-origin/main}"
DATA="$ROOT/tests/fixtures/semconv"
CODE="$ROOT/tests/conformance"

if ! git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1; then
  echo "no rius-semconv checkout at $SRC (set RIUS_SEMCONV)" >&2
  exit 1
fi
if [[ "$REF" == origin/* ]]; then
  git -C "$SRC" fetch --quiet origin
fi
COMMIT="$(git -C "$SRC" rev-parse "$REF^{commit}")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$SRC" archive "$COMMIT" fixtures registry/media_parts.yaml | tar -x -C "$WORK"

rm -rf "$DATA"
mkdir -p "$DATA/fixtures"
cp "$WORK/fixtures/README.md" "$DATA/fixtures/README.md"
cp "$WORK/registry/media_parts.yaml" "$DATA/media_parts.yaml"
for family in "$WORK"/fixtures/*/; do
  name="$(basename "$family")"
  [[ "$name" == _* ]] && continue
  mkdir -p "$DATA/fixtures/$name"
  for file in "$family"*.json; do
    case "$(basename "$file")" in
      package.json | package-lock.json) continue ;;
    esac
    cp "$file" "$DATA/fixtures/$name/"
  done
done
printf 'rius-semconv %s\n' "$COMMIT" >"$DATA/SOURCE"

mkdir -p "$CODE"
for module in _recording.py _replay.py; do
  {
    printf '# Vendored from rius-semconv %s fixtures/%s by\n' "$COMMIT" "$module"
    printf '# scripts/sync-semconv-fixtures.sh. Do not edit: change it in rius-semconv.\n'
    cat "$WORK/fixtures/$module"
  } >"$CODE/$module"
done

echo "synced $(find "$DATA/fixtures" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') families from rius-semconv ${COMMIT:0:7}"
