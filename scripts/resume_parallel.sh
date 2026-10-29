#!/usr/bin/env bash
# Append the tail of a partially downloaded file using N parallel HTTP range
# requests. Useful when the origin throttles each connection (cs.toronto.edu
# gives ~80 KB/s per connection but ~500 KB/s across 8-16).
#
# Safety: the target file is only modified after every segment has arrived at
# its full expected length. On any failure the file is left exactly as it was.
set -uo pipefail

FILE="$1"
URL="$2"
TOTAL="${3:-}"
N="${4:-8}"

if [ -z "$TOTAL" ]; then
  TOTAL=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
fi
SPLIT_DIR="${TMPDIR:-/tmp}/parallel-resume.$$"

have=$(stat -c %s "$FILE")
if [ "$have" -gt "$TOTAL" ]; then
  echo "file is larger than expected; refusing to touch it" >&2
  exit 1
fi
if [ "$have" -eq "$TOTAL" ]; then
  echo "already complete: $have bytes"
  exit 0
fi

mkdir -p "$SPLIT_DIR"
remaining=$((TOTAL - have))
chunk=$(( (remaining + N - 1) / N ))
echo "have=$have total=$TOTAL remaining=$remaining chunk=$chunk parts=$N"

: > "$SPLIT_DIR/manifest"
i=0
start=$have
while [ "$start" -lt "$TOTAL" ]; do
  end=$((start + chunk - 1))
  [ "$end" -ge "$((TOTAL - 1))" ] && end=$((TOTAL - 1))
  echo "$i $start $end $((end - start + 1))" >> "$SPLIT_DIR/manifest"
  i=$((i + 1))
  start=$((end + 1))
done

fetch() {
  local idx="$1" s="$2" e="$3" expect="$4" attempt
  for attempt in $(seq 1 25); do
    if curl -sS -L --fail --connect-timeout 20 --max-time 900 \
        --retry 5 --retry-delay 2 --retry-all-errors \
        -r "$s-$e" -o "$SPLIT_DIR/part$idx" "$URL" 2>>"$SPLIT_DIR/err.log" \
      && [ "$(stat -c %s "$SPLIT_DIR/part$idx" 2>/dev/null || echo 0)" -eq "$expect" ]; then
      return 0
    fi
    sleep $(( attempt < 5 ? attempt * 2 : 10 ))
  done
  echo "part$idx permanently failed" >&2
  return 1
}

while read -r idx s e expect; do
  fetch "$idx" "$s" "$e" "$expect" &
done < "$SPLIT_DIR/manifest"

fail=0
for job in $(jobs -p); do
  wait "$job" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "some segments failed; target file left unchanged" >&2
  tail -5 "$SPLIT_DIR/err.log" >&2 2>/dev/null
  echo "(partial segments kept in $SPLIT_DIR for inspection)" >&2
  exit 1
fi

while read -r idx _ _ expect; do
  got=$(stat -c %s "$SPLIT_DIR/part$idx")
  if [ "$got" -ne "$expect" ]; then
    echo "part$idx short: $got != $expect; target file left unchanged" >&2
    exit 1
  fi
  cat "$SPLIT_DIR/part$idx" >> "$FILE"
done < "$SPLIT_DIR/manifest"

rm -rf "$SPLIT_DIR"
echo "final size: $(stat -c %s "$FILE") (expected $TOTAL)"
