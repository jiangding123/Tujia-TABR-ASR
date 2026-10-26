#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 <src_dir> <dst_dir> [sampling_rate=16000]" >&2
  exit 1
fi
src_dir=$1
dst_dir=$2
sr=${3:-16000}
speeds=("0.9" "1.0" "1.1")

command -v sox >/dev/null || { echo "Missing command: sox" >&2; exit 1; }
[[ -s "$src_dir/wav.scp" && -s "$src_dir/text" ]] || {
  echo "Source requires nonempty wav.scp and text: $src_dir" >&2
  exit 1
}
[[ "$sr" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid sampling rate: $sr" >&2; exit 1; }
# Prevent overwriting any existing destination or source audio.
if [[ -e "$dst_dir" || -L "$dst_dir" ]]; then
  echo "Destination exists; choose a new directory: $dst_dir" >&2
  exit 1
fi
mkdir -p "$dst_dir/wav"
: > "$dst_dir/wav.scp"
: > "$dst_dir/text"

for sp in "${speeds[@]}"; do
  tag=$(echo "$sp" | tr -d '.')
  echo "Processing pitch-preserving tempo $sp ..."
  while read -r utt wav; do
    [[ -n "$utt" ]] || continue
    [[ -f "$wav" ]] || { echo "Missing input WAV: $wav" >&2; exit 1; }
    new_utt="${utt}_sp${tag}"
    out_wav="$dst_dir/wav/${new_utt}.wav"
    if [[ "$sp" == "1.0" ]]; then
      ln -s "$(realpath "$wav")" "$out_wav"
    else
      sox "$wav" "$out_wav" tempo -s "$sp" rate "$sr"
    fi
    echo "$new_utt $out_wav" >> "$dst_dir/wav.scp"
  done < "$src_dir/wav.scp"
  awk -v tag="_sp${tag}" '{
    utt=$1
    $1=""
    sub(/^ /, "")
    print utt tag " " $0
  }' "$src_dir/text" >> "$dst_dir/text"
done

if [[ -f "$src_dir/utt2spk" ]]; then
  : > "$dst_dir/utt2spk"
  for sp in "${speeds[@]}"; do
    tag=$(echo "$sp" | tr -d '.')
    awk -v tag="_sp${tag}" '{print $1 tag " " $2}' \
      "$src_dir/utt2spk" >> "$dst_dir/utt2spk"
  done
  awk '{
    spk=$2
    utt=$1
    utts[spk]=utts[spk]" "utt
  }
  END {
    for (spk in utts) print spk utts[spk]
  }' "$dst_dir/utt2spk" | sort > "$dst_dir/spk2utt"
fi

echo "Done: pitch-preserving tempo augmentation."
echo "Original wavs: $(wc -l < "$src_dir/wav.scp")"
echo "New wavs:      $(wc -l < "$dst_dir/wav.scp")"
echo "Original text: $(wc -l < "$src_dir/text")"
echo "New text:      $(wc -l < "$dst_dir/text")"
