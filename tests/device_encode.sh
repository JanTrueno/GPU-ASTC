#!/bin/sh
# Runs on the device. Encodes every .png in a folder at 4x4, 5x5 and 6x6 into ./q and packs them as q.tar,
# for tests/device_score.py on a PC: the only check that shows a driver producing valid but wrong blocks
# (-upload passes those).
# usage: device_encode.sh <gpuastc binary> <folder with .png files> [extra gpuastc options]
# Set SDL_VIDEODRIVER and friends as the device needs before calling.
BIN=$1; IN=$2; shift 2
rm -rf q; mkdir -p q
for b in 4 5 6; do
  for f in "$IN"/*.png; do
    n=$(basename "$f" .png)
    "$BIN" "$f" "q/${n}_$b.astc" -b $b -force -nocache "$@" > /dev/null 2>&1 || echo "FAILED $n ${b}x${b}"
  done
done
tar cf q.tar q && echo "$(ls q | wc -l) files in q.tar"
