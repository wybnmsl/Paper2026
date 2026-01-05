#!/usr/bin/env bash
set -e

DEST="${1:-zTSP/external/tsplib_gt}"
mkdir -p "$DEST" && cd "$DEST"

HEI="https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp"
RICE="https://softlib.rice.edu/pub/tsplib/tsp"

# <1000 节点、常用且有官方真值的实例（可自行增删）
NAMES=(
  att48 berlin52 eil51 eil76 eil101
  ch130 ch150
  pr76 pr107 pr124 pr136 pr144 pr152
  rat195 rd100 rd400
  pcb442 d493 gil262
  gr120 gr137 gr202 gr229 gr431 gr48
  st70 ts225 tsp225
  bier127 ts225
)

download_one () {
  local name="$1"
  # tsp
  for base in "$HEI" "$RICE"; do
    if curl -fL -o "${name}.tsp.gz" "${base}/${name}.tsp.gz" >/dev/null 2>&1; then
      break
    fi
  done
  # opt.tour
  for base in "$HEI" "$RICE"; do
    if curl -fL -o "${name}.opt.tour.gz" "${base}/${name}.opt.tour.gz" >/dev/null 2>&1; then
      break
    fi
  done
}

echo "Downloading TSPLIB files into: $(pwd)"
for n in "${NAMES[@]}"; do
  echo " - ${n}"
  download_one "$n" || true
done

echo "Done. Files:"
ls -1 | head -n 20
