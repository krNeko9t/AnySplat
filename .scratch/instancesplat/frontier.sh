#!/usr/bin/env bash
# 列出地图的前沿：status=open、assignee=-、且 blocked-by 全部已 closed 的票。
# 用法： .scratch/instancesplat/frontier.sh  [-a 全部票及状态]
set -euo pipefail
cd "$(dirname "$0")/tickets"

declare -A STATUS TITLE FILE
for f in *.md; do
  id=$(sed -n 's/^id: *//p'      "$f" | head -1)
  STATUS[$id]=$(sed -n 's/^status: *//p'   "$f" | head -1)
  TITLE[$id]=$(sed -n 's/^title: *//p'     "$f" | head -1)
  FILE[$id]=$f
done

if [[ "${1:-}" == "-a" ]]; then
  for id in $(printf '%s\n' "${!STATUS[@]}" | sort -V); do
    printf '%-4s %-8s %s\n' "$id" "${STATUS[$id]}" "${TITLE[$id]}"
  done
  exit 0
fi

echo "== 前沿（可以直接认领动手的票）=="
found=0
for id in $(printf '%s\n' "${!STATUS[@]}" | sort -V); do
  f=${FILE[$id]}
  [[ "${STATUS[$id]}" == "open" ]] || continue
  [[ "$(sed -n 's/^assignee: *//p' "$f" | head -1)" == "-" ]] || continue
  deps=$(sed -n 's/^blocked-by: *\[\(.*\)\] *$/\1/p' "$f" | head -1 | tr -d ' ' | tr ',' ' ')
  blocked=""
  for d in $deps; do
    [[ -n "$d" ]] || continue
    [[ "${STATUS[$d]:-open}" == "closed" ]] || blocked="$blocked $d"
  done
  if [[ -z "$blocked" ]]; then
    type=$(sed -n 's/^type: *wayfinder://p' "$f" | head -1)
    printf '  %-4s [%-9s] %-24s  tickets/%s\n' "$id" "$type" "${TITLE[$id]}" "$f"
    found=1
  fi
done
[[ $found == 1 ]] || echo "  （空——要么全被 blocked/认领了，要么地图走完了）"
