#!/bin/bash
# tải song song trọng số timm từ HuggingFace (HTTP Range), mỗi file N đoạn, tự thử lại
cd "$(dirname "$0")"
N=8
dl() {
  repo=$1
  url="https://huggingface.co/timm/$repo/resolve/main/model.safetensors"
  size=$(curl -sI -L "$url" | grep -i "^content-length" | tail -1 | tr -dc '0-9')
  ch=$(( (size + N - 1) / N ))
  for i in $(seq 0 $((N-1))); do
    (s=$((i*ch)); e=$((s+ch-1)); [ $e -ge $size ] && e=$((size-1)); want=$((e-s+1))
     for a in $(seq 1 60); do have=0; [ -f $repo.part$i ] && have=$(stat -c %s $repo.part$i); [ $have -ge $want ] && break
       curl -s -L -m 900 -r $((s+have))-$e "$url" >> $repo.part$i; done) &
  done
  wait
  cat $(for i in $(seq 0 $((N-1))); do echo $repo.part$i; done) > $repo.safetensors && rm -f $repo.part*
  echo "$repo $(stat -c %s $repo.safetensors)/$size"
}
for r in "$@"; do dl $r & done
wait
echo ALL_DONE
