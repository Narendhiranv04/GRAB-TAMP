#!/bin/bash
ssh -i ~/keyfile -p 27617 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  long-horizon@0.tcp.in.ngrok.io "bash -lc '
cd ~/zsrun
b=\$(find out/zs_baseline -name result.json 2>/dev/null | wc -l)
z=\$(find out/zs_zs -name result.json 2>/dev/null | wc -l)
g=\$(wc -l < progress/done.txt 2>/dev/null || echo 0)
w=\$(pgrep -fc evaluate_vlm_zs 2>/dev/null || echo 0)
echo \"baseline=\$b/309 zs=\$z/309 total=\$((b+z))/618 groups_done=\$g/20 workers=\$w\"
'" 2>/dev/null
