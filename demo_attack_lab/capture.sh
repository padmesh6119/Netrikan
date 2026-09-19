#!/bin/bash
pkill -f "http.server 8080" 2>/dev/null
pkill -f "http.server 4444" 2>/dev/null
python3 -m http.server 8080 > /dev/null 2>&1 &
python3 -m http.server 4444 > /dev/null 2>&1 &
echo "[*] HTTP listeners up on :8080 and :4444"

IFACE=$(ip route | awk '/default/{print $5; exit}')
echo "[*] Capturing on $IFACE -> attack_capture.pcap"
sudo tcpdump -i "$IFACE" -w /home/pontiff/SIH/demo_attack_lab/attack_capture.pcap &
echo $! > /home/pontiff/SIH/demo_attack_lab/tcpdump.pid
echo "[*] PID $(cat /home/pontiff/SIH/demo_attack_lab/tcpdump.pid) — run stop_capture.sh when done"
