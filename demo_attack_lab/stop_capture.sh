#!/bin/bash
sudo kill $(cat /home/pontiff/SIH/demo_attack_lab/tcpdump.pid) 2>/dev/null
sleep 1
echo "[*] Capture stopped. Trimming to 3000 packets..."
tcpdump -r /home/pontiff/SIH/demo_attack_lab/attack_capture.pcap \
  -w /home/pontiff/SIH/demo_attack_lab/attack_small.pcap -c 3000 2>/dev/null
echo "[*] Converting to flows..."
python3 /home/pontiff/SIH/src/pcap_ingest.py \
  /home/pontiff/SIH/demo_attack_lab/attack_small.pcap \
  /home/pontiff/SIH/demo_attack_lab/attack_flows.csv
echo "[*] Done — upload attack_flows.csv to Netrikan"
