NETRIKAN DEMO — ATTACK LAB
==========================

TARGET (Padmesh's machine): 10.50.45.35

STEP 1 — Padmesh runs on his machine:
    chmod +x capture.sh stop_capture.sh
    bash capture.sh

STEP 2 — Friend runs on their machine:
    chmod +x attack.sh
    sudo bash attack.sh

    Requirements (install if missing):
        sudo apt install nmap hydra hping3 netcat curl

STEP 3 — Padmesh runs when attack is done:
    bash stop_capture.sh

STEP 4 — Feed into Netrikan:
    python ~/SIH/src/pcap_ingest.py \
        ~/SIH/demo_attack_lab/attack_capture.pcap \
        /tmp/demo_flows.csv
    Then upload demo_flows.csv in the dashboard.

ATTACK CLASSES COVERED:
    [1/4] Port scan      -> Class 1 (InitialAccess)
    [2/4] SSH brute      -> Class 1 (InitialAccess)
    [3/4] SYN flood      -> Class 2 (DoS)
    [4/4] C2 beaconing   -> Class 4 (Botnet)

NOTE: Both machines must be on the same network (LAN/hotspot).
      attack.sh needs sudo for hping3.
