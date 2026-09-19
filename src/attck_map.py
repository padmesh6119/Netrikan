BENIGN, INITIAL_ACCESS, DOS, LATERAL, C2, EXFIL = 0, 1, 2, 3, 4, 5

STAGE_INFO = {
    BENIGN: {
        "name": "Benign",
        "short": "benign",
        "mitre": "—",
        "color": "#22c55e",
        "description": "Normal traffic. No attack activity.",
        "techniques": [],
    },
    INITIAL_ACCESS: {
        "name": "Initial Access",
        "short": "initial access",
        "mitre": "TA0001",
        "capec": ["CAPEC-49 Password Brute Forcing", "CAPEC-112 Brute Force", "CAPEC-1 Accessing/Intercepting/Modifying HTTP"],
        "color": "#eab308",
        "description": "Adversary gaining a foothold — brute force, credential stuffing, public-facing exploitation.",
        "techniques": ["T1110 Brute Force", "T1190 Exploit Public-Facing App", "T1078 Valid Accounts"],
    },
    DOS: {
        "name": "DoS / Impact",
        "short": "DoS",
        "mitre": "TA0040",
        "capec": ["CAPEC-125 Flooding", "CAPEC-469 HTTP DoS"],
        "color": "#f97316",
        "description": "Service degradation via volumetric or slow-rate resource exhaustion.",
        "techniques": ["T1498 Network DoS", "T1499 Endpoint DoS"],
    },
    LATERAL: {
        "name": "Lateral Movement",
        "short": "lateral movement",
        "mitre": "TA0008",
        "capec": ["CAPEC-555 Remote Services with Stolen Credentials", "CAPEC-561 Windows Admin Shares"],
        "color": "#a855f7",
        "description": "Attacker moving between internal hosts — remote services, tool transfer, internal recon.",
        "techniques": ["T1021 Remote Services", "T1570 Lateral Tool Transfer", "T1083 File Discovery"],
    },
    C2: {
        "name": "Command & Control",
        "short": "command & control",
        "mitre": "TA0011",
        "capec": ["CAPEC-664 Server Side Request Forgery", "CAPEC-183 IMAP/SMTP Command Injection"],
        "color": "#ef4444",
        "description": "Compromised host beaconing to attacker infrastructure.",
        "techniques": ["T1071 Application Layer Protocol", "T1090 Proxy", "T1102 Web Service"],
    },
    EXFIL: {
        "name": "Exfiltration",
        "short": "exfiltration",
        "mitre": "TA0010",
        "capec": ["CAPEC-158 Sniffing Network Traffic", "CAPEC-117 Interception"],
        "color": "#991b1b",
        "description": "Data leaving the network to attacker-controlled infrastructure.",
        "techniques": ["T1041 Exfil Over C2 Channel", "T1048 Exfil Over Alternative Protocol"],
    },
}

N_STAGES = 6

# model emits 5 classes; EXFIL is forecast-only (never observed as a label)
MODEL_TO_CHAIN = {0: BENIGN, 1: INITIAL_ACCESS, 2: DOS, 3: LATERAL, 4: C2}

# damage is already done at or beyond these stages — a signature IDS fires here
DAMAGE_STAGES = {DOS, C2, EXFIL}


def get_stage(sid: int) -> dict:
    return STAGE_INFO.get(sid, STAGE_INFO[BENIGN])


def short(sid: int) -> str:
    return STAGE_INFO.get(sid, STAGE_INFO[BENIGN])["short"]
