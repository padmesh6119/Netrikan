"""Phase 0 data audit. Read-only over data/; prints everything docs/DATA_AUDIT.md cites.

Usage: make audit
"""
import glob
import os

import duckdb
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

con = duckdb.connect()
con.execute("set TimeZone='UTC'")
show = lambda q: print(con.execute(q).fetchdf().to_string())


def h(t):
    print(f"\n{'=' * 8} {t}")


# ---------------------------------------------------------------- DAPT2020
h("DAPT2020")
fs = sorted(glob.glob("data/dapt2020/csv/*.csv"))
hdr = pd.read_csv(fs[0], nrows=0).columns.tolist()
dfs = []
for f in fs:
    headed = open(f).readline().startswith("Flow ID")
    d = pd.read_csv(f, header=0 if headed else None, names=None if headed else hdr, low_memory=False)
    d["file"] = os.path.basename(f).replace("enp0s3-", "").replace(".pcap_Flow.csv", "")
    d["has_header"] = headed
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df["ts"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
print("columns:", len(hdr), "rows:", len(df))
print("files without header row:", sorted(df.loc[~df.has_header, "file"].unique()))
g = df.groupby("file").agg(
    n=("ts", "size"), t0=("ts", "min"), t1=("ts", "max"),
    time_ordered=("ts", lambda s: s.is_monotonic_increasing),
    dup_flow_id=("Flow ID", lambda s: s.duplicated().sum()),
    src_ips=("Src IP", "nunique"), dst_ips=("Dst IP", "nunique"))
print(g)
print("full-row duplicates:", df.drop(columns=["file", "has_header"]).duplicated().sum())
print("rows == 1,048,575:", (df.groupby("file").size() == 1048575).any())
print("\nStage x Activity:"); print(df.groupby(["Stage", "Activity"]).size())
print("\nStage x file:"); print(pd.crosstab(df.file, df.Stage))
df["mal"] = ~df.Stage.isin(["BENIGN", "Benign"])
m = df[df.mal]
print("\nmalicious flows:", len(m), " malicious src IPs:", m["Src IP"].nunique())
print(m.groupby("Src IP").agg(n=("ts", "size"), stages=("Stage", lambda s: sorted(set(s))),
                              days=("ts", lambda s: s.dt.date.nunique()), t0=("ts", "min"), t1=("ts", "max")))
for side in ["Src IP", "Dst IP"]:
    k = m.groupby(side).Stage.nunique()
    print(f"{side}: hosts with >=1 attack stage: {len(k)}, with >1 distinct attack stage: {(k > 1).sum()}")
pair = m.groupby(["Src IP", "Dst IP"]).Stage.nunique()
print(f"(Src,Dst) pairs with attack: {len(pair)}, with >1 stage: {(pair > 1).sum()}")

# ---------------------------------------------------------------- Z22
h("ZeekData22 (parquet)")
con.execute("""create temp view zraw as select regexp_extract(filename,'parquet__([0-9-]+)',1) wk, * exclude(filename)
               from read_parquet('data/ZeekData22/parquet/*.parquet', filename=true)""")
print("columns:", [c[0] for c in con.execute("describe zraw").fetchall()])
show("""select wk, count(*) n_rows, count(distinct uid) uids, round(count(*)/count(distinct uid),1) rows_per_uid,
        min(datetime) dt_min, max(datetime) dt_max, count(distinct src_ip_zeek) src_ips, count(distinct dest_ip_zeek) dst_ips
        from zraw group by 1 order by 1""")
show("select count(*) total_rows, count(distinct uid) uids, (select count(*) from (select distinct * from zraw)) distinct_rows from zraw")
print("uid multiplicity in attack week:")
show("select c copies, count(*) n_uids from (select uid, count(*) c from zraw where wk='2022-02-06' group by 1) group by 1 order by 1")
con.execute("create temp view z as select distinct * from zraw")
print("\nafter exact-row dedup: label counts")
show("select label_tactic, count(*) n, count(distinct uid) uids, count(distinct src_ip_zeek) src_ips from z group by 1 order by 2 desc")
print("distinct-row count per uid after dedup (uids with >1 label row):")
show("select count(*) from (select uid from z group by 1 having count(*)>1)")
print("\nweeks x label (after dedup):")
show("select wk, label_tactic, count(*) n from z group by 1,2 order by 1,3 desc")
print("\nts (epoch) vs datetime offset, hours (should be constant tz shift):")
show("select round(avg(ts - epoch(datetime))/3600,2) mean_h, round(min(ts - epoch(datetime))/3600,2) min_h, round(max(ts - epoch(datetime))/3600,2) max_h from z")
print("\nts range (UTC via epoch):")
show("select to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1 from z")
print("\nhost overlap: benign-src vs attack-src")
show("""with b as (select distinct src_ip_zeek s from z where label_tactic='none'), a as (select distinct src_ip_zeek s from z where label_tactic<>'none')
        select (select count(*) from b) benign_hosts, (select count(*) from a) attack_hosts, (select count(*) from a join b using(s)) in_both""")
print("\nbenign vs attack calendar overlap:")
show("""select label_tactic='none' is_benign, to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1 from z group by 1""")
print("\nattack hosts (dedup) with tactic sets:")
show("""select src_ip_zeek, count(*) flows, count(distinct label_tactic) n_tactics, string_agg(distinct label_tactic, ', ') tactics
        from z where label_tactic<>'none' group by 1 order by 2 desc""")
print("\nbenign hosts per day:")
show("select date_trunc('day', to_timestamp(ts)) d, count(*) flows, count(distinct src_ip_zeek) hosts from z where label_tactic='none' group by 1 order by 1")

h("ZeekData22 (csv) - relation to parquet")
show("""select count(*) n_rows, count(distinct uid) uids from read_csv('data/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)""")
for f in sorted(glob.glob("data/ZeekData22/csv/*.csv")):
    show(f"select '{os.path.basename(f)[38:46]}' part, count(*) n_rows, mitre_attack_tactics tactic from read_csv('{f}', sample_size=-1) group by 3 order by 2 desc")
show("""select (select count(distinct uid) from read_csv('data/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)) csv_uids,
        (select count(distinct uid) from read_csv('data/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)
         where uid in (select uid from z)) csv_uids_in_parquet""")

# ---------------------------------------------------------------- Fall22
h("ZeekDataFall22")
con.execute("""create temp view f as select regexp_extract(filename,'csv__([A-Za-z_]+)__part',1) file_tactic, * exclude(filename)
               from read_csv('data/ZeekDataFall22/csv_by_tactic/*.csv', sample_size=-1, union_by_name=true, filename=true)""")
print("columns:", [c[0] for c in con.execute("describe f").fetchall()])
show("""select file_tactic, count(*) n_rows, count(distinct uid) uids, to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1, count(distinct src_ip_zeek) src_ips
        from f group by 1 order by 2 desc""")
show("select label_binary, label_tactic='none' is_none, count(*) n from f group by 1,2 order by 1,2")
print("\nparquet folder (only 1 week present):")
show("select label_tactic, label_binary, count(*) n, to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1 from read_parquet('data/ZeekDataFall22/parquet/*.parquet') group by 1,2")
print("\nIndependence vs Z22 (uid / community_id / ts):")
show("""select (select count(distinct uid) from f) fall_uids,
        (select count(distinct uid) from f where uid in (select uid from z)) uids_in_z22,
        (select count(distinct uid) from f where label_tactic='none') fall_benign_uids,
        (select count(distinct uid) from f where label_tactic='none' and uid in (select uid from z)) fall_benign_uids_in_z22,
        (select count(distinct uid) from f where label_tactic<>'none') fall_attack_uids,
        (select count(distinct uid) from f where label_tactic<>'none' and uid in (select uid from z)) fall_attack_uids_in_z22""")
print("\nFall22 attack hosts:")
show("""select src_ip_zeek, count(*) flows, count(distinct label_tactic) n_tactics, string_agg(distinct label_tactic, ', ') tactics,
        to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1 from f where label_tactic<>'none' group by 1 order by 2 desc""")
show("""with fa as (select distinct src_ip_zeek s from f where label_tactic<>'none'), za as (select distinct src_ip_zeek s from z where label_tactic<>'none'),
        zb as (select distinct src_ip_zeek s from z where label_tactic='none'), fb as (select distinct src_ip_zeek s from f where label_tactic='none')
        select (select count(*) from fa) fall_attack_hosts, (select count(*) from fa join za using(s)) also_z22_attack_hosts,
        (select count(*) from fa join zb using(s)) also_z22_benign_hosts, (select count(*) from fb) fall_benign_hosts,
        (select count(*) from fb join zb using(s)) fall_benign_hosts_in_z22""")
show("select label_technique, count(*) n from f group by 1 order by 2 desc")

# ---------------------------------------------------------------- CIC2018
h("cic-2018-improved (Botnet-Friday)")
C = "read_parquet('data/cic-2018-improved/Botnet-Friday-02-03-2018.parquet')"
cols = [c[0] for c in con.execute(f"describe select * from {C}").fetchall()]
print("n columns:", len(cols))
print("has IP/time/id columns:", [c for c in cols if any(k in c.lower() for k in ("ip", "time", "stamp", "id", "port")) and "Bulk" not in c and "Idle" not in c][:10])
show(f'select count(*) n_rows from {C}')
show(f'select "Label", count(*) n from {C} group by 1 order by 2 desc')
show(f"select count(*) n_rows, (select count(*) from (select distinct * from {C})) distinct_rows from {C}")
print("files present:", os.listdir("data/cic-2018-improved"))
