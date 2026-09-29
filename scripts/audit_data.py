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

UWF = "data/UWF_Datasets"
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
con.execute(f"""create temp view zraw as select regexp_extract(filename,'parquet__([0-9-]+)',1) wk, * exclude(filename)
               from read_parquet('{UWF}/ZeekData22/parquet/*.parquet', filename=true)""")
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
show(f"""select count(*) n_rows, count(distinct uid) uids from read_csv('{UWF}/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)""")
for f in sorted(glob.glob(f"{UWF}/ZeekData22/csv/*.csv")):
    show(f"select '{os.path.basename(f)[38:46]}' part, count(*) n_rows, mitre_attack_tactics tactic from read_csv('{f}', sample_size=-1) group by 3 order by 2 desc")
show(f"""select (select count(distinct uid) from read_csv('{UWF}/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)) csv_uids,
        (select count(distinct uid) from read_csv('{UWF}/ZeekData22/csv/*.csv', sample_size=-1, union_by_name=true)
         where uid in (select uid from z)) csv_uids_in_parquet""")

# ---------------------------------------------------------------- Fall22
if os.path.isdir("data/ZeekDataFall22"):
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

else:
    h("ZeekDataFall22"); print("not present in data/ (removed by the user after the first audit)")

# ---------------------------------------------------------------- Z24
h("ZeekData24 (parquet)")
con.execute(f"""create temp view z24raw as select regexp_extract(filename,'zeekdata24__([0-9-]+)',1) wk, * exclude(filename)
                from read_parquet('{UWF}/ZeekData24/parquet/*.parquet', filename=true)""")
print("columns:", [c[0] for c in con.execute("describe z24raw").fetchall()])
show("""select wk, count(*) n_rows, count(distinct uid) uids, round(count(*)/count(distinct uid),2) rows_per_uid,
        to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1, count(distinct src_ip_zeek) src_ips, count(distinct dest_ip_zeek) dst_ips
        from z24raw group by 1 order by 1""")
show("select count(*) total_rows, (select count(*) from (select distinct * from z24raw)) distinct_rows from z24raw")
con.execute("create temp view z24 as select distinct * from z24raw")
print("tactic x week:")
show("select wk, label_tactic, count(*) n from z24 group by 1,2 order by 1,3 desc")
print("technique x tactic (attack rows):")
show("select label_technique, label_tactic, count(*) n from z24 where label_tactic<>'none' group by 1,2 order by 1,3 desc")
print("uids carrying more than one attack row (multi-tactic labelling of one flow):")
show("select count(*) n_uids from (select uid from z24 group by 1 having count(*)>1)")
print("host overlap between benign weeks and attack weeks:")
show("""with b as (select src_ip_zeek s from z24 where label_tactic='none' union select dest_ip_zeek from z24 where label_tactic='none'),
        a as (select src_ip_zeek s from z24 where label_tactic<>'none' union select dest_ip_zeek from z24 where label_tactic<>'none'),
        bs as (select distinct src_ip_zeek s from z24 where label_tactic='none'), as_ as (select distinct src_ip_zeek s from z24 where label_tactic<>'none')
        select (select count(*) from b) benign_hosts, (select count(*) from a) attack_hosts, (select count(*) from a join b using(s)) hosts_in_both,
               (select count(*) from bs) benign_src, (select count(*) from as_) attack_src, (select count(*) from as_ join bs using(s)) attack_src_also_benign_src""")
print("attack sources:")
show("""select src_ip_zeek, count(distinct uid) flows, count(distinct dest_ip_zeek) victims, string_agg(distinct label_technique, ', ') techniques
        from z24 where label_tactic<>'none' group by 1 order by 2 desc""")
print("chronology for one attacker (143.88.7.11): technique first/last time per week:")
show("""select wk, label_technique tech, count(distinct uid) flows, to_timestamp(min(ts)) first_t, to_timestamp(max(ts)) last_t
        from z24 where src_ip_zeek='143.88.7.11' and label_tactic<>'none' group by 1,2 order by 1, min(ts)""")
print("order in which techniques first appear per (attacker, victim, week):")
show("""with f as (select wk, src_ip_zeek s, dest_ip_zeek d, label_technique t, min(ts) t0 from z24 where label_tactic<>'none' and label_technique<>'Duplicate' group by 1,2,3,4),
        seq as (select wk,s,d,string_agg(t, ' > ' order by t0) pattern from f group by 1,2,3)
        select pattern, count(*) n_pairs from seq group by 1 order by 2 desc limit 12""")
print("exfiltration (T1048) sources/victims:")
show("select src_ip_zeek, dest_ip_zeek, wk, count(distinct uid) n, to_timestamp(min(ts)) t0 from z24 where label_technique='T1048' group by 1,2,3 order by 3")
print("ts vs datetime offset (hours):")
show("select round(min(ts-epoch(datetime))/3600,2) mn, round(max(ts-epoch(datetime))/3600,2) mx from z24")
print("benign vs attack rows / sources:")
show("select label_tactic='none' is_benign, count(*) n_rows, count(distinct src_ip_zeek) src_ips, to_timestamp(min(ts)) t0, to_timestamp(max(ts)) t1 from z24 group by 1")
print("benign-week host activity, flows per source-day quantiles:")
show("""with d as (select src_ip_zeek, date_trunc('day',to_timestamp(ts)) day, count(*) n from z24 where label_tactic='none' group by 1,2)
        select count(*) source_days, quantile_cont(n,0.5) median_flows, quantile_cont(n,0.9) p90_flows from d""")


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
