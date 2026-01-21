import sqlite3, pandas as pd, numpy as np, math, os

db_path = "/Users/evankellener/Desktop/MMA-AI/sqlite_scrapper.db"
conn = sqlite3.connect(db_path)

# Load core tables
match = pd.read_sql_query("SELECT * FROM ufc_fighter_match_stats;", conn)
events = pd.read_sql_query("SELECT jevent, DATE as event_date FROM ufc_event_details;", conn)
results = pd.read_sql_query("SELECT jevent, jbout, WEIGHTCLASS as weightclass, weightindex, ROUND as finish_round, TIME as finish_time, `TIME FORMAT` as time_format, METHOD as method FROM ufc_fight_results;", conn)
wlko = pd.read_sql_query("SELECT jevent, jbout, jfighter, DATE as event_date, win, loss, udec, mdec, sdec, ko, subw, fight_time_minutes FROM ufc_winlossko;", conn)
fighters = pd.read_sql_query("SELECT jfighter, HEIGHT, REACH, STANCE, DOB, sex, weightindex as fighter_weightindex FROM ufc_fighter_tott;", conn)
wclu = pd.read_sql_query("SELECT weightindex, weightclass, weight, sex FROM weightclass_lookup;", conn)

# Normalize types
events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
wlko["event_date"] = pd.to_datetime(wlko["event_date"], errors="coerce")
fighters["DOB"] = pd.to_datetime(fighters["DOB"], errors="coerce")

# Join match stats to event date and fight metadata
df = match.merge(events, on="jevent", how="left")
df = df.merge(results[["jevent","jbout","weightclass","weightindex","finish_round","finish_time","time_format","method"]], on=["jevent","jbout"], how="left")
df = df.merge(fighters[["jfighter","HEIGHT","REACH","STANCE","DOB","sex","fighter_weightindex"]], on="jfighter", how="left")

# Join outcomes and duration from ufc_winlossko (more reliable for duration minutes)
df = df.merge(wlko[["jevent","jbout","jfighter","win","loss","udec","mdec","sdec","ko","subw","fight_time_minutes","event_date"]].rename(columns={"event_date":"event_date_wlko"}), 
              on=["jevent","jbout","jfighter"], how="left")

# Prefer wlko event_date if missing
df["event_date"] = df["event_date"].fillna(df["event_date_wlko"])
df = df.drop(columns=["event_date_wlko"])

# Basic identifiers
df["fight_id"] = df["jevent"].astype(str) + "::" + df["jbout"].astype(str)
df["fighter_id"] = df["jfighter"].astype(str)

# Duration
df["fight_time_minutes"] = pd.to_numeric(df["fight_time_minutes"], errors="coerce")
df["timesec"] = df["fight_time_minutes"] * 60.0
# Fallback: if missing, approximate from finish_round + finish_time (MM:SS). If still missing, use 15 minutes.
def parse_mmss(x):
    if pd.isna(x): 
        return np.nan
    s = str(x).strip()
    if ":" not in s:
        return np.nan
    mm, ss = s.split(":")
    try:
        return int(mm)*60 + int(ss)
    except:
        return np.nan

df["finish_round_num"] = pd.to_numeric(df["finish_round"], errors="coerce")
finish_sec = df["finish_time"].apply(parse_mmss)
# scheduled minutes from time_format like '3 Rnd (5-5-5)'
def parse_time_format(tf):
    if pd.isna(tf): 
        return np.nan
    s = str(tf)
    if "Rnd" in s:
        try:
            n = int(s.split("Rnd")[0].strip())
            return n * 5 * 60
        except:
            return np.nan
    return np.nan

sched_sec = df["time_format"].apply(parse_time_format)
approx_timesec = (df["finish_round_num"] - 1) * 5 * 60 + finish_sec
df["timesec"] = df["timesec"].fillna(approx_timesec)
df["timesec"] = df["timesec"].fillna(sched_sec)
df["timesec"] = df["timesec"].fillna(15*60)

# Ensure numeric for match stats
numeric_cols = ["kd","rev","ctrl",
                "sigstracc","sigstratt","tdacc","tdatt",
                "subatt","totalacc","totalatt",
                "headacc","headatt","bodyacc","bodyatt","legacc","legatt",
                "distacc","distatt","clinchacc","clinchatt","groundacc","groundatt"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# Compute opponent within fight: the other fighter row in same fight_id
opp_map = df[["fight_id","fighter_id"]].merge(df[["fight_id","fighter_id"]], on="fight_id", suffixes=("_a","_b"))
opp_map = opp_map[opp_map["fighter_id_a"] != opp_map["fighter_id_b"]].drop_duplicates(["fight_id","fighter_id_a"])
opp_map = opp_map.rename(columns={"fighter_id_a":"fighter_id","fighter_id_b":"opponent_id"})[["fight_id","fighter_id","opponent_id"]]
df = df.merge(opp_map, on=["fight_id","fighter_id"], how="left")

# Derive fighter attributes
df["age"] = (df["event_date"] - df["DOB"]).dt.days / 365.25
# UFC age proxy: years since first UFC appearance in our dataset
first_date = df.groupby("fighter_id")["event_date"].transform("min")
df["ufc_age"] = (df["event_date"] - first_date).dt.days / 365.25

# Days since last fight
df = df.sort_values(["fighter_id","event_date","fight_id"])
prev_date = df.groupby("fighter_id")["event_date"].shift(1)
df["days_since_last_fight"] = (df["event_date"] - prev_date).dt.days
df["days_since_last_fight"] = df["days_since_last_fight"].fillna(9999)

# Stance clean
df["STANCE"] = df["STANCE"].fillna("Unknown").astype(str)

# Weightclass, fallback from weightindex lookup if needed
df = df.merge(wclu[["weightindex","weightclass","sex"]].rename(columns={"weightclass":"weightclass_lookup","sex":"sex_lookup"}), 
              on="weightindex", how="left")
df["weightclass"] = df["weightclass"].fillna(df["weightclass_lookup"])
df["weightclass"] = df["weightclass"].fillna("Unknown")

# Compute instant derived stats (no smoothing in v1 yet; use raw as "instant")
minutes = (df["timesec"] / 60.0).replace(0, np.nan)

# helper funcs
def safe_div(a,b):
    return np.where(b==0, 0.0, a/b)

# Landed/attempted naming aligned with dictionary base stats
inst = pd.DataFrame(index=df.index)

# land/att families
land_att = {
    "sig_str": ("sigstracc","sigstratt"),
    "total_str": ("totalacc","totalatt"),
    "head": ("headacc","headatt"),
    "body": ("bodyacc","bodyatt"),
    "leg": ("legacc","legatt"),
    "distance": ("distacc","distatt"),
    "clinch": ("clinchacc","clinchatt"),
    "ground": ("groundacc","groundatt"),
    "takedown": ("tdacc","tdatt"),
}
for base,(lcol,acol) in land_att.items():
    landed = df[lcol].astype(float)
    att = df[acol].astype(float)
    inst[f"{base}_landed"] = landed
    inst[f"{base}_attempted"] = att
    inst[f"{base}_per_min_landed"] = safe_div(landed, minutes)
    inst[f"{base}_per_min_attempted"] = safe_div(att, minutes)
    inst[f"{base}_accuracy"] = safe_div(landed, att)

# shares relative to sig landed
sig_landed = inst["sig_str_landed"].replace(0, np.nan)
for base in ["head","body","leg","distance","clinch","ground"]:
    inst[f"{base}_share_of_sig_landed"] = safe_div(inst[f"{base}_landed"], sig_landed.fillna(0))

# counts
inst["sub_att_count"] = df["subatt"].astype(float)
inst["sub_att_per_min"] = safe_div(inst["sub_att_count"], minutes)

inst["reversals_count"] = df["rev"].astype(float)
inst["reversals_per_min"] = safe_div(inst["reversals_count"], minutes)

inst["knockdowns_count"] = df["kd"].astype(float)
inst["knockdowns_per_min"] = safe_div(inst["knockdowns_count"], minutes)

# control
inst["control_sec_sec"] = df["ctrl"].astype(float)
inst["control_sec_per_min"] = safe_div(inst["control_sec_sec"], minutes)
inst["control_sec_share"] = safe_div(inst["control_sec_sec"], df["timesec"].replace(0,np.nan).fillna(1))

# binaries
inst["win_indicator"] = pd.to_numeric(df["win"], errors="coerce").fillna(0.0).astype(float)
inst["ko_indicator"] = pd.to_numeric(df["ko"], errors="coerce").fillna(0.0).astype(float)
# decision proxy: any decision win (udec+mdec+sdec)
inst["decision_indicator"] = (pd.to_numeric(df["udec"], errors="coerce").fillna(0.0)
                             +pd.to_numeric(df["mdec"], errors="coerce").fillna(0.0)
                             +pd.to_numeric(df["sdec"], errors="coerce").fillna(0.0)).clip(0,1).astype(float)

# attributes instant
inst["age_value"] = df["age"].fillna(df["age"].median())
inst["ufc_age_value"] = df["ufc_age"].fillna(0.0)
inst["days_since_last_fight_value"] = df["days_since_last_fight"].fillna(9999).astype(float)
inst["height_value"] = pd.to_numeric(df["HEIGHT"], errors="coerce").fillna(df["HEIGHT"].median())
inst["reach_value"] = pd.to_numeric(df["REACH"], errors="coerce").fillna(df["REACH"].median())

# stance one hot will be handled after rolling features

# Build list of instant stats needed for rolling / decayed (excluding stance)
instant_cols = inst.columns.tolist()

# Prepare base frame
base_cols = ["fight_id","jevent","EVENT","jbout","BOUT","fighter_id","FIGHTER","opponent_id","event_date","weightclass","weightindex","timesec"]
base = df[base_cols].copy()

print(base.head(), inst.shape)

