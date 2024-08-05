import pandas as pd
import matplotlib.pyplot as plt
import os

alerts_path = "./ztfsummerschool_alert_data/ztf_alerts_2460474.5.parquet"

if not os.path.isfile(alerts_path):
    print(f"ERROR: File {alerts_path} not found.")

df_alerts = pd.read_parquet(alerts_path)
print(f"Loaded {len(df_alerts)} alerts.")

print (f"This ZTF data contains {len(df_alerts)} alerts, which {len(df_alerts['objectId'].unique())} are unique objects.")
# objectIds are attributed to detections with a spatial proximity
# detections are new positons create new objectIds, and subsequent detections at
# the same position are associated with the same objectId
# this is how we keep track of the same object over time, to build lightcurves
print(f"5 random objects: {df_alerts['objectId'].sample(5).values}")

# alerts also come with a unique alert identifier, the candid
# the candid is unique for each alert, and is used to identify the alert
# in the ZTF database
print(f"5 random candid: {df_alerts['candidate.candid'].sample(5).values}")

lightcurve_path = "./ztfsummerschool_alert_data/ztf_alerts_2460474.5_prv_candidates.parquet"

if not os.path.isfile(lightcurve_path):
    print(f"ERROR: No file found at {lightcurve_path}. Make sure to download the data first!")

df_lightcurve = pd.read_parquet(lightcurve_path)

# the objectId was set as an index here, but we can reset it to have it as a column which
# will be easier to work with
df_lightcurve.reset_index(inplace=True)

print(df_lightcurve.head())

# we can look for a specific object's lightcurve using the objectId column from the alerts, that are used as the index in the lightcurve data
print(df_lightcurve[df_lightcurve['objectId'] == 'ZTF24aarwzgs'])

def get_lightcurve(alert):
    prv_canidates = df_lightcurve[df_lightcurve['objectId'] == alert['objectId']]
    prv_canidates = prv_canidates[prv_canidates["jd"] < alert["candidate.jd"]]
    # before concatenating the alert, we need to format it the same way as the prv_candidates
    # that is, remove the "candidate." prefix from the column names
    alert_copy = alert.copy()
    alert_copy = {
        key.replace("candidate.", ""): value
        for key, value in alert_copy.items()
    }
    # remove the classificaitons. and coordinates. fields
    alert_copy = {key: value for key, value in alert_copy.items() if not key.startswith("classifications.") and not key.startswith("coordinates.")}
    prv_canidates = pd.concat([prv_canidates, pd.DataFrame([alert_copy])])
    return prv_canidates

# let's test the function
alert = df_alerts[df_alerts["objectId"] == "ZTF24aarwzgs"].iloc[0]
lc = get_lightcurve(alert)
print(lc.head())

fid2color = {1: "green", 2: "red", 3: "orange"}
fid2filter = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

def plot_lightcurve(lc, nondet=True):
    # add a "xaxis" column to the lightcurve = jd - jd.min()
    lc["xaxis"] = lc["jd"] - lc["jd"].min() if nondet else lc["jd"] - lc[lc["magpsf"] > 0]["jd"].min()
    plt.figure(figsize=(10, 6))
    for fid, color in fid2color.items():
        lc_fid = lc[lc["fid"] == fid]
        if len(lc_fid) == 0:
            continue
        # then we split detections and non-detections (no magpsf)
        mask_det = lc_fid["magpsf"] > 0
        lc_det = lc_fid[mask_det]
        lc_nondet = lc_fid[~mask_det]
        if len(lc_det) > 0:
            plt.errorbar(lc_det["xaxis"], lc_det["magpsf"], yerr=lc_det["sigmapsf"], fmt='o', color=color, label=fid2filter[fid])
        if len(lc_nondet) > 0 and nondet:
            plt.scatter(lc_nondet["xaxis"], lc_nondet["diffmaglim"], color=color, marker='v', alpha=0.5)
    plt.gca().invert_yaxis() # lower magnitude is brighter (reverse the y-axis)
    plt.xlabel("Time (JD)")
    plt.ylabel("Magnitude")
    plt.title(f"Lightcurve for {lc.iloc[0]['objectId']} at JD {lc.iloc[0]['jd']}")

# let's test it here:
lc = get_lightcurve( df_alerts[df_alerts["objectId"] == "ZTF24aarwzgs"].iloc[0])
print(plot_lightcurve(lc))