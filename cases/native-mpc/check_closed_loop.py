"""Recompute exported metrics and render the independent native SLX traces."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    output = parser.parse_args().out
    summary = json.loads((output/"report.json").read_text("utf-8-sig"))
    protocol=json.loads((Path(__file__).with_name("protocol.json")).read_text("utf-8"))
    chosen=json.loads((output/"selection.json").read_text("utf-8-sig"))
    sequence=("design","validation","independent_step","fresh_test")
    assert len({(s["step_s"],s["amplitude"],s["stop_s"]) for s in summary["scenarios"]})==4
    rank=lambda entries: sorted(entries,key=lambda v:(v["score"],v["id"]))
    short=[c["id"] for c in rank(chosen["candidates"])[:protocol["shortlist_size"]]]
    assert short==chosen["shortlist_ids"]==[c["id"] for c in chosen["validation_candidates"]]
    assert rank(chosen["validation_candidates"])[0]["id"]==chosen["selected_id"]
    assert chosen["selected_Q2_R"]==summary["selected_Q2_R"]==protocol["weights_Q2_R"][chosen["selected_id"]-1]
    parse_time=lambda s: datetime.fromisoformat(s.replace("Z","+00:00"))
    assert parse_time(protocol["frozen_utc"])<parse_time(chosen["selected_utc"])
    for stage in ("independent_step","fresh_test"):
        assert parse_time(chosen["selected_utc"])<parse_time(summary[stage]["started_utc"])
    old=json.loads((output/"previous-two-scenario-report.json").read_text("utf-8-sig")) if (output/"previous-two-scenario-report.json").is_file() else None
    actual = {s:{} for s in sequence}
    columns = [("original", "Original", "#454b55"), ("archive_tuned", "Archive tuned", "#bb622f"),
               ("corrected", "Bounded MPC", "#087e75")]
    for scenario in sequence:
        for key, _, _ in columns:
            data = np.genfromtxt(output/f"{scenario}-{key.replace('_','-')}.csv",names=True,delimiter=",")
            ref = summary[scenario][key]
            np.testing.assert_allclose(np.diff(data["t"]),summary["sample_time_s"],rtol=0,atol=1e-12)
            np.testing.assert_allclose(data["u"]-data["previous"],data["du"],rtol=0,atol=1e-12)
            err = data["r"]-data["y"]
            computed = {"rmse":np.sqrt(np.mean(err**2)), "rms_u":np.sqrt(np.mean(data["u"]**2)),
                        "peak_u":np.max(np.abs(data["u"])),"peak_du":np.max(np.abs(data["du"]))}
            for metric, value in computed.items():
                np.testing.assert_allclose(value,ref[metric],rtol=0,atol=1e-11)
                if old and scenario in ("design","independent_step") and key in ("original","archive_tuned"):
                    np.testing.assert_allclose(value,old[scenario][key][metric],rtol=0,atol=1e-11)
            if key=="corrected":
                assert computed["peak_u"]<=2+1e-10 and computed["peak_du"]<=.2+1e-10
            actual[scenario][key]=data
    plot(actual["independent_step"],columns,output/"independent-step.png",
         "Historical negative-step regression on the original Simulink plant")
    plot(actual["fresh_test"],columns,output/"fresh-test.png",
         "Final positive-step test on the original Simulink plant")
    checks={"native_trajectories_checked":sum(len(v) for v in actual.values()),
            "design_candidate_count":len(chosen["candidates"]),
            "validation_candidate_ids":short,"selected_id":chosen["selected_id"],
            "selected_utc":chosen["selected_utc"],"fresh_test_started_utc":summary["fresh_test"]["started_utc"],
            "fresh_test_started_after_selection":True,"historical_controls_rechecked":old is not None,
            "reported_metrics_match_exported_timeseries":True,"corrected_bounds_pass_all_four_cases":True}
    (output/"acceptance.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    print("Validated design shortlist, independent selection, test chronology, twelve native traces and bounds.")


def plot(actual,columns,path,title):
    fig,axes=plt.subplots(3,1,figsize=(8,5.9),sharex=True,layout="constrained")
    fig.suptitle(title,fontsize=13)
    axes[0].step(actual["original"]["t"],actual["original"]["r"],where="post",
                 color="black",ls=":",lw=1.4,label="Reference")
    for key,label,color in columns:
        data=actual[key]
        for ax,field in zip(axes,("y","u","du")):
            ax.plot(data["t"],data[field],lw=1.4,color=color,label=label)
    for ax,title in zip(axes,("Position","Actuator input","Input increment")):
        ax.set_ylabel(title); ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    for ax,bound in zip(axes[1:],(2,.2)):
        for sign in (-1,1):
            ax.axhline(sign*bound,ls="--",color="#75808c",lw=.8)
    axes[0].legend(ncol=4,fontsize=8,loc="lower left")
    axes[-1].set(xlabel="Time (s)",xlim=(0,actual["original"]["t"][-1]))
    fig.savefig(path,dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()
