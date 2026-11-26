"""Recheck field deliverables, original-plant fidelity and policy handoffs from disk."""
from pathlib import Path
import numpy as np
from .io import read, write, digest, runtime

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/"runs"


def run():
    checks={}
    files={}
    def report(path):
        f=RUNS/path
        files[path]=digest(f)
        return read(f)
    source=report("object-plant/source_check.json")
    pair=report("object-plant/paired/report.json")
    validation=report("object-plant/independent.json")
    only=report("object-plant/data-only/report.json")
    checks["object_source_replay"]=source["original_rows"]==2001 and max(source["maximum_archive_error_t_u_y"])<1e-10
    checks["object_same_data_and_fallback"]=(pair["data_sha256"]==only["data_sha256"]==digest(RUNS/"object-plant/trajectories.npz")
        and validation["identical_supervised_weights_without_equation"] and validation["same_split"] and set(only["results"])=={"nn"})
    thresholds=validation["thresholds"]
    for name,r in validation["results"].items():
        checks[f"{name}_changing_input"]=r["changed_u_first_affected_step"]==3 and all(
            trajectory["y_rmse"]<thresholds["arbitrary_y_rmse"] and trajectory["v_rmse"]<thresholds["arbitrary_v_rmse"]
            for trajectory in r["arbitrary_hold_sequences"].values())
    checks["object_derivative_independent"]=validation["results"]["pinn"]["residual"]["max_difference_analytic_vs_finite"]<thresholds["finite_difference_abs"]
    trans=report("plant-transfer/parity.json")
    checks["slx_state_and_delay"]=len(trans["traces"])==3 and all(
        tr["aligned"]["one_step"]["passed"] and tr["aligned"]["free_rollout"]["passed"]
        and tr["delayed"]["free_rollout"]["first_failure"] is not None for tr in trans["traces"].values())
    ctrl=report("native-mpc/report.json")
    audit=report("native-mpc/source-audit.json")
    checks["native_source_pair"]=audit["scoped"]["chart_equals_external_MPC_QP"] and audit["scoped"]["model_sha256"]!=audit["top_level"]["model_sha256"]
    checks["native_qp_validation"]=(ctrl["solver_tests"]["cases"]==40 and ctrl["solver_tests"]["quadprog_plan_max_abs"]<1e-7
        and ctrl["solver_tests"]["nonoptimal"]==0 and ctrl["solver_tests"]["active_first_moves"]>0)
    tr=np.genfromtxt(RUNS/"native-mpc/independent_step-corrected.csv",delimiter=",",names=True)
    m=ctrl["independent_step"]["corrected"]
    checks["native_closed_loop_constraints"]=m["u_violation_samples"]==m["du_violation_samples"]==0 and np.max(np.abs(tr["u"]))<=2+1e-8
    checks["native_deployable"]=(RUNS/"native-mpc/native_mpc_corrected.slx").is_file()
    native_acceptance=report("native-mpc/acceptance.json")
    checks["mpc_frozen_selection"]=native_acceptance["fresh_test_started_after_selection"] and native_acceptance["reported_metrics_match_exported_timeseries"] and native_acceptance["corrected_bounds_pass_all_four_cases"]
    checks["neural_on_native_traces"]=report("object-plant/native_traces.json")["passed"]
    data=report("heat-policy/data.json")
    policy=report("heat-policy/report.json")
    sel=report("heat-policy/selected.json")
    checks["heat_expert_archive"]=data["npz_sha256"]==digest(RUNS/"heat-policy/expert.npz") and sum(
        ep["rows"] for ep in data["episodes"])==11200 and all(max(ep["max_state_replay_abs"])<1e-6 for ep in data["episodes"])
    checks["heat_warmstarts_and_updates"]=all(v["preupdate_max_abs_physical"]==0 and v["max_parameter_change"]>0 and len(v["updates"])==14 for v in policy["pgac"].values())
    probe=policy["gradient_probe"]
    checks["heat_perception_gradient"]=probe["actor_gradient_delta_L2"]>1e-8 and probe["coupled_perception_calls"]==probe["horizon"]==12
    checks["heat_cost_expert_parity"]=max(policy["fault_checks"]["source_stage_vs_torch_terms_max_abs"])<1e-3
    checks["heat_policy_ship"]=sel["sha256"]==digest(RUNS/"heat-policy"/sel["name"]/"inference.ts") and all(
        t["bound_violation_steps"]==0 and t["motor_violation_steps"]==0 for t in policy["test"] if t["policy"]==sel["name"])
    out=RUNS/"field-acceptance.json"
    write(out,dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},source_reports=files,runtime=runtime()))
    if not all(checks.values()):
        raise ValueError([k for k,v in checks.items() if not v])
    print(len(checks),"field checks passed",out)


if __name__=="__main__":
    run()
