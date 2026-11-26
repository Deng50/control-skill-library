"""Recompute the acceptance decision from the three frozen case outputs and check local asset links."""
import re
from pathlib import Path
import numpy as np
from .io import read, write, digest, runtime
from .lookup import ROOT, lookup


def run():
    checks={}
    def record(name,value):
        checks[name]=bool(value)
    def result(name):
        return read(ROOT/"runs"/name)
    index=read(ROOT/"index.json")
    actual={p.parent.name for p in (ROOT/".agents/skills").glob("*/SKILL.md")}
    record("all_skills_indexed",actual=={r["skill"] for r in index})
    record("all_routing_targets_exist",all((ROOT/r[key]).is_file() for r in index for key in ("instructions","knowledge","case","code","evidence")))
    broken=[]
    docs=[ROOT/"README.md",*(ROOT/"knowledge").glob("*.md"),*(ROOT/"cases").glob("*/README.md"),*(ROOT/".agents/skills").glob("*/SKILL.md")]
    for doc in docs:
        for target in re.findall(r"\]\(([^)\s]+)\)",doc.read_text(encoding="utf-8")):
            destination=(doc.parent/target.partition("#")[0]).resolve()
            if destination==ROOT/"runs/acceptance.json":
                continue  # This invocation creates that report after collecting these checks.
            if not target.startswith(("https://","http://","#")) and not destination.exists():
                broken.append([str(doc.relative_to(ROOT)),target])
    record("markdown_links",not broken)
    queries={"用恒功率温度轨迹训练 PINN":"fit-conditional-thermal-pinn",
             "核验预测矩阵与逐步递推并定位首个出错的预测步":"verify-mpc-prediction",
             "在二阶对象上设计 MPC 并调优权重":"tune-constrained-mpc",
             "用同一对象的模仿权重暖启动 PGAC 并检验感知残差":"warmstart-pgac"}
    routes={q:lookup(q)[0]["skill"] for q in queries}
    record("routing_queries",routes==queries)
    correct=result("prediction/pass.json")
    bad=result("prediction/legacy.json")
    record("prediction_pass",correct["passed"] and correct["max_abs_error"]<=1e-12)
    record("prediction_first_error",not bad["passed"] and (bad["first"]["step"],bad["first"]["channel"])==(3,1)
           and abs(bad["first"]["signed_error"]+.00072)<1e-12)
    record("independent_mimo",result("prediction/independent.json")["status"]=="passed")
    paired=result("thermal/paired/report.json")
    only=result("thermal/data-only/report.json")
    record("thermal_paired_and_no_physics",set(paired["results"])=={"nn","pinn"} and set(only["results"])=={"nn"}
           and paired["split"]==only["split"] and paired["data_sha256"]==digest(ROOT/"runs/thermal/data.csv")
           and paired["results"]["nn"]==only["results"]["nn"])
    record("thermal_reference_regression",abs(paired["results"]["pinn"]["test_rmse_K"]-.155278)<.003
           and abs(paired["results"]["nn"]["test_rmse_K"]-.137497)<.003)
    record("thermal_permutation_leakage_and_reload",result("thermal/independent.json")["status"]=="passed")
    table=result("table/report.json")
    record("tabular_windows",table["rows"]=={"train":153,"validation":34,"test":51} and
           table["heldout_rmse_per_target"][0]<.30)
    mpc=result("mpc/report.json")
    test=mpc["test"]
    record("mpc_three_sets_and_dt",mpc["period_mismatch_rejected"] and len(mpc["trials"])==6 and
           test["selected"]["rmse_m"]<test["baseline"]["rmse_m"] and
           all(m["max_force_violation_N"]<=1e-7 and m["max_position_violation_m"]<=1e-7 and
               m["max_speed_violation_mps"]<=1e-7 for m in [*test.values(),mpc["rediscretized_double_dt"]]))
    experts=result("expert/data.json")
    plan=result("expert/plan.json")
    record("sampling_plan",len(plan["training_pilots"])==len(experts["splits"]["train"]) and
           len(plan["recommendations"])==2 and plan["scenario_sha256"]==experts["scenario_sha256"])
    pgac=result("pgac/report.json")
    record("expert_data_and_imitation",experts["rows"]==432 and experts["samples_near_limit"]>0
           and experts["dataset_sha256"]==digest(ROOT/"runs/expert/expert.npz")
           and result("imitation/report.json")["heldout_action_rmse_N"]<.18)
    imported=result("archive-import/imitation/report.json")
    record("existing_expert_entry",result("archive-import/data.json")["rows"]==432 and imported["heldout_action_rmse_N"]<.18)
    record("pgac_binding",pgac["source_actor_sha256"]==digest(ROOT/"runs/imitation/weights.pt")
           and pgac["controller_sha256"]==experts["controller_sha256"]==digest(ROOT/"runs/mpc/controller.json"))
    record("pgac_warmstart_gradient",all(b["warmstart_max_action_error_N"]==0 and
           all(u["gradient_norm"]>0 for u in b["updates"]) for b in pgac["branches"].values()))
    record("pgac_perception",abs(np.diff(list(pgac["critic_at_same_actor"].values())).item())>1e-9 and
           np.all(np.array(pgac["val_corrected_next_rmse"])<pgac["val_nominal_next_rmse"]))
    record("pgac_heldout",pgac["accepted_on_test"] and set(pgac["test"])=={"expert","imitation","nominal_only","residual_coupled"}
           and result("pgac/independent.json")["status"]=="passed")
    native=result("thermal-critic/report.json")
    record("archive_heat_critic_probe",native["passed"] and native["perception_forward_calls"]==0)
    mat=result("matlab/report.json")
    record("slx_replay_and_dimension",mat["matched_archive"] and mat["six_parameter_layout_rejected"] and mat["samples"]==2001)
    record("greybox_and_linearization",result("identification/report.json")["passed"] and result("linearization/report.json")["accepted"])
    h=result("harness/signatures.json")
    record("harness_signatures",len(h["native"])==9 and all("inputsSchema" in v and "configSchema" in v for v in h["domain"].values()))
    out=ROOT/"runs/acceptance.json"
    write(out,dict(runtime=runtime(),case_regression_limits_only=True,passed=all(checks.values()),checks=checks,
                   route_queries=routes,broken_links=broken,skill_count=len(actual)))
    print(sum(checks.values()),"of",len(checks),"checks passed;",out)
    if not all(checks.values()):
        raise SystemExit(str([k for k,v in checks.items() if not v])+" "+str(broken))


if __name__=="__main__":
    run()
