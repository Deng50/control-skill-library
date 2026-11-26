"""Fingerprint local source artifacts without redistributing source or archived weights."""
import argparse
from pathlib import Path
from .io import digest, write


def build(source, out):
    root = Path(source).resolve()
    paths = set(root.glob("*.py")) | set(root.glob("*.m")) | set(root.glob("*.slx"))
    for folder in [p for p in root.iterdir() if p.is_dir() and p.name not in ("pgac","skills")]:
        paths.update(p for p in folder.rglob("*") if p.suffix.lower() in {".m",".slx",".mat",".p",".txt"})
    paths.update((root/"skills").rglob("*.md"))
    paths.update((root/"pgac").glob("*.py"))
    kernel = root/"pgac/pgac_auto_train_agent_v1"
    paths.update((kernel/"kernels").rglob("*.py"))
    paths.update((kernel/"app").rglob("*.py"))
    spring = root/"pgac/spring-damper-ppt-20261030"
    paths.update((spring/"presentation/reproduce/case-config").glob("*.json"))
    for folder in (spring/"artifacts").iterdir():
        if folder.is_dir():
            paths.update(folder.glob("*.json"))
            paths.update(folder.glob("*.npz"))
    links = {
        "55-系统辨识MPC-ADRC-PID/Linear_2nd_order_model4DataProduction.slx":"runs/matlab/report.json",
        "55-系统辨识MPC-ADRC-PID/Data4Ident.mat":"runs/identification/report.json",
        "55-系统辨识MPC-ADRC-PID/model_param.mat":"runs/matlab/report.json",
        "pgac/thermal-management-model.py":"runs/linearization/report.json",
        "pgac/pgac_auto_train_agent_v1/kernels/pgac_core.py":"runs/thermal-critic/report.json",
        "pgac/pgac_auto_train_agent_v1/kernels/thermal-management-model.py":"runs/thermal-critic/report.json",
        "pgac/spring-damper-ppt-20261030/artifacts/expert_dataset-506f49fac84b/expert.npz":"runs/archive-import/imitation/report.json",
    }
    for name in ("nominal","reference","expert","controller"):
        links["pgac/spring-damper-ppt-20261030/presentation/reproduce/case-config/"+name+".json"]="cases/spring/"+name+".json"
    rows=[]
    for path in sorted(paths):
        key=path.relative_to(root).as_posix()
        item=dict(path=key,sha256=digest(path),bytes=path.stat().st_size,
                  status="historical source or archive; fingerprint only")
        if key in links:
            item.update(status="consumed or compared by an executed case",library_evidence=links[key])
        if key.startswith("遗传算法热模型参数辨识/"):
            item["redistribution"]="third-party notice; inspect original locally, source not included"
        rows.append(item)
    write(out,dict(materials=str(root),scope="selected engineering sources, role instructions, data and archived case artifacts",
                   attribution="自编 Python 算法、MATLAB 回放包装及检查器；弹簧配置取自课题组存档。第三方再分发权需按具体来源确认。",files=rows))
    print(len(rows),"fingerprinted source artifacts")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials",default="../素材")
    p.add_argument("--out",default="sources.json")
    args=p.parse_args()
    build(args.materials,args.out)
