"""Extract verifiable SLX source, topology and version differences without loading Simulink."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def prop(node, name):
    return node.findtext(f"P[@Name='{name}']")


def inspect(path, external):
    with ZipFile(path) as archive:
        systems = {}
        for entry in archive.namelist():
            if entry.startswith("simulink/systems/system_") and entry.endswith(".xml"):
                sid = Path(entry).stem.removeprefix("system_")
                systems[sid] = ET.fromstring(archive.read(entry))
        names, signals, params = {}, [], {}

        def visit(sid, ancestors):
            system = systems[sid]
            for b in system.findall("Block"):
                id_ = b.get("SID")
                full = ancestors + "/" + b.get("Name")
                names[id_] = full
                params[id_] = {p.get("Name"): p.text for p in b.findall("P")}
                if id_ in systems:
                    visit(id_, full)
            for line in system.findall("Line"):
                src = prop(line, "Src")
                for dst in line.iterfind(".//P[@Name='Dst']"):
                    signals.append([src, dst.text])

        visit("root", path.stem)
        chart = ET.fromstring(archive.read("simulink/stateflow/chart_56.xml"))
        code = chart.findtext(".//P[@Name='script']")
        assert code
        cfg = ET.fromstring(archive.read("simulink/configSet0.xml"))
        clocks = {p.get("Name"): p.text for p in cfg.iter("P") if p.get("Name") in
                  {"StartTime", "StopTime", "FixedStep", "SolverName", "SignalLogging"}}
        def describe(endpoint):
            sid, port = endpoint.split("#")
            return names[sid] + "#" + port
        focus = {"157", "187", "189", "341", "170", "202"}
        connected = [[describe(src), describe(dst)] for src, dst in signals
                     if (src.split("#")[0] in focus or dst.split("#")[0] in focus)]
        settings = {names[id_]: {k: v for k, v in params[id_].items()
                                 if k in {"Value", "SampleTime", "Before", "Time", "After",
                                          "DelayLength", "Gain", "FunctionName"}}
                    for id_ in ["149","150","151","156","165","166","167","168","170",
                                "216","223","344","345","346","347","352","187::74"]}
        normalize = lambda s: s.replace("\r\n", "\n").strip()
        return {"model_sha256": sha256(path.read_bytes()).hexdigest(),
                "source": str(path), "chart_path": names["187"], "chart_type": prop(chart,"type"),
                "chart_equals_external_MPC_QP": normalize(code) == normalize(external),
                "chart_sha256_normalized": sha256(normalize(code).encode()).hexdigest(),
                "config": clocks, "selected_block_parameters": settings,
                "workspace_exports": [p["VariableName"] for p in params.values() if "VariableName" in p],
                "s_functions": [{"path": names[k], "name": p["FunctionName"]}
                                for k, p in params.items() if "FunctionName" in p],
                "linked_library_references": [p["SourceBlock"] for p in params.values() if "SourceBlock" in p],
                "selected_connections": connected}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path); parser.add_argument("out", type=Path)
    args = parser.parse_args()
    top = args.materials.resolve()
    scoped = top / "55-系统辨识MPC-ADRC-PID"
    m = "LMPC_2nd_order_Kang_2.slx"
    script = top / "MPC_QP.m"
    report = {"scoped": inspect(scoped/m, script.read_text(encoding="utf-8-sig")),
              "top_level": inspect(top/m, script.read_text(encoding="utf-8-sig")),
              "input_files": {str(f): sha256(f.read_bytes()).hexdigest() for f in
                              (script, top/"Run_MPC_For_2nd_order_Plant_auto_tuning.m",
                               scoped/"model_param.mat", scoped/"Run_MPC_For_2nd_order_Plant.m")}}
    assert report["scoped"]["chart_equals_external_MPC_QP"]
    assert report["scoped"]["model_sha256"] != report["top_level"]["model_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
