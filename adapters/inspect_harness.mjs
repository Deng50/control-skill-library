// Snapshot local source declarations and the installed read-only catalog.
import {readFile, readdir, writeFile, mkdir} from "node:fs/promises";
import {resolve, join, dirname} from "node:path";
import {pathToFileURL} from "node:url";
import {createHash} from "node:crypto";

const [plugin="../control-agent-harness/harness-plugin/src", legacy="../素材/skills",
       output="runs/harness/signatures.json"] = process.argv.slice(2);
async function markdown(folder) {
  const docs=[];
  for (const entry of await readdir(folder,{withFileTypes:true})) {
    const path=join(folder,entry.name);
    if(entry.isDirectory()) docs.push(...await markdown(path));
    else if(entry.name.endsWith(".md")) docs.push(await readFile(path,"utf8"));
  }
  return docs;
}
const old=(await markdown(legacy)).join("\n");
const sources={};
for (const file of ["index","simulink-build","simulink-plant","simulink-wire","plant-replay","plant-workspace"])
  sources[file+".mjs"]=await readFile(join(plugin,file+".mjs"),"utf8");
const sha=text=>createHash("sha256").update(text).digest("hex");
const native={};
for (const id of new Set(old.match(/\bcontrol_[a-z_]+\b/g))) {
  const pattern=id==="control_recheck_plant_workspace"?/name:\s*recheck\b/:new RegExp("name:\\s*['\"]"+id+"['\"]");
  for (const [file,src] of Object.entries(sources)) {
    const match=pattern.exec(src);
    if (!match) continue;
    const tail=src.slice(match.index);
    const start=/\bparameters\s*:\s*\{/.exec(tail);
    if (!start) throw Error(id+" parameters absent");
    const end=/\}\s*,\s*(?:output[\s,:]|execute\s*[:(])/.exec(tail.slice(start.index));
    if (!end) throw Error(id+" output boundary absent");
    native[id]={file,line:src.slice(0,match.index).split("\n").length,
      source_sha256:sha(src),parameters_source:tail.slice(start.index,start.index+end.index+1)};
    break;
  }
  if (!native[id]) throw Error("Missing registration "+id);
}
const {operations,operationHelp}=await import(pathToFileURL(resolve(plugin,"catalog.mjs")).href);
const domain={};
for (const name of Object.keys(operations)) {
  if(!new RegExp("\\b"+name+"\\b").test(old)) continue;
  const h=operationHelp({operation:name});
  domain[name]={agents:h.agents,inputsSchema:h.inputsSchema,configSchema:h.configSchema,
                artifactInputs:h.artifactInputs,example:h.request};
}
await mkdir(dirname(output),{recursive:true});
await writeFile(output,JSON.stringify({scope:"source declarations and operationHelp; runtime jobs not invoked",
  plugin:resolve(plugin),catalog_sha256:sha(await readFile(join(plugin,"catalog.mjs"),"utf8")),native,domain},null,2)+"\n");
console.log(Object.keys(native).length,"native signatures and",Object.keys(domain).length,"operation schemas");
