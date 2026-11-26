function report=run_native_mpc(materials,outDir)
% Reproduce baseline, weight-only transfer, bounded MPC and independent scenario.
arguments
    materials (1,1) string
    outDir (1,1) string
end
here=fileparts(mfilename('fullpath')); addpath(here);
[ok,a]=fileattrib(materials); assert(ok); materials=string(a.Name);
if ~isfolder(outDir), mkdir(outDir); end
[ok,a]=fileattrib(outDir); assert(ok); outDir=string(a.Name);
source=fullfile(materials,"55-系统辨识MPC-ADRC-PID");
sourceFile=fullfile(source,"LMPC_2nd_order_Kang_2.slx");
saved=load(fullfile(source,"model_param.mat"),"x_model");
p=saved.x_model; assert(isequal(size(p),[1 4]) && all(isfinite(p)));
Ts=.01;
protocol=jsondecode(fileread(fullfile(here,'protocol.json')));
scenarios=[protocol.design protocol.validation protocol.previously_seen_regression protocol.fresh_test];
weights=protocol.weights_Q2_R;
archive=jsondecode(fileread(fullfile(outDir,"archived-fmincon.json")));
old=pwd; scratch=tempname; mkdir(scratch); cd(scratch);
restore=onCleanup(@() cd(old)); %#ok<NASGU>
original='archived_native_case';
assert(~bdIsLoaded(original) && ~bdIsLoaded('native_mpc_corrected'));
copyfile(sourceFile,fullfile(scratch,original+".slx"));
load_system(original);
models=onCleanup(@() closeModels(original,'native_mpc_corrected')); %#ok<NASGU>
observe(original);
matrix=expm([0 1 0 0;p(2) p(1) p(3) 1;zeros(2,4)]*Ts);
report.solver_tests=test_native_mpc(matrix(1:2,1:2),matrix(1:2,3),matrix(1:2,4),Ts);
report.matlab=version; report.source_a_c_b_f0=p; report.sample_time_s=Ts;
report.scenarios=scenarios; report.Q1=1; report.archived_weights_Q2_R=archive.weights_Q2_R;
report.design_weights_Q2_R=weights;
corrected=build_native_copy(sourceFile,outDir,p,Ts,weights(1,1),weights(1,2));
observe(corrected);
for j=1:size(weights,1)
    [entry,~]=rollout(corrected,p,Ts,weights(j,:),protocol.design,true);
    entry.id=j; entry.Q2=weights(j,1); entry.R=weights(j,2); candidates(j)=entry; %#ok<AGROW>
end
ordered=sortrows([[candidates.score]' [candidates.id]'],[1 2]);
shortlist=ordered(1:protocol.shortlist_size,2);
for j=1:numel(shortlist)
    id=shortlist(j);
    [entry,~]=rollout(corrected,p,Ts,weights(id,:),protocol.validation,true);
    entry.id=id; entry.Q2=weights(id,1); entry.R=weights(id,2); validation(j)=entry; %#ok<AGROW>
end
ranked=sortrows([[validation.score]' [validation.id]'],[1 2]); best=ranked(1,2);
selection=struct("protocol_frozen_utc",protocol.frozen_utc,"selected_utc",utcNow(), ...
    "design",protocol.design,"validation",protocol.validation,"candidates",candidates, ...
    "validation_candidates",validation,"shortlist_ids",shortlist,"selected_id",best, ...
    "selected_Q2_R",weights(best,:),"selection_rule",protocol.selection_rule);
writelines(jsonencode(selection,PrettyPrint=true),fullfile(outDir,"selection.json"),Encoding="UTF-8");
report.candidates=candidates; report.validation_candidates=validation;
report.selected_Q2_R=selection.selected_Q2_R; report.selected_utc=selection.selected_utc;
for s=1:numel(scenarios)
    scenario=scenarios(s);
    report.(scenario.label).started_utc=utcNow();
    [m,tr]=rollout(original,p,Ts,weights(1,:),scenario,false);
    report.(scenario.label).original=m;
    writetable(tr,fullfile(outDir,string(scenario.label)+"-original.csv"));
    [m,tr]=rollout(original,p,Ts,archive.weights_Q2_R,scenario,false);
    report.(scenario.label).archive_tuned=m;
    writetable(tr,fullfile(outDir,string(scenario.label)+"-archive-tuned.csv"));
    [m,tr]=rollout(corrected,p,Ts,weights(best,:),scenario,true);
    report.(scenario.label).corrected=m;
    writetable(tr,fullfile(outDir,string(scenario.label)+"-corrected.csv"));
end
mw=get_param(corrected,'ModelWorkspace');
assignin(mw,'Q2',weights(best,1)); assignin(mw,'R1',weights(best,2));
save_system(corrected,fullfile(outDir,corrected+".slx"));
report.score_definition="RMSE(x1-r) + .01*RMS(u)/2 + .01*RMS(du)/(.2)";
report.envelope=struct("abs_u",2,"abs_du_per_s",20,"selected_by","design shortlist then independent validation");
report.historical_exposure=protocol.historical_exposure;
writelines(jsonencode(report,PrettyPrint=true),fullfile(outDir,"report.json"),Encoding="UTF-8");
end

function t=utcNow()
t=char(datetime('now','TimeZone','UTC','Format',"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"));
end

function observe(m)
items={156,1,'reference';189,1,'position';189,2,'velocity';157,1,'action'; ...
    157,2,'increment';202,1,'filtered_reference';170,1,'previous'};
for i=1:size(items,1)
    block=Simulink.ID.getFullName(sprintf('%s:%d',m,items{i,1}));
    h=get_param(block,'PortHandles');
    set_param(h.Outport(items{i,2}),'DataLogging','on','DataLoggingNameMode','Custom', ...
        'DataLoggingName',items{i,3});
end
end

function [metrics,tr]=rollout(m,p,Ts,w,s,isCorrected)
in=Simulink.SimulationInput(m);
fields=struct("a",p(1),"c",p(2),"b",p(3),"f0",p(4),"T_sample",Ts,"Q1",1,"Q2",w(1),"R1",w(2));
for key=string(fieldnames(fields))'
    if isCorrected
        in=in.setVariable(char(key),fields.(key),'Workspace',m);
    else
        in=in.setVariable(char(key),fields.(key));
    end
end
step=Simulink.ID.getFullName(sprintf('%s:156',m));
in=in.setBlockParameter(step,'Time',num2str(s.step_s));
in=in.setBlockParameter(step,'After',sprintf('[%.17g;0]',s.amplitude));
in=in.setModelParameter('StopTime',string(s.stop_s),'SignalLogging','on');
out=sim(in);
t=out.logsout.get('position').Values.Time(:);
assert(abs(max(abs(diff(t)-Ts)))<1e-9);
y=series(out,'position',t); v=series(out,'velocity',t);
r=series(out,'reference',t); r=r(:,1);
rf=series(out,'filtered_reference',t); rf=rf(:,1);
u=series(out,'action',t); du=series(out,'increment',t);
previous=series(out,'previous',t);
assert(max(abs(u-previous-du))<1e-8);
assert(max(abs(previous-[0;u(1:end-1)]))<1e-8);
tr=table(t,r,rf,y,v,previous,u,du);
e=y-r;
metrics=struct("n",numel(t),"rmse",sqrt(mean(e.^2)),"iae",trapz(t,abs(e)), ...
    "itse",trapz(t,t.*e.^2),"rms_u",sqrt(mean(u.^2)),"peak_u",max(abs(u)), ...
    "rms_du",sqrt(mean(du.^2)),"peak_du",max(abs(du)),"total_variation",sum(abs(du)), ...
    "u_violation_samples",sum(abs(u)>2+1e-8),"du_violation_samples",sum(abs(du)>20*Ts+1e-8), ...
    "score",sqrt(mean(e.^2))+.01*sqrt(mean(u.^2))/2+.01*sqrt(mean(du.^2))/(20*Ts));
if isCorrected, assert(metrics.u_violation_samples==0 && metrics.du_violation_samples==0); end
end

function data=series(out,signal,t)
raw=out.logsout.get(signal).Values;
assert(isequal(t,raw.Time(:)),sprintf('Signal clock mismatch: %s',signal));
data=raw.Data;
if size(data,1)~=numel(t)
    data=reshape(data,[],numel(t))';
else
    data=reshape(data,numel(t),[]);
end
assert(all(isfinite(data),'all'));
end

function closeModels(varargin)
for i=1:nargin
    if bdIsLoaded(varargin{i}), close_system(varargin{i},0); end
end
end
