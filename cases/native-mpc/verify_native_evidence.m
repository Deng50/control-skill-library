function result=verify_native_evidence(materials,outDir,runHostMex)
% Inspect compiled sample times, capture the original lifted matrix, exercise codegen.
arguments
    materials (1,1) string
    outDir (1,1) string
    runHostMex (1,1) logical = false
end
here=fileparts(mfilename('fullpath')); addpath(here);
[ok,a]=fileattrib(materials); assert(ok); materials=string(a.Name);
[ok,a]=fileattrib(outDir); assert(ok); outDir=string(a.Name);
old=pwd; scratch=tempname; mkdir(scratch); cd(scratch);
restore=onCleanup(@() cd(old)); %#ok<NASGU>
scoped=fullfile(materials,"55-系统辨识MPC-ADRC-PID");
params=load(fullfile(scoped,"model_param.mat")); p=params.x_model;
Ts=.01; Ac=[0 1;p(2) p(1)]; Bc=[0;p(3)]; Ec=[0;1];
N=10; src=fileread(fullfile(materials,"MPC_QP.m"));
% Instrument only the output list and function name, leaving the original body intact.
patched=replace(src,'function controlInput = MPC_QP(', ...
    'function [controlInput,inputMatrix] = capture_archived_matrix(');
assert(~strcmp(src,patched));
writelines(string(patched),fullfile(scratch,"capture_archived_matrix.m"));
[~,candidate]=capture_archived_matrix(N,diag([1 .015]),.1,Ts,Ac,Bc,Ec, ...
    zeros(2,1),zeros(2,1),zeros(2,1),0);
writeJson(outDir,'archived-matrix.json',candidate);
probe=struct("A",eye(2)+Ts*Ac,"B",Ts*Bc,"C",eye(2),"D",[0;0],"E",Ts*Ec, ...
    "x0",[.3;-.1],"previous_u",.12,"delta_u",zeros(N,1),"disturbances",ones(N,1)*.18, ...
    "abs_tol",1e-10);
% JSON needs nested vectors for matrices with one column.
for key=["B","D","E","disturbances"]
    probe.(key)=arrayfun(@(v) {v},probe.(key),'UniformOutput',false);
end
probe.delta_u=arrayfun(@(v) {v},[0;1;zeros(N-2,1)],'UniformOutput',false);
probe.previous_u={probe.previous_u}; writeJson(outDir,'archived-prediction-input.json',probe);
name='native_mpc_clock_probe';
copyfile(fullfile(scoped,"LMPC_2nd_order_Kang_2.slx"),fullfile(scratch,name+".slx"));
load_system(name); cleanupProbe=onCleanup(@() closeIfLoaded(name)); %#ok<NASGU>
mw=get_param(name,'ModelWorkspace');
variables=struct("a",p(1),"c",p(2),"b",p(3),"f0",p(4),"Q1",1,"Q2",.015,"R1",.1,"T_sample",Ts);
for v=string(fieldnames(variables))', assignin(mw,v,variables.(v)); end
result.archived_compiled=compiledClocks(name);
target="native_mpc_corrected"; load_system(fullfile(outDir,target+".slx"));
cleanupTarget=onCleanup(@() closeIfLoaded(target)); %#ok<NASGU>
result.corrected_compiled=compiledClocks(char(target));
% Replay the delivered model without parameter overrides, proving the saved
% model workspace contains the selected weights, calibration and ZOH matrices.
in=Simulink.SimulationInput(char(target));
out=sim(in);
design=readtable(fullfile(outDir,"design-corrected.csv"));
actualU=out.logsout.get('action').Values;
actualY=out.logsout.get('position').Values;
assert(isequal(actualU.Time(:),actualY.Time(:)) && numel(design.t)==numel(actualU.Time));
result.saved_model_replay_max_abs=max(abs([design.t-actualU.Time(:); ...
    design.u-actualU.Data(:);design.y-actualY.Data(:)]));
assert(result.saved_model_replay_max_abs<1e-10);
result.matlab=version;
% Build the same double-precision implementation and compare generated MEX outputs.
result.coder_license=logical(license('test','MATLAB_Coder'));
if result.coder_license
    Z=expm([Ac Bc Ec; zeros(2,4)]*Ts);
    args={Z(1:2,1:2),Z(1:2,3),Z(1:2,4),[.15;-.25],.5,[1;0],p(4),diag([1 .002]),.01,Ts};
    buildDir=fullfile(scratch,'mexbuild');
    % Generate a standalone C interface even if there is no host C compiler.
    lib=coder.config('lib'); lib.GenCodeOnly=true;
    codegen('-config',lib,'-d',fullfile(scratch,'csrc'),'native_mpc_step','-args',args);
    result.codegen_c_source_generated=isfile(fullfile(scratch,'csrc','native_mpc_step.c'));
    assert(result.codegen_c_source_generated);
    installed=mex.getCompilerConfigurations('C','Selected');
    result.selected_host_C_compiler=~isempty(installed);
    result.host_mex_requested=runHostMex;
    if runHostMex
        codegen('-config:mex','-d',buildDir,'native_mpc_step','-args',args);
        result.host_mex_built=true;
        [reference,status,plan]=native_mpc_step(args{:});
        try
            [compiled,mexStatus,mexPlan]=native_mpc_step_mex(args{:});
            result.host_mex_executed=true;
        catch loadError
            result.host_mex_executed=false;
            result.host_mex_diagnostic=loadError.message;
        end
        if result.host_mex_executed
            result.mex_max_abs=max(abs([reference-compiled;plan-mexPlan]));
            result.mex_status=mexStatus;
            if ~isempty(installed), result.compiler=installed.Name; end
            assert(status==1 && mexStatus==1 && result.mex_max_abs<1e-9);
        end
    end
end
writeJson(outDir,'native-verification.json',result);
end

function results=compiledClocks(model)
feval(model,[],[],[],'compile');
term=onCleanup(@() feval(model,[],[],[],'term')); %#ok<NASGU>
ids=[156 157 170 187 189 348 349];
for k=1:numel(ids)
    block=Simulink.ID.getFullName(sprintf('%s:%d',model,ids(k)));
    results(k).path=block; %#ok<AGROW>
    results(k).compiled_sample_time=get_param(block,'CompiledSampleTime');
end
end

function writeJson(folder,name,value)
writelines(jsonencode(value,PrettyPrint=true),fullfile(folder,name),Encoding="UTF-8");
end

function closeIfLoaded(name)
if bdIsLoaded(name), close_system(name,0); end
end
