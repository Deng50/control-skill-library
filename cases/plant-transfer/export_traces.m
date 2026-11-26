function export_traces(materials,outdir)
% Replay the original four-coefficient identification plant under known ZOH input.
arguments
    materials (1,1) string
    outdir (1,1) string
end
orig = fullfile(materials,"55-系统辨识MPC-ADRC-PID","Linear_2nd_order_model4Ident.slx");
[ok, src] = fileattrib(orig); assert(ok,"Original SLX not found");
if ~isfolder(outdir), mkdir(outdir); end
[~, target] = fileattrib(outdir);
scratch = tempname;
mkdir(scratch);
previousFolder = pwd;
cd(scratch);
restoreFolder = onCleanup(@() cd(previousFolder));
[~,suffix] = fileparts(scratch);
mdl = matlab.lang.makeValidName("parity_"+suffix);
file = fullfile(scratch,mdl+".slx");
copyfile(src.Name,file);
load_system(file);
cleanup = onCleanup(@() close_system(mdl,0));
sub = mdl+"/2nd_order_dynamic";
add_block("simulink/Sinks/To Workspace",sub+"/velocity log","VariableName","v_ref", ...
    "SaveFormat","Timeseries","Position",[415 180 515 210]);
add_line(sub,"Integrator/1","velocity log/1","autorouting","on");
inblocks = find_system(mdl,"SearchDepth",1,"BlockType","FromWorkspace");
dt = 0.01; t=(0:100)'*dt;
meta = struct([]);
for i=1:3
    q0=[0 .4 -.6]; v0=[0 -.15 .3];
    switch i
        case 1, u = double(t>=0.15 & t<0.45) - double(t>=0.70);
        case 2, u = .7*sin(8*t) + .2*cos(3*t);
        case 3, u = 2*double(mod(floor((0:100)'/8),2)==0)-1;
    end
    input = Simulink.SimulationInput(mdl);
    for k=1:numel(inblocks)
        input=input.setBlockParameter(inblocks{k},"Interpolate","off");
        input=input.setBlockParameter(inblocks{k},"OutputAfterFinalValue","Holding final value");
    end
    input=input.setBlockParameter(sub+"/Integrator","InitialCondition",num2str(v0(i)));
    input=input.setBlockParameter(sub+"/Integrator1","InitialCondition",num2str(q0(i)));
    vals=struct("a",-4.2,"b",9,"c",-9,"f0",1,"T_sample",dt,"time",t, ...
        "u",u,"y",zeros(size(t)),"max_sim_time",1);
    for key=string(fieldnames(vals))'
        input=input.setVariable(key,vals.(key));
    end
    input=input.setModelParameter("StopTime","1","ReturnWorkspaceOutputs","on");
    simout=sim(input);
    pos=simout.out_data; vel=simout.v_ref;
    assert(max(abs(pos.Time(:)-t))<1e-10 && max(abs(vel.Time(:)-t))<1e-10,"Time grids differ");
    data=table(t,u,pos.Data(:),vel.Data(:),'VariableNames',{'t','u','q','v'});
    writetable(data,fullfile(target.Name,sprintf('trace%d.csv',i)));
    meta(i).x0=[q0(i),v0(i)];
    meta(i).samples=height(data);
end
report=struct("model",src.Name,"solver",get_param(mdl,"Solver"),"dt",dt,"MATLAB",version, ...
    "input_hold","zero-order; override From Workspace Interpolate off","velocity_block","2nd_order_dynamic/Integrator", ...
    "position_block","2nd_order_dynamic/Integrator1","trajectories",meta);
writelines(jsonencode(report,PrettyPrint=true),fullfile(target.Name,"native.json"),Encoding="UTF-8");
end
