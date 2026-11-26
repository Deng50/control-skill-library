function model=build_native_copy(sourceFile,outDir,p,Ts,q2,r)
% Reproducible installation of the bounded controller into a copy of the SLX.
% SID resolution enforces the expected archived revision and wiring.
model='native_mpc_corrected';
target=fullfile(outDir,model+".slx");
scratch=tempname; mkdir(scratch);
workFile=fullfile(scratch,model+".slx");
assert(~bdIsLoaded(model),"Close the deployed model before rebuilding it");
copyfile(sourceFile,workFile);
load_system(workFile);
sid=@(id) Simulink.ID.getFullName(sprintf('%s:%d',model,id));
ctl=sid(157); block=sid(187);
chart=sfroot().find('-isa','Stateflow.EMChart','Path',block);
assert(isscalar(chart) && contains(chart.Script,'inputMatrix(start:stop, columns)'));
port=get_param(block,'PortHandles');
line=get_param(port.Inport(9),'Line');
assert(isequal(get_param(line,'SrcBlockHandle'),get_param(sid(186),'Handle')));
delete_line(line);
prior=get_param(sid(170),'PortHandles');
add_line(ctl,prior.Outport(1),port.Inport(9),'autorouting','on');
terminal=add_block('simulink/Sinks/Terminator',[ctl '/Unused state differences']);
oldMux=get_param(sid(186),'PortHandles');
terminalPort=get_param(terminal,'PortHandles');
add_line(ctl,oldMux.Outport(1),terminalPort.Inport(1),'autorouting','on');
chart.Script=strjoin([
    "function controlInput = MPC_bounded(predictionHorizon,stateWeight,inputWeight,samplingTime,Ad,Bd,Ed,currentState,previousApplied,referenceState,disturbance)"
    "%#codegen"
    "% Input 9 carries the previous applied command. Ports 5 to 7 carry ZOH matrices."
    "assert(predictionHorizon==10);"
    "[u,status] = native_mpc_step(Ad,Bd,Ed,currentState,previousApplied,referenceState,disturbance,stateWeight,inputWeight,samplingTime);"
    "assert(status==1);"
    "controlInput=u-previousApplied;"
    "end"],newline);
% The source's plants and reference filter are preserved. Force ZOH at controller
% output via a discrete chart update; the plant's integrators remain continuous.
chart.ChartUpdate='DISCRETE'; chart.SampleTime='T_sample';
set_param(sid(166),'Value','A_native_d');
set_param(sid(167),'Value','B_native_d');
set_param(sid(168),'Value','E_native_d');
set_param(sid(352),'Value','f0');
matrix=expm([0 1 0 0;p(2) p(1) p(3) 1;zeros(2,4)]*Ts);
mw=get_param(model,'ModelWorkspace');
fields=struct("a",p(1),"c",p(2),"b",p(3),"f0",p(4),"T_sample",Ts, ...
    "A_native_d",matrix(1:2,1:2),"B_native_d",matrix(1:2,3), ...
    "E_native_d",matrix(1:2,4),"Q1",1,"Q2",q2,"R1",r);
for key=string(fieldnames(fields))'
    assignin(mw,key,fields.(key));
end
save_system(model,target);
end
