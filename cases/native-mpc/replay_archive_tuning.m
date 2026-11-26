function replay_archive_tuning(materials,outDir)
% Run the archived top-level script verbatim, with its own SLX revision.
arguments
    materials (1,1) string
    outDir (1,1) string
end
[ok,attr]=fileattrib(materials); assert(ok); materials=string(attr.Name);
if ~isfolder(outDir), mkdir(outDir); end
[ok,attr]=fileattrib(outDir); assert(ok); outDir=string(attr.Name);
scratch=tempname; mkdir(scratch);
old=pwd; cd(scratch);
restoreFolder=onCleanup(@() cd(old)); %#ok<NASGU>
model="LMPC_2nd_order_Kang_2";
script="Run_MPC_For_2nd_order_Plant_auto_tuning";
copyfile(fullfile(materials,model+".slx"),scratch);
copyfile(fullfile(materials,script+".m"),scratch);
copyfile(fullfile(materials,"55-系统辨识MPC-ADRC-PID","model_param.mat"),scratch);
assert(~bdIsLoaded(model),"Close the already loaded model of the same name before replay");
cleanupMdl=onCleanup(@() close_system(model,0)); %#ok<NASGU>
timer=tic;
transcript=evalc("evalin('base',script)");
writelines(transcript,fullfile(outDir,"archived-fmincon-log.txt"),Encoding="UTF-8");
result=struct("weights_Q2_R",evalin('base','x_opt'),"archived_objective", ...
    evalin('base','fval'),"elapsed_s",toc(timer),"model_revision","materials root", ...
    "matlab",version,"initial_weights",[1 1],"tuning_period_assumed_s",0.1);
writelines(jsonencode(result,PrettyPrint=true),fullfile(outDir,"archived-fmincon.json"),Encoding="UTF-8");
end
