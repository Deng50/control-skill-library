function result = replay_second_order(materials, outputDir)
% Replay the archived SLX using its scoped four-coefficient parameters.
arguments
    materials (1,1) string
    outputDir (1,1) string
end
[found, attr] = fileattrib(fullfile(materials,"55-系统辨识MPC-ADRC-PID"));
assert(found,'Scoped source case was not found under the materials directory');
source = attr.Name;
if ~isfolder(outputDir), mkdir(outputDir); end
[found, attr] = fileattrib(outputDir);
assert(found,'Output directory could not be resolved');
outDir = attr.Name;
saved = load(fullfile(source,"Data4Ident.mat"),"Data_collect");
params = load(fullfile(source,"model_param.mat"),"x_model");
p = requireFour(params.x_model);
mismatchCaught = false;
try
    requireFour([p 0 0]);
catch err
    mismatchCaught = strcmp(err.identifier,'library:ParameterLayout');
end
scratch = tempname;
mkdir(scratch);
oldFolder = pwd;
restore = onCleanup(@() cd(oldFolder));
cd(scratch);
[~, uniqueName] = fileparts(scratch);
model = matlab.lang.makeValidName("replay_" + uniqueName);
copyfile(fullfile(source,"Linear_2nd_order_model4DataProduction.slx"),model+".slx");
closeModel = onCleanup(@() closeIfLoaded(model));
in = Simulink.SimulationInput(char(model));
values = struct("a",-4.2,"c",-9,"b",9,"f0",1,"T_sample",0.01, ...
    "fluctuation_range",1,"base",0,"max_sim_time",20);
for key = string(fieldnames(values))'
    in = in.setVariable(char(key),values.(key));
end
out = sim(in);
t = out.out_data.Time(:);
u = out.input_data.Data(:);
y = out.out_data.Data(:);
assert(isequal(size(saved.Data_collect),[numel(t),3]),'Archive shape mismatch');
maxError = max(abs([t u y]-saved.Data_collect),[],1);
result = struct("evidence","executed","matlab",version,"model","Linear_2nd_order_model4DataProduction.slx", ...
    "samples",numel(t),"sample_time_s",0.01,"max_errors_time_u_y",maxError, ...
    "stored_fit_a_c_b_f0",p,"six_parameter_layout_rejected",mismatchCaught, ...
    "match_tolerance",1e-8,"matched_archive",all(maxError < 1e-8),"work_directory",scratch);
writetable(table(t,u,y),fullfile(outDir,"trajectory.csv"));
writelines(jsonencode(result,PrettyPrint=true),fullfile(outDir,"report.json"),Encoding="UTF-8");
assert(mismatchCaught,'Six-parameter form was silently accepted');
assert(result.matched_archive,'Simulated trace differs from the scoped archive');
end

function p = requireFour(p)
if numel(p) ~= 4 || ~isreal(p) || any(~isfinite(p(:)))
    error('library:ParameterLayout','Expected [a c b f0] with four finite coefficients');
end
p = reshape(p,1,4);
end

function closeIfLoaded(name)
if bdIsLoaded(name), close_system(name,0); end
end
