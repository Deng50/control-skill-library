function [u,status,plan] = native_mpc_step(Ad,Bd,Ed,x,previous,reference,d,Q,R,Ts)
%#codegen
% State feedback and current, held two-state reference. d is acceleration.
% Returns an absolute actuator command; status 1 = optimal, 0 = hold.
% Preconditions: finite calibrated states, Q PSD, R>0, Ts>0, |previous|<=2.
[free,S]=native_mpc_prediction(Ad,Bd,Ed,x,d);
% Minimize z'*H*z + 2*f'*z over z=[u(k); u(k+1)].
% Every future state is weighted. Each distinct increment is penalized once.
D=[1 0;-1 1];
H=R*(D'*D) + 0.0001*diag([1 9]);
f=-R*D'*[previous;0];
for k=1:10
    rows=2*k-1:2*k;
    G=S(rows,:);
    H=H+G'*Q*G;
    f=f+G'*Q*(free(rows)-reference);
end
% Six half planes impose the absolute actuator bound and both step-rate bounds.
slew=20*Ts;
N=[1 0;-1 0;0 1;0 -1;-1 1;1 -1];
bound=[min(2,previous+slew);-max(-2,previous-slew);2;2;slew;slew];
plan=[previous;previous];
status=0;
best=Inf;
% Strictly convex 2D QP: optimum lies in the interior, on an edge or vertex.
unconstrained=-(H\f);
if all(N*unconstrained<=bound+1e-10)
    plan=unconstrained;
    status=1;
else
    for i=1:6
        normal=N(i,:);
        direction=H\normal';
        trial=unconstrained-direction*((normal*unconstrained-bound(i))/(normal*direction));
        [plan,best,status]=consider(trial,plan,best,status,H,f,N,bound);
        for j=i+1:6
            pair=[normal; N(j,:)];
            if abs(det(pair))>0.5
                trial=pair\[bound(i);bound(j)];
                [plan,best,status]=consider(trial,plan,best,status,H,f,N,bound);
            end
        end
    end
end
u=plan(1);
end

function [plan,best,status]=consider(trial,plan,best,status,H,f,N,bound)
if all(N*trial<=bound+1e-10)
    value=trial'*H*trial+2*f'*trial;
    if value<best
        plan=trial;
        best=value;
        status=1;
    end
end
end
