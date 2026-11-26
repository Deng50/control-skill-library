function result=test_native_mpc(Ad,Bd,Ed,Ts)
% Test decision columns, previous input and drift using independent propagation.
% Compare the bounded 2D solve with Optimization Toolbox quadprog.
rng(456,"twister");
maxPredictionError=0; maxQpError=0; zeroStatuses=0; active=0;
options=optimoptions('quadprog','Display','off','OptimalityTolerance',1e-11, ...
    'ConstraintTolerance',1e-10,'StepTolerance',1e-12);
for j=1:40
    x=randn(2,1); d=randn; r=randn(2,1); prev=4*rand-2;
    Q=diag([1 .001+.15*rand]); R=.001+.12*rand;
    [free,S]=native_mpc_prediction(Ad,Bd,Ed,x,d);
    for plan=[1 0;0 1;prev randn]'
        predicted=free+S*plan;
        state=x;
        for i=1:10
            state=Ad*state+Bd*plan(min(i,2))+Ed*d;
            maxPredictionError=max(maxPredictionError,norm(predicted(2*i-1:2*i)-state,inf));
        end
    end
    [u,status,plan]=native_mpc_step(Ad,Bd,Ed,x,prev,r,d,Q,R,Ts);
    zeroStatuses=zeroStatuses+(status~=1);
    % Independent finite-difference reconstruction of exact quadratic from rollout cost.
    J=@(z) cost(z,Ad,Bd,Ed,x,prev,r,d,Q,R);
    zero=[0;0]; basis=eye(2); f=zeros(2,1); H=zeros(2);
    for i=1:2
        H(i,i)=J(basis(:,i))+J(-basis(:,i))-2*J(zero);
        f(i)=(J(basis(:,i))-J(-basis(:,i)))/2;
    end
    H(1,2)=J([1;1])-J(basis(:,1))-J(basis(:,2))+J(zero); H(2,1)=H(1,2);
    slew=20*Ts;
    [expected,~,flag]=quadprog(H,f,[-1 1;1 -1],[slew;slew],[],[], ...
        [max(-2,prev-slew);-2],[min(2,prev+slew);2],[],options);
    assert(flag>0); maxQpError=max(maxQpError,norm(plan-expected,inf));
    active=active+(abs(abs(u-prev)-slew)<1e-6 || abs(abs(u)-2)<1e-6);
end
result=struct("cases",40,"basis_and_mixed_prediction_max_abs",maxPredictionError, ...
    "quadprog_plan_max_abs",maxQpError,"nonoptimal",zeroStatuses,"active_first_moves",active);
assert(maxPredictionError<1e-10 && maxQpError<2e-5 && zeroStatuses==0 && active>0);
end

function value=cost(z,A,B,E,x,prev,r,d,Q,R)
value=0;
for k=1:10
    u=z(min(k,2)); x=A*x+B*u+E*d; e=x-r;
    value=value+e'*Q*e+R*(u-prev)^2+.0001*u^2;
    prev=u;
end
end
