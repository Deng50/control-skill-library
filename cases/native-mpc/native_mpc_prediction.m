function [free, influence] = native_mpc_prediction(Ad,Bd,Ed,x,d)
%#codegen
% Ten future states. Decisions are u(k),u(k+1), with the latter held.
free=zeros(20,1);
influence=zeros(20,2);
offset=x;
gain=zeros(2,2);
for k=1:10
    offset=Ad*offset + Ed*d;
    gain=Ad*gain;
    col=min(k,2);
    gain(:,col)=gain(:,col) + Bd;
    rows=2*k-1:2*k;
    free(rows)=offset;
    influence(rows,:)=gain;
end
end
