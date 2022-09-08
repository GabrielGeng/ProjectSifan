clear all; close all; 
load('DatasetAssignBs2.mat');
J = 2;
u = [conv(x(1,:),delay(0,235,1000,1),'same');conv(x(J,:),delay(0,0,1000,1),'same')];
% n = -100:1:100;
% tau0 = 0.235;
% d0 = sinc(n-tau0);
% u = [conv(x(1,:),d0,'same');x(J,:)];

% plot(x(1,:),'b');
% hold on
% plot(u(1,:),'m--');
% hold off
v0 = sum(u,1)/J;
% soundsc(x(1,:))
% soundsc(v0);
% audiowrite('v0_2sensors.wav',v0,8000);
% return
%% Adaptive GSC
B = [-1 1];
% (N)LMS
if 0
% soundsc(B*u(1:J,:));
% audiowrite('block desired source.wav',B*u(1:J,:),8000);
M = 10;                 % filter length: need to be adjusted
v = [zeros(J-1,M/2) B*u zeros(J-1,M/2)].';
sz = size(x);                      
w = 1*zeros(M,J-1);
alpha = 0.0001;
epsilon = 0.001;
n = 40000;
% d = v0(M);
% g = diag(v'*w);
% e = sum(g)/(J-1);
% Loss = (d - e)^2;
iter = 1;
W = [];
Y = [];
while iter <= n
    variance = epsilon +v(iter:iter+M-1)'*v(iter:iter+M-1)/M;
    w = w + 2*alpha*v(iter:iter+M-1)*conj(v0(iter) - sum(diag((w'*v(iter:iter+M-1))))/(J-1));
%     Loss = (v0(M+iter-1) - sum(diag((v'*w)))/(J-1))^2;
%     v = (B*u(1:J,(1+iter):(M+iter))).';
    W = [W w];
    iter = iter + 1;
end
figure
for num = 1:length(W(:,1))
    plot(W(num,:));
    hold on
end
hold off
for k = 1:n
    y = v0(k) - sum(w'*v(k:k+M-1))/(J-1);
    Y = [Y y];
end
soundsc(Y);
% audiowrite('NLMS 2sensor.wav',Y,8000);
end

%% RLS
if 1
% soundsc(B*u(1:J,:));
M = 8;                 % filter length: need to be adjusted
% v = (B*u(1:J,1:M)).';
v = [zeros(J-1,M/2) B*u zeros(J-1,M/2)].';
sz = size(x);                      
w = 1*zeros(M,J-1);
lambda = 0.999999;
Rx_inv = 1 * eye(M);
rex = zeros(M,1);

n = 40000;
% d = v0(M);
% g = diag(v'*w);
% e = sum(g)/(J-1);
% Loss = (d - e)^2;
iter = 1;
W = [];
Y = [];
while iter <= n
    Rx_inv = lambda^(-2)*(Rx_inv - Rx_inv*v(iter:iter+M-1)/(lambda^2+v(iter:iter+M-1)'*Rx_inv*v(iter:iter+M-1))*v(iter:iter+M-1)'*Rx_inv);
    rex = lambda^2*rex+v(iter:iter+M-1)*v0(iter);
    w = Rx_inv * rex;
%     y = v0(M+iter-1) - sum(diag((v'*w)))/(J-1);
%     v = (B*u(1:J,(1+iter):(M+iter))).';
    W = [W w];
%     Y = [Y y];
    iter = iter + 1;
end
figure
for num = 1:length(W(:,1))
    plot(W(num,:));
    hold on
end
hold off
for k = 1:n
    y = v0(k) - sum(w'*v(k:k+M-1))/(J-1);
    Y = [Y y];
end
% for k = 1:M-1
%     Y = [v0(k) - sum(diag(B*u(1:J,1:k))).''*w(1:k) Y];
% end

% Y = v0 - filter(w,1,B*u(1:J,:));
soundsc(Y);
% audiowrite('RLS 2sensor_3.wav',Y,8000);
end