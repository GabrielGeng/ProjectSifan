clear all; close all; 
load('DatasetAssignBs2.mat');
J = 4;
u = [conv(x(1,:),delay(0,235,1000,1),'same');conv(x(2,:),delay(0,0,1000,1),'same');conv(x(3,:),delay(0,0,1000,1),'same');conv(x(4,:),delay(0,235,1000,1),'same');];

% n = -100:1:100;
% tau0 = 0.235;
% d0 = sinc(n-tau0);
% u = [conv(x(1,:),d0,'same');x(2,:);x(3,:);conv(x(4,:),d0,'same')];

% plot(x(1,:),'b');
% hold on
% plot(u(1,:),'m--');
% hold off
v0 = sum(u,1)/J;
% audiowrite('v0_4sensors.wav',v0,8000);
% soundsc(x(1,:))
% soundsc(v0);

%% Adaptive GSC
B = [-1 1 0 0;
    -1 0 1 0;
    -1 0 0 1;];
%% RLS
if 0
% soundsc(B*u(1:J,:));
% audiowrite('block desired source 4 sensor.wav',B*u(1:J,:),8000);
M = 40;                 % filter length: need to be adjusted
v = [zeros(J-1,M/2) B*u zeros(J-1,M/2)].';   % (40000+M)*3
sz = size(x);                      
w = 1*zeros(M,J-1);     % M*3
lambda =0.999999;
Rx_inv =  0.0001 * eye(M);
rex = zeros(M,1);

n = 40000;
d = v0(M);
% g = diag(v'*w);
% e = sum(g)/(J-1);
% Loss = (d - e)^2;
iter = 1;
W = [];
Y = [];
while iter <= n
    for num_sensor = 1:J-1
    Rx_inv = lambda^(-2)*(Rx_inv - Rx_inv*v(iter:iter+M-1,num_sensor)/(lambda^2+v(iter:iter+M-1,num_sensor)'*Rx_inv*v(iter:iter+M-1,num_sensor))*v(iter:iter+M-1,num_sensor)'*Rx_inv);
    rex = lambda^2*rex+v(iter:iter+M-1,num_sensor)*v0(iter);
    w_col = Rx_inv * rex;
%     y = v0(M+iter-1) - sum(diag((v'*w)))/(J-1);
    w(:,num_sensor) = w_col;
    end
    W(:,:,iter) = w;
%     v = (B*u(1:J,(1+iter):(M+iter))).';
%     Y = [Y y];
    iter = iter + 1;
end
for sensor = 1: J-1
figure(sensor)
for num = 1:M
    plot(squeeze(W(num,sensor,:)));
    hold on
end
hold off
end
for k = 1:n
    y = v0(k) - sum(diag((w'*(v(k:k+M-1,:)))))/(J-1);
    Y = [Y y];
end

% Y = v0 - filter(w,1,B*u(1:J,:));
% soundsc(Y);
% audiowrite('RLS 4sensor.wav',Y,8000);

end

%% FDAF
if 1
M = 64;                 % filter length: need to be adjusted
v = [zeros(J-1,M/2) B*u zeros(J-1,M/2)].';   % (40000+M)*3
sz = size(x);                      
w = 1*zeros(M,J-1);     % M*3
P = zeros(M,1);
n = 40000;
iter = 1;
W = [];
Y = [];
alpha = 0.000015;
beta = 0.99997;
% alpha = 0.000014;
% beta = 0.999975;

while iter <= n
%     for num_sensor = 1:J-1
        X = fft(v(iter:iter+M-1,:));
        W_ifft = ifft(w); %conj(fft(state.w))/state.nw;
        % state.X = dftmtx(state.nw)*state.x;
        P = beta*P + (1-beta)*abs(X).^2/M;
        % state.W = 1/state.nw*conj(dftmtx(state.nw))*state.w;
        W_ifft = W_ifft + 2*alpha/M*P.^(-1).*conj(X)*(v0(iter) - sum(diag(w'*v(iter:iter+M-1,:)))/(J-1));
        % state.w = real(dftmtx(state.nw)*state.W);
        w = fft(W_ifft);
%     end
%     W(:,:,iter) = w;
        iter = iter + 1;
end
% for sensor = 1:J-1
% figure(sensor)
% for num = 1:M
%     plot(squeeze(W(num,sensor,:)));
%     hold on
% end
% hold off
% end
for k = 1:n
    y = v0(k) - sum(diag((w'*(v(k:k+M-1,:)))))/(J-1);
    Y = [Y y];
end
soundsc(real(Y));
% soundsc(v(:,3))
% audiowrite('FDAF 4sensor.wav',real(Y),8000);

end