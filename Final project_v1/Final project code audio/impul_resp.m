clear all; close all;
N = 500;
n = 0:0.05:N;
tau0 = 3;
tau1 = 5.15;
% d0 = sin(pi.*(n-tau0))./pi./(n-tau0);
d0 = sinc(n-tau0);
% d1 = sin(pi.*(n-tau1))./pi./(n-tau1);
d1 = sinc(n-tau1);
figure
subplot(2,1,1);
plot(n,d0,'b'); hold on;
h0 = delay(3,0,10,4);
% d0_hat = conv(sin(pi.*n)./pi./n,h0);
% stem(n(1:end-1),d0_hat(~isnan(d0_hat)));
% stem(circshift(h0,-183));
stem(h0);
hold off;

subplot(2,1,2)
plot(n,d1,'b');
hold on
h1 = delay(5,3,20,6);
stem(h1)
hold off
