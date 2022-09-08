clear; close all;
theta = -pi:0.01:pi;
r = 2/4.*(cos(3/4*pi*sin(theta))+cos(1/4*pi*sin(theta)));
% B = 1/64.*(cos(255*pi.*sin(theta))+cos(85*pi.*sin(theta))).^2;
B = 1/16*abs(r).^2;
figure
plot(theta*180/pi,10*log(B));
figure
polarplot(theta,B);