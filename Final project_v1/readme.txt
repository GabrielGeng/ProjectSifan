/////////////CODES////////////////
1.2：
code position: \array\Assign_A2s1_h.m 
descripition: change my_array = arrays.ULA_x1234(J,dx,dy); in L25 to change the sensor 
array. and theta = [0 135]; in L95
parameters: 
1.2 a)i)J=2; dx=0; dy=0.01; my_array = arrays.ULA_x1x2(J,dx,dy); theta = [0 135]
ii)J=2; dx=0.17; dy=0; my_array = arrays.ULA_x1x4(J,dx,dy);theta = [0 135]
iii)J=4; dx=0.17; dy=0.01; my_array = arrays.ULA_x1234(J,dx,dy);theta = [0 135]
1.2 b)i)J=2; dx=0; dy=0.01; my_array = arrays.ULA_x1x2(J,dx,dy); theta = [0 45]
ii)J=2; dx=0.17; dy=0; my_array = arrays.ULA_x1x4(J,dx,dy);theta = [0 45]
iii)J=4; dx=0.17; dy=0.01; my_array = arrays.ULA_x1234(J,dx,dy);theta = [0 45]

1.3: impul_resp.m

1.4:
a) Scenario2a.m  
run soundsc(v0); in L16
b) Scenario2a.m 
run soundsc(B*u(1:J,:)); in L22
c)Scenario2a.m 
run L46-51(NLMS) L108-113(RLS) to plot the coefficient
d)Copy_of_Scenario2a.m 
L58-74(NLMS) L124-140(RLS) shows the beampattern after tranfer function

1.5:
a) Scenario3.m  
run soundsc(v0); in L17
b) Scenario3.m 
run audiowrite('block desired source 4 sensor_1.wav',acb(1,:),8000); in L25-27 to write 3 
blocked signals in differnet channels
c)Scenario3a.m 
run L62-69(RLS) L129-136(FDAF) to plot the coefficient
d)Copy_of_Scenario3a.m 
L75-91(RLS) L141-157(FDAF) shows the beampattern after tranfer function

//////////WAVE///////////////
v0_2sensors.wav:monaural hearing sound after delay and sum beamformer
v0_4sensors.wav:binaural hearing sound after delay and sum beamformer
block desired source.wav:monaural hearing sound after B
block desired source 4 sensor_1(23).wav：binaural hearing sound ind ifferent channels after B
NLMS 2sensor.wav: monaural hearing results by using NLMS
RLS 2sensor.wav: monaural hearing results by using RLS
RLS 4sensor.wav: binaural hearing results by using RLS
FDAF 4sensor.wav: binaural hearing results by using FDAF
