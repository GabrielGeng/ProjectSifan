%% Assignment A2 Scenario 1: Narrowband beamformers
%
%   Implement tools for narrowband beamformer design and evaluation
%   Result plots: sensor locations, beampatterns in plot format.
%   Note:   Each step uses different functions of the beamformermer class.
%           These functions are denoted with each step.
%

clear all;
close all;
clear classes;
clc;
iter = 0; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 b)
% 1. Setup the ULA settings as in figure 3.1 of the assignment

J = 4;                  % Number of sensors
dx = 0.17;                 % meters of element spacing in x-direction
dy = 0.01;                 % meters of element spacing in y-direction
for nb_f = [500,1000,2000,4000]               % narrowband (nb) frequency in Hz
iter = iter + 1;
c = 340;
% Setup an ULA array from the settings and plot the array configuration
my_array = arrays.ULA_x1234(J,dx,dy);

% Use the plot function that belongs to the array class (in @array
% directory)
% figure;
% my_array.plot();

% Create a beamformer object and put settings in the beamformer object.
b = beamformer;
set(b, 'array',         my_array);
set(b, 'angles',        -180:0.1:180);
set(b, 'nb_frequency',  nb_f);

% Display all properties of the beamformer b:
% b

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 c)
% 1. Implement the array_response_vector.m method that is located in the
%    @beamformer folder.
% 2. Verify the result of the matlab function with your answer at a)

% Remove this return to continu with the assignment
% return
% theta_d = 30;
% A = b.array_response_vector(theta_d, nb_f)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 e)
% 1. Implement the calc_nb_beampattern.m method that is located in the
%    @beamformer folder.
% 2. Verify the result of the matlab function with your answer at d)

% Remove this return to continu with the assignment
% return;

% Set the beamformer weights to 1/J
% b.nb_weights = 1/J*ones(J,1);
% b.calc_nb_beampattern;

% Use the plot function that belongs to the narrowband beamformer
% figure;
% b.plot_nb;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 f)
% 1. Implement the beam_steering_nb.m method that is located in the
%    @beamformer folder.
% 2. Verify the result of the matlab function by visual inspection.

% Remove this return to continu with the assignment
%  return;

% theta = [30]; % row vector containing angles for which constraints hold
% target = [1]; % row vector containing target values for the beampattern
% b.beam_steering_nb(theta, target);
% b.calc_nb_beampattern;

% Use the plot function that belongs to the narrowband beamformer
% figure;
% b.plot_nb([],theta, {'k-.','LineWidth',2});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 g)
% 1. Add the undesired source direction and make sure that the beamformer
% has unity response at 30 degrees and a zero response at -50 degrees.

% Remove this return to continu with the assignment
% return;

theta = [0 135]; % row vector containing angles for which constraints hold
target = [1 0]; % row vector containing target values for the beampattern
b.beam_steering_nb(theta, target);
b.calc_nb_beampattern;

% Use the plot function that belongs to the narrowband beamformer
% figure;
subplot(2,2,iter)
b.plot_nb([],theta, {'k-.','LineWidth',2});title(['f=',num2str(nb_f),'Hz']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%













