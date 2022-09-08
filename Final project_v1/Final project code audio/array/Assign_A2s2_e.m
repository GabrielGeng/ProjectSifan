%% Assignment A2 Scenario 2: Multi-tone beamformers
%
%   Implement tools for multi-tone beamformer design and evaluation
%   Result plots: sensor locations, beampatterns in mesh format
%   Note:   Expand the beamformermer class functions to work for multi-tone
%           signals.
%

clear all;
close all;
clear classes;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s2 a)
% 1. Setup the ULA settings as in figure 3.1 of the assignment

J = 4;                  % Number of sensors
dx = 0.17;                 % meters of element spacing in x-direction
dy = 0.01;                 % meters of element spacing in y-direction
mt_f = 100:100:5000;               % multiple tones in Hz

% Setup an ULA array from the settings and plot the array configuration
my_array = arrays.square_array(J,dx,dy);

figure;
% Use the plot function that belongs to the array class (in @array
% directory)
my_array.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create a beamformer object and put settings in the beamformer object.
b = beamformer;
set(b, 'array',         my_array);
set(b, 'angles',        -180:1:180);
set(b, 'mt_frequency',  mt_f);

% Display all properties of the beamformer b:
b

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s2 b)
% 1. Implement the calc_mt_beampattern.m method that is located in the
%    @beamformer folder.
% 2. Verify the result of the matlab function with your answer at d)

% Remove this return to continu with the assignment
% return;

% Set the multi-tone beamformer weights to 1/J
b.mt_weights = 1/J*ones(J,length(mt_f));
b.calc_mt_beampattern;

% Use the plot function that belongs to the multi-tone beamformer
figure;
b.plot_mt;
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 c)
% 1. Implement the beam_steering_mt.m method that is located in the
%    @beamformer folder.
% 2. Verify the result of the matlab function by visual inspection.

% Remove this return to continu with the assignment
% return;

theta = [30]; % row vector containing angles for which constraints hold
target = [1]; % row vector containing target values for the beampattern
b.beam_steering_mt(theta, target);
b.calc_mt_beampattern;

% Use the plot function that belongs to the narrowband beamformer
figure;
b.plot_mt(theta, {'k-.','LineWidth',2});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assign A2s1 d)
% 1. Add the undesired source direction and make sure that the beamformer
% has unity response at 30 degrees and a zero response at -50 degrees.

% Remove this return to continu with the assignment
% return;

theta = [30 -50]; % row vector containing angles for which constraints hold
target = [1 0]; % row vector containing target values for the beampattern
b.beam_steering_mt(theta, target);
b.calc_mt_beampattern;

% Use the plot function that belongs to the narrowband beamformer
figure;
b.plot_mt(theta, {'k-.','LineWidth',2});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



