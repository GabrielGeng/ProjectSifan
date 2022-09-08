function beam_steering_nb( b, theta, target)
%BEAM_STEERING_NB(b, theta, target) Calculate constrained nb beamformer
%
% Use beamsteering to obtain a desired beampattern. The beamformer has to
% fullfill the following linear constraints:
%  w' * A(theta) = target
% where:
%   w are the narrow band beamformer weights, stored as a column vector in
%   the property b.nb_weights.
%   A(theta) is the array response `matrix' containing the arrar response
%     vector for each theta
%   target is a vector containing the desired beampattern values for theta.

b.nb_weights = [];
A = b.array_response_vector(theta, b.nb_frequency); % 4*length(theta)
% exp(1i*2*pi*b.nb_frequency*sin(theta))
b.nb_weights = (target*pinv(A))';
end
