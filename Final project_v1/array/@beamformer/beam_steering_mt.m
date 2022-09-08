function beam_steering_mt(b, theta, target)
%BEAM_STEERING_MT(b, theta, target) Calculate constrained mt beamformer
%
% Use beamsteering to obtain a desired beampattern for a multi-tone
% beamformer.
%
% The beamformer has to fullfill the following linear constraints for each
% frequency:
%  w' * A(theta) = target
% where:
%   w are the multi-tone beamformer weights that should be stored in the
%   property b.mt_weights with for each frequency a column.
%   A(theta) is the array response `matrix' containing the array response
%     vector for each theta
%   target is a vector containing the desired beampattern values for theta.

b.mt_weights = [];
for f = 1:length(b.mt_frequency)
    A = b.array_response_vector(theta, b.mt_frequency(f)); % 4*length(theta)
    % exp(1i*2*pi*b.nb_frequency*sin(theta))
    b.mt_weights = [b.mt_weights (target*pinv(A))'];
end
end

