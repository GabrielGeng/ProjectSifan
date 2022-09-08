function calc_mt_beampattern(b)
%CALC_MT_BEAMPATTERN(b) Calculate a multitone beampattern
%
% Calculate the beampattern for each frequency in b.mt_frequency
% The matrix b.mt_weights contains the weigth vector per frequency as
% column vectors, i.e.,
%   b.mt_weights(:,f1)
%   b.mt_weights(:,f2)
%
% The output b.mt_beampattern should contain the beampattern per frequency
% as row vectors, i.e.,
%   b.mt_beampattern(f1,:) = ...
%   b.mt_beampattern(f2,:) = ...
%

b.mt_beampattern = [];
for f = 1:length(b.mt_frequency)
    r = b.mt_weights(:,f)'*b.array_response_vector(b.angles, b.mt_frequency(f));
    b.mt_beampattern = [b.mt_beampattern; 1/b.array.number_of_sensors^2*abs(r).^2];
end
end

