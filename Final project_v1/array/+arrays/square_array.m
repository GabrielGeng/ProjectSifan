function sa = square_array(J, dx, dy )
%SQUARE_ARRAY(J, dx, dy) Generate a 2D array with four elements
%   Generate a square array with:
%     - sensor spacing in the x direction of dx
%     - sensor spacing in the y direction of dy
%     - center of the array in the origin

% Calculate the sensor positions in a matrix with all x-axis locations in
% the first column and all y-axis locations in the second column.
px = [-1/2;-1/2; 1/2; 1/2]*dx;
py = [-1/2; 1/2; -1/2; 1/2]*dy;
p = [px, py];

% Generate the square array with sensor positions p
sa = array(p, 'Square array');
end

