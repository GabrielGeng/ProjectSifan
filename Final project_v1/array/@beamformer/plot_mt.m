function plot_mt(this, DOA, linspec_DOA)
%PLOT_WB(this, DOA, linspec_DOA) Plot mt_beampattern via surf

B = 20*log10(this.mt_beampattern);
% Make sure that the frequencies are contineous
if (min(this.mt_frequency) < 0) && (this.mt_frequency(1)==0)
  surf(this.angles, fftshift(this.mt_frequency), fftshift(B,1));
else
  surf(this.angles, this.mt_frequency, B);
end


xlabel('Angle');
ylabel('Frequency');
colorbar;
shading FLAT
axis tight;
%% Check if DOAs are submitted, othwerwise return

if ~exist('DOA','var') || isempty(DOA)
  return
end

%% Check linspec_DOA input
if ~exist('linspec_DOA', 'var') || isempty(linspec_DOA)
  linspec_DOA = {'k-.','LineWidth',2};
elseif isa(linspec_DOA, 'char')
  c{1} = linspec_DOA;
  linspec_DOA = c;
end

%% Plot DOAs as lines with linspec_DOA line options
holdstate = ishold;

hold on;
NDOA = length(DOA);
DOA = reshape(DOA, 1, NDOA);
plot3([DOA; DOA], [min(this.mt_frequency);max(this.mt_frequency)]*ones(1,NDOA),...
  max(B(:))*ones(2, NDOA),linspec_DOA{:});

% Set back hold state
if holdstate
  hold on;
else
  hold off;
end


end % plot_wb

