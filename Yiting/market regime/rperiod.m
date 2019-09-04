function [sh, pnl1] = rperiod(x, M,scaling)

downThresh = 80;
upThresh = 20;
S = length(x);

% take the RSI of the detrended series
[~,c] = hpfilter(x,100);
r = rsi2(x-c,M);

% Compute RSI Positions
rpos = zeros(S-1,1);
% Crossing threshold down
I = r(2:end) <= downThresh & r(1:end-1) > downThresh; 
rpos(I) = -1;
% Crossing threshold up
I = r(2:end) >= upThresh & r(1:end-1) < upThresh;
rpos(I) = 1; 
% copy down previous position values
rpos = [0; rpos]; % Start from 0 state
for i = 2:S
    if rpos(i) == 0
        rpos(i) = rpos(i-1);
    end
end


trades  = [ 0; diff(rpos(1:end))]; 
    cash    = cumsum(-trades.*log(x)); 
    pandl   = rpos.*log(x) + cash;
    r = diff(pandl);
    pnl1=r(r~=0);
    sh = scaling*mean(pnl1)/std(pnl1);

