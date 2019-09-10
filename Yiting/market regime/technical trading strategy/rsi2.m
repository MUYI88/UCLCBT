function r = rsi2(x,N)
% Copyright 2013 MathWorks, Inc.
L = length(x);
dx = diff([0;x]); 
up=dx;
down=abs(dx);
% up and down moves
I=dx<=0;
up(I) = 0;
down(~I)=0;
% calculate exponential moving averages
m1 = movavg(up,'simple',N); m2 = movavg(down,'simple',N);
%warning off
r = 100*m1./(m1+m2);
%r(isnan(r))=50;
I2=~((up+down)>0);
r(I2)=50;
%warning on