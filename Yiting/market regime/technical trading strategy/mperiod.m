function [sh, pnl1] = mperiod(P,N,M,scaling)

 s = zeros(size(P));
    lead = movavg(P,'simple',N);
    lag = movavg(P,'simple',M);
    s(lead>lag) = 1;
    s(lag>lead) = -1;
    %s(lead>lag*1.0035) = 1;%for ma with threshold
    %s(lag*1.0035>lead) = -1;%for ma with threshold
    
    trades  = [ 0; diff(s(1:end))]; 
    cash    = cumsum(-trades.*log(P)); 
    pandl   = s.*log(P) + cash;
    r = diff(pandl);
    pnl1=r(r~=0);
    sh = scaling*mean(pnl1)/std(pnl1);
    


