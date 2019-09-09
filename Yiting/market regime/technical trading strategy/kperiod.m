

function sh = kperiod(OHLC,x,y,z,scaling)

[STR,K,D] = stoc1(OHLC,x,y,z,20,80,60);

trades=zeros(length(OHLC),1);
for i=1:length(STR(:,1))
    if STR(i,1)~=0
        trades(STR(i,1))=1;
    elseif STR(i,1)==0
        i=i+1;
    end
end
for i=1:length(STR(:,2))
    if STR(i,2)~=0
        trades(STR(i,2))=-1;
    elseif STR(i,2)==0
        i=i+1;
    end
end


kpos=trades;
for i = 2:length(kpos)
    if kpos(i) == 0
        kpos(i) = kpos(i-1);
    end
end
    cash    = cumsum(-trades.*log(OHLC(:,4))); 
    pandl   = kpos.*log(OHLC(:,4)) + cash;
    r = diff(pandl);
    pnl1=r(r~=0);
    sh = scaling*mean(pnl1)/std(pnl1);

