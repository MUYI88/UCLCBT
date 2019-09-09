%% import data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%PAY ATTENTION%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The column may change for different version of MATLAB
%%% Need to check it when importing
%%% The column in my MATLAB version is:
%%% date-open-high-low-close-volume-maketcap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
format long

%hourly (short-term)
xrp=importdata('df.xlsx');
close11=xrp.data(:,5);
close22=close11;
closedd=log(close22);

%daily (long-term)
xrp=importdata('xrp_20140101_20190630.xlsx');
close11=xrp.data(:,5);
close22=close11(1187:end);
closedd=log(close22);

%% half year (USED FOR TRADING SIGNALS PLOT)(figure 3.4, 3.7 and 3.9)
clear all
close all
format long
ri=importdata('ripple.xlsx');
close11=ri.data(:,5);
opend=ri.data(:,2);
highd=ri.data(:,3);
lowd=ri.data(:,4);

%%%DATA(HALF YEAR)%%%
close22=close11(903:length(close11));
closedd=log(close22);

openh=opend(903:length(opend));
highh=highd(903:length(highd));
lowh=lowd(903:length(lowd));
volumeh=ri.data(903:end,6);

%%%DATE(HALF YEAR)%%%
th1=datetime(2018,12,2);
th2=datetime(2019,6,12);
timeh=th1:th2;
Date=timeh';
%DATE FOR XLIM
thh1=datetime(2018,12,1);
thh2=datetime(2019,6,13);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% buy and hold (for summary table)

r = [0;log(close22(2:end)./close22(1:end-1))];
cr=sum(r) %cumulative return
pd=length(closedd) %position days/hours

stdh=std(r) %standard deviation
prh=length(find(r>0)) %positive return days/hours
sharph=sharpe(r,0) %sharpe ratio

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ma parameter selection (for figure 3.5 and 3.6)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%PAY ATTENTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%need to modify the file of mperiod.m when finding the optimal parameter
%with threshold for short-term strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

annualScaling = sqrt(365*24); %for short-term strategy
%annualScaling = sqrt(365);  %for long-term strategy
sharpes = nan(100,100);

tic
for n = 1:100
    for m = n:100
        sharpes(n,m) = mperiod(close22,n,m,annualScaling);
    end
end
toc
sweepPlotMA(sharpes)
%zlim([-6,4]) %set the range of z-axis %for short-term strategy

[~, bestInd] = max(sharpes(:)); % (Linear) location of max value
[bestM, bestN] = ind2sub(100, bestInd); % Lead and lag at best value

%% MA backtesting (for summary table)
lead = movavg(close22,'simple',23);  %short-term MA %need to set
lag = movavg(close22,'simple',27);  %long-term MA   %need to set
mabuy=zeros(size(closedd));
masell=zeros(size(closedd));

s = zeros(size(closedd));
s(lead>lag*1.0035) = 1; %for threshold
s(lag*1.0035>lead) = -1;%for threshold
%s(lead>lag) = 1;
%s(lag>lead) = -1;
trades  = [ 0; diff(s(1:end))]; 


%i=find(trades~=0);
%I=i(1);buytrades=[];selltrades=[];
%if trades(I)>0
    %buytrades=trades;buytrades(I)=2;buytrades=buytrades./2;
    %selltrades=trades;selltrades(I)=0;selltrades=selltrades./2;
    %masell(1:I)=0;
%elseif trades(I)<0
    %selltrades=trades;selltrades(I)=-2;selltrades=selltrades./2;
    %buytrades=trades;buytrades(I)=0;buytrades=buytrades./2;
    %mabuy(1:I)=0;
%end


mabuy(lead>lag*1.0035) = 1; %for threshold
%mabuy(lead>lag) = 1;
buytrades=[ 0; diff(mabuy(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=mabuy.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rmb=[];
rmb=rb(rb~=0);

masell(lag*1.0035>lead) = -1;
%masell(lag>lead) = -1;
selltrades=[ 0; diff(masell(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=masell.*closedd+cashs;
rs=[0;diff(masell1)];
rms=[];
rms=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(mabuy==1)); %position days
pds=length(find(masell==-1));
stdb=std(rmb); %sd
stds=std(rms);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=sharpe(rmb,0);%sharpe ratio
shs=sharpe(rms,0);

%% plot (for figure 3.4)
OHLC = [openh highh lowh close22];
rise=find(OHLC(:,1)>OHLC(:,4));
data2=OHLC;
data2(rise,:)=0;%up
%timeh2=datenum(timeh');
%timeh2(rise,:)=NaN;
%date=datenum2datetime(timeh2);
open=data2(:,1);
high=data2(:,2);
low=data2(:,3);
Close=data2(:,4);
stock2=timetable(Date,open,high,low,Close);

figure(10)

candle(stock2,'g');
hold on

data3=OHLC;
down=find(OHLC(:,1)<OHLC(:,4));
data3(down,:)=0;%down
%timeh3=datenum(timeh');
%timeh3(down,:)=NaN;
%date=datenum2datetime(timeh3);
open=data3(:,1);
high=data3(:,2);
low=data3(:,3);
Close=data3(:,4);
stock3=timetable(Date,open,high,low,Close);
candle(stock3,'r');
hold on



stay=find(OHLC(:,1)~=OHLC(:,4));
data4=OHLC;
data4(stay,:)=0;%stay
%timeh4=datenum(timeh');
%timeh4(down,:)=NaN;
%date=datenum2datetime(timeh4);
open=data4(:,1);
high=data4(:,2);
low=data4(:,3);
Close=data4(:,4);
stock4=timetable(Date,open,high,low,Close);
candle(stock4,'w');
hold on

lead = movavg(close22,'simple',7);  %short-term MA
lag = movavg(close22,'simple',14);  %long-term MA
v2=movavg(close22,'simple',120);
longp=[];
shortp=[];
l=(lead+lag)/2;
longp(trades>0)=l(trades>0);
longt(trades>0)=timeh(trades>0);
shortp(trades<0)=l(trades<0);
shortt(trades<0)=timeh(trades<0);

h1=plot(timeh,lead,'Linewidth',1.5); hold on; h2=plot(timeh,lag,'Linewidth',1.5);hold on
%h3=plot(timeh,v1,'Linewidth',1.5); hold on; 
h4=plot(timeh,v2,'Linewidth',1.5);
hold on
h5=plot(longt,longp,'g^','Linewidth',2,'MarkerSize',10)
hold on
h6=plot(shortt,shortp,'rv','Linewidth',2,'MarkerSize',10)

legend([h1 h2 h4 h5 h6],'MA(7)','MA(14)','MA(120)','Buy Signal','Sell Signal','Location','Best');

%grid on

ylim([0.28,0.48]);
xlim([thh1 thh2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% rsi parameter selection (for figure 3.8)
annualScaling = sqrt(365);


sharpes = [];

tic
for n = 1:50
    sharpes(n) = rperiod(close22, n,annualScaling);
    
end
toc
figure(5)
plot(1:1:50,sharpes)
xlabel('Period')
ylabel('Sharpe Ratio')

[~, bestInd] = max(sharpes(:));
%% RSI backtesting (for summary table)
M=1; %period of RSI
downThresh=80;
upThresh=20;
x=close22;
S = length(x);
[~,c] = hpfilter(close22,100);
r = rsi2(x-c,M);
n1=length(find(r>50));
n2=length(find(r<50));
%r = rsi2(x-ema(x,15*M),M);
I = r(2:end) <= downThresh & r(1:end-1) > downThresh;
s = zeros(S-1,1);
s(I) = -1;
% Crossing threshold up
I = r(2:end) >= upThresh & r(1:end-1) < upThresh;
s(I) = 1; 
% copy down previous position values
s = [0; s]; % Start from 0 state
for i = 2:S
    if s(i) == 0
        s(i) = s(i-1);
    end
end
s1 = s;
s2 = s;
s1(find(s1==-1))=0;
s2(find(s2==1))=0;
trades=[0;diff(s)];

buytrades  = [ 0; diff(s1(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=s1.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rrb=[];
rrb=rb(rb~=0);

selltrades=[ 0; diff(s2(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=s2.*closedd+cashs;
rs=[0;diff(masell1)];
rrs=[];
rrs=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(s1==1)); %position days
pds=length(find(s2==-1));
stdb=std(rrb); %sd
stds=std(rrs);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=sharpe(rrb,0);%sharpe ratio
shs=sharpe(rrs,0);


%% plot (for figure 3.7)
OHLC = [openh highh lowh close22];
rise=find(OHLC(:,1)>OHLC(:,4));
data2=OHLC;
data2(rise,:)=0;%up
%timeh2=datenum(timeh');
%timeh2(rise,:)=NaN;
%date=datenum2datetime(timeh2);
open=data2(:,1);
high=data2(:,2);
low=data2(:,3);
Close=data2(:,4);
stock2=timetable(Date,open,high,low,Close);

figure(6)
subplot(3,1,1)
candle(stock2,'g');
hold on

data3=OHLC;
down=find(OHLC(:,1)<OHLC(:,4));
data3(down,:)=0;%down
%timeh3=datenum(timeh');
%timeh3(down,:)=NaN;
%date=datenum2datetime(timeh3);
open=data3(:,1);
high=data3(:,2);
low=data3(:,3);
Close=data3(:,4);
stock3=timetable(Date,open,high,low,Close);
candle(stock3,'r');
hold on


stay=find(OHLC(:,1)~=OHLC(:,4));
data4=OHLC;
data4(stay,:)=0;%stay
%timeh4=datenum(timeh');
%timeh4(down,:)=NaN;
%date=datenum2datetime(timeh4);
open=data4(:,1);
high=data4(:,2);
low=data4(:,3);
Close=data4(:,4);
stock4=timetable(Date,open,high,low,Close);
candle(stock4,'w');

ylim([0.28,0.48]);
xlim([thh1 thh2]);

r1 = rsi2(close22,M);
longpr=[];
shortpr=[];
longpr(trades>0)=r(trades>0);
longtr(trades>0)=timeh(trades>0);
shortpr(trades<0)=r(trades<0);
shorttr(trades<0)=timeh(trades<0);
subplot(3,1,[2 3])
h1=plot(timeh,r1,'Linewidth',1);
hold on
h2=plot(timeh,r,'Linewidth',1.5);
hold on
h3=plot(longtr,longpr,'g^','Linewidth',2,'MarkerSize',8)
hold on
h4=plot(shorttr,shortpr,'rv','Linewidth',2,'MarkerSize',8)
hold on
h5=plot([th1,th2],[80,80],'--')
hold on
h6=plot([th1,th2],[20,20],'--')


legend([h1 h2 h3 h4 h5 h6],'RSI(6)','Detrended RSI(6)','Buy Signal','Sell Signal','Overbought','Oversold','Location','Best');
grid on

ylim([0,100]);
xlim([thh1 thh2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% kdj(for summary table)
%OHLC=xrp.data(1187:end,2:5); %daily (for long-term strategy)
OHLC=xrp.data(:,2:5); %hourly (for short-term strategy)
%OHLC = [openh highh lowh close22];%for sample plot


%[STR,K,D] = stoc1(OHLC,12,5,3,20,80,60);%for long-term
[STR,K,D] = stoc1(OHLC,12,2,3,20,80,55);%for short-term
%[STR,K,D] = stoc1(OHLC,10,5,5,20,80,55);%for sample plot

trades=zeros(length(closedd),1);
for i=1:length(STR(:,1))
    if STR(i,1)~=0
        trades(STR(i,1))=1;
    elseif STR(i,1)==0
        trades(STR(i,1))=0;
    end
end
for i=1:length(STR(:,2))
    if STR(i,2)~=0
        trades(STR(i,2))=-1;
    elseif STR(i,2)==0
        i=i+1;
    end
end

trades1=trades;
for i = 2:length(trades)
    if trades(i) == 0
        trades(i) = trades(i-1);
    end
end


s1 = trades;
s2 = trades;
s1(s1==-1)=0;%long position
s2(s2==1)=0;%short position

buytrades  = [ 0; diff(s1(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=s1.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rkb=[];
rkb=rb(rb~=0);

selltrades=[ 0; diff(s2(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=s2.*closedd+cashs;
rs=[0;diff(masell1)];
rks=[];
rks=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(s1==1)); %position days
pds=length(find(s2==-1));
stdb=std(rkb); %sd
stds=std(rks);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=sharpe(rkb,0);%sharpe ratio
shs=sharpe(rks,0);
%% plot (for figure 3.9)
figure(11)
OHLC = [openh highh lowh close22];
rise=find(OHLC(:,1)>OHLC(:,4));
data2=OHLC;
data2(rise,:)=0;%up
%timeh2=datenum(timeh');
%timeh2(rise,:)=NaN;
%date=datenum2datetime(timeh2);
open=data2(:,1);
high=data2(:,2);
low=data2(:,3);
Close=data2(:,4);
stock2=timetable(Date,open,high,low,Close);

figure(11)
subplot(3,1,1)
candle(stock2,'g');
hold on

data3=OHLC;
down=find(OHLC(:,1)<OHLC(:,4));
data3(down,:)=0;%down
%timeh3=datenum(timeh');
%timeh3(down,:)=NaN;
%date=datenum2datetime(timeh3);
open=data3(:,1);
high=data3(:,2);
low=data3(:,3);
Close=data3(:,4);
stock3=timetable(Date,open,high,low,Close);
candle(stock3,'r');
hold on


stay=find(OHLC(:,1)~=OHLC(:,4));
data4=OHLC;
data4(stay,:)=0;%stay
%timeh4=datenum(timeh');
%timeh4(down,:)=NaN;
%date=datenum2datetime(timeh4);
open=data4(:,1);
high=data4(:,2);
low=data4(:,3);
Close=data4(:,4);
stock4=timetable(Date,open,high,low,Close);
candle(stock4,'w');

ylim([0.28,0.48]);
xlim([thh1 thh2]);

longpkk=[];
shortpkk=[];
longpkk(trades1>0)=D(trades1>0);
longtkk(trades1>0)=timeh(trades1>0);
shortpkk(trades1<0)=D(trades1<0);
shorttkk(trades1<0)=timeh(trades1<0);

subplot(3,1,[2 3])
h1=plot(timeh,K);
hold on
h2 = plot(timeh,D,'Linewidth',1.5);
hold on
J=3*K-2*D;
h3 = plot(timeh,J);
hold on
h4=plot(longtkk,longpkk,'g^','Linewidth',2,'MarkerSize',8);
hold on
h5=plot(shorttkk,shortpkk,'rv','Linewidth',2,'MarkerSize',8);
hold on
h7=plot([th1,th2],[20,20],'--');
hold on
h6=plot([th1,th2],[80,80],'--');
hold on
plot([th1,th2],[55,55],'--')

legend([h1 h2 h3 h4 h5 h6 h7],{'K Value (10)','D Value (5)','J Value (5)','Buy Signal','Sell Signal','Overbought','Oversold'},'Location','Best');
grid on

%ylim([0.28,0.48]);
xlim([thh1 thh2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% rsi+ma (for summary table)
lead = movavg(close22,'simple',23); %need to set
lag = movavg(close22,'simple',27);  %need to set
mabuy=zeros(size(closedd));
masell=zeros(size(closedd));

%mabuy(lead>lag) = 1;
%masell(lag>lead) = -1;
mabuy(lead>lag*1.0035) = 1; %for ma with threshold
masell(lag*1.0035>lead) = -1; %for ma with threshold

M=1;
downThresh=80;
upThresh=20;
x=close22;
S = length(x);
[~,c] = hpfilter(close22,100);
r = rsi2(x-c,M);
%r = rsi2(x-ema(x,15*M),M);
I = r(2:end) <= downThresh & r(1:end-1) > downThresh;
s = zeros(S-1,1);
s(I) = -1;
% Crossing threshold up
I = r(2:end) >= upThresh & r(1:end-1) < upThresh;
s(I) = 1; 
% copy down previous position values
s = [0; s]; % Start from 0 state
for i = 2:S
    if s(i) == 0
        s(i) = s(i-1);
    end
end
s1 = s;
s2 = s;
s1(find(s1==-1))=0;
s2(find(s2==1))=0;

pos1 = zeros(S,1);
pos2 = zeros(S,1);
pos1(s1 ==  1 & mabuy ==  1) =  1;
pos2(s1 == -1 & masell == -1) = -1;

buytrades  = [ 0; diff(pos1(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=pos1.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rmrb=[];
rmrb=rb(rb~=0);


selltrades=[ 0; diff(pos2(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=pos2.*closedd+cashs;
rs=[0;diff(masell1)];
rmrs=[];
rmrs=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(pos1==1)); %position days
pds=length(find(pos2==-1));
stdb=std(rmrb); %sd
stds=std(rmrs);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=mean(rmrb)/std(rmrb);%sharpe ratio
shs=mean(rmrs)/std(rmrs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% KDJ+MA (for summary table)
%OHLC=xrp.data(1187:end,2:5); %for long-term
OHLC=xrp.data(:,2:5); %for short-term

%STR = stoc1(OHLC,12,5,3,20,80,60); %for long-term
STR = stoc1(OHLC,12,2,3,20,80,55); %for short-term

trades=zeros(length(closedd),1);
for i=1:length(STR(:,1))
    if STR(i,1)~=0
        trades(STR(i,1))=1;
    elseif STR(i,1)==0
        trades(STR(i,1))=0;
    end
end
for i=1:length(STR(:,2))
    if STR(i,2)~=0
        trades(STR(i,2))=-1;
    elseif STR(i,2)==0
        i=i+1;
    end
end

trades1=trades;
for i = 2:length(trades)
    if trades(i) == 0
        trades(i) = trades(i-1);
    end
end



s1 = trades;
s2 = trades;
s1(s1==-1)=0;%long position
s2(s2==1)=0;%short position

lead = movavg(close22,'simple',23); %need to set
lag = movavg(close22,'simple',27);  %need to set
mabuy=zeros(size(closedd));
masell=zeros(size(closedd));

%mabuy(lead>lag) = 1;
%masell(lag>lead) = -1;
mabuy(lead>lag*1.0035) = 1; %for ma with threshold
masell(lag*1.0035>lead) = -1; %for ma with threshold

pos1 = zeros(S,1);
pos2 = zeros(S,1);
pos1(s1 ==  1 & mabuy ==  1 ) =  1;
pos2(s2 == -1 & masell == -1 ) = -1;

buytrades  = [ 0; diff(pos1(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=pos1.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rmkb=[];
rmkb=rb(rb~=0);

selltrades=[ 0; diff(pos2(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=pos2.*closedd+cashs;
rs=[0;diff(masell1)];
rmks=[];
rmks=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(pos1==1)); %position days
pds=length(find(pos2==-1));
stdb=std(rmkb); %sd
stds=std(rmks);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=sharpe(rmkb,0);%sharpe ratio
shs=sharpe(rmks,0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ma+kdj+rsi (for summary table)
%OHLC=xrp.data(1187:end,2:5); %for long-term
OHLC=xrp.data(:,2:5); %for short-term

%STR = stoc1(OHLC,12,5,3,20,80,60);%for long-term
STR = stoc1(OHLC,12,2,3,20,80,55);%for short-term


trades=zeros(length(closedd),1);
for i=1:length(STR(:,1))
    if STR(i,1)~=0
        trades(STR(i,1))=1;
    elseif STR(i,1)==0
        trades(STR(i,1))=0;
    end
end
for i=1:length(STR(:,2))
    if STR(i,2)~=0
        trades(STR(i,2))=-1;
    elseif STR(i,2)==0
        i=i+1;
    end
end

trades1=trades;
for i = 2:length(trades)
    if trades(i) == 0
        trades(i) = trades(i-1);
    end
end


k1 = trades;
k2 = trades;
k1(k1==-1)=0;%long position
k2(k2==1)=0;%short position

lead = movavg(close22,'simple',23);%need to set
lag = movavg(close22,'simple',27); %need to set
mabuy=zeros(size(closedd));
masell=zeros(size(closedd));

%mabuy(lead>lag) = 1;
%masell(lag>lead) = -1;
mabuy(lead>lag*1.0035) = 1; %for ma with threshold
masell(lag*1.0035>lead) = -1; %for ma with threshold

M=1;
downThresh=80;
upThresh=20;
x=close22;
S = length(x);
[~,c] = hpfilter(close22,100);
r = rsi2(x-c,M);
n1=length(find(r>50));
n2=length(find(r<50));
%r = rsi2(x-ema(x,15*M),M);
I = r(2:end) <= downThresh & r(1:end-1) > downThresh;
s = zeros(S-1,1);
s(I) = -1;
% Crossing threshold up
I = r(2:end) >= upThresh & r(1:end-1) < upThresh;
s(I) = 1; 
% copy down previous position values
s = [0; s]; % Start from 0 state
for i = 2:S
    if s(i) == 0
        s(i) = s(i-1);
    end
end
s1 = s;
s2 = s;
s1(find(s1==-1))=0;
s2(find(s2==1))=0;

pos1 = zeros(S,1);
pos2 = zeros(S,1);
pos1(k1 ==  1 & mabuy ==  1 & s1==1) =  1;
pos2(k2 == -1 & masell == -1 & s2==-1 ) = -1;

buytrades  = [ 0; diff(pos1(1:end))];
cashb=cumsum(-buytrades.*closedd);
mabuy1=pos1.*closedd+cashb;%cumulative return
rb=[0;diff(mabuy1)];
rmrkb=[];
rmrkb=rb(rb~=0);

selltrades=[ 0; diff(pos2(1:end))];
cashs=cumsum(-selltrades.*closedd);
masell1=pos2.*closedd+cashs;
rs=[0;diff(masell1)];
rmrks=[];
rmrks=rs(rs~=0);

mabuy1(end)
masell1(end)

pdb=length(find(pos1==1)); %position days
pds=length(find(pos2==-1));
stdb=std(rmrkb); %sd
stds=std(rmrks);
ntb=length(find(buytrades~=0)); %number of transactions 
nts=length(find(selltrades~=0));
cashb1=cumsum(-buytrades.*close22);
cashs1=cumsum(-selltrades.*close22);
Cb=cashb(end)/ntb; %break-even transation fee
Cs=cashs(end)/nts;
prb=length(find(rb>0));%positive return days
prs=length(find(rs>0));
shb=sharpe(rmrkb,0);%sharpe ratio
shs=sharpe(rmrks,0);

%% boostrapping

%rb=rmb;rs=rms; %for MA
%rb=rrb;rs=rrs; %for RSI
%rb=rkb;rs=rks; %for KDJ
%rb=rmrb;rs=rmrs; %for MA+RSI
%rb=rmkb;rs=rmks; %for MA+KDJ
rb=rmrkb;rs=rmrks; %for MA+KDJ+RSI

r = [0;log(close22(2:end)./close22(1:end-1))];
m=bootstrp(10000,@mean,r);
m1=bootstrp(10000,@mean,rb);
mn=m1-m;
mean1=mean(mn);

CI= prctile(mn,[2.5 97.5])

figure(37)
h1=histogram(mn,800);
hold on 
h11=plot([mean1,mean1],[0,70],'LineWidth',2)
hold on
h12=plot([CI(1),CI(1)],[0,70],'y','LineWidth',2)
hold on
h13=plot([0,0],[0,70],'--','LineWidth',2);
hold on
h14=plot([CI(2),CI(2)],[0,70],'y','LineWidth',2)
xlim([-0.01 0.01]); %for short-term
%xlim([-0.074 0.074]); %for long-term
ylim([0,70]);
legend([h11,h12,h13],'Average Excess Return','95% Confidence Interval','Zero','Location','Best')


figure(38)
m2=bootstrp(10000,@mean,rs);

mn2=m2-m;
mean2=mean(mn2);

CI2= prctile(mn2,[2.5 97.5])

h1=histogram(mn2,800);
hold on 
h11=plot([mean2,mean2],[0,70],'LineWidth',2)
hold on
h12=plot([CI2(1),CI2(1)],[0,70],'y','LineWidth',2)
hold on
h13=plot([0,0],[0,70],'--','LineWidth',2);
hold on
h14=plot([CI2(2),CI2(2)],[0,70],'y','LineWidth',2)

xlim([-0.01 0.01]); %for short-term
%xlim([-0.074 0.074]); %for long-term
ylim([0,70]);        
legend([h11,h12,h13],'Average Excess Return','95% Confidence Interval','Zero','Location','Best')




