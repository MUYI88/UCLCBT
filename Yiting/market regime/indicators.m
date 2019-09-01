clear all
close all
format long

%EXTRACT DATA
ri=importdata('ripple.xlsx');
closed=ri.data(:,4);
opend=ri.data(:,1);
highd=ri.data(:,2);
lowd=ri.data(:,3);

%%%DATA(HALF YEAR)%%%
closeh=closed(903:length(closed));
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

%%
%%%%%%%%%%
%%%%%%%%%%%%%%%K-LINE%%%%%%%%%%%%%%%%%%%%
OHLC = [openh highh lowh closeh];
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

figure(8)
subplot(3, 1, [1,2]);
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

closed=ri.data(:,4);
opend=ri.data(:,1);
highd=ri.data(:,2);
lowd=ri.data(:,3);

%%%5-day ma
sp5=closed(898:length(closed));
for i = 1:1:length(sp5)-5
    movingAveManual(i) = mean(sp5(i:i+5));
end
h(1)=plot(timeh,movingAveManual,'linewidth',1);
hold on

%%%10-day ma
sp10=closed(893:length(closed));
for i = 1:1:length(sp10)-10
    movingAveManual2(i) = mean(sp10(i:i+10));
end
h(2)=plot(timeh,movingAveManual2,'linewidth',1);
hold on

%%%30-day ma
sp30=closed(873:length(closed));
for i = 1:1:length(sp30)-30
    movingAveManual3(i) = mean(sp30(i:i+30));
end
h(3)=plot(timeh,movingAveManual3,'linewidth',1);
hold on

%%%60-day ma
sp60=closed(843:length(closed));
for i = 1:1:length(sp60)-60
    movingAveManual4(i) = mean(sp60(i:i+60));
end
h(4)=plot(timeh,movingAveManual4,'linewidth',1);
legend(h([1 2 3 4]), 'MA(5)', 'MA(10)', 'MA(30)', 'MA(60)','location','northwest')


xlim([thh1 thh2])
ylim([0.28,0.48]);

title('K-line Chart for XRP','FontWeight','Bold', 'FontSize', 12)
set(gcf, 'PaperPosition', [-0.75 0.2 26.5 26]);
set(gcf, 'PaperSize', [25 25]);
saveas(gcf, 'k_lines.pdf');

%%
%%%%%%
%%%%%%%%%%VOLUME%%%%%%%%%%
%figure(9)
subplot(3, 1, 3);
volumeup=volumeh;
volumeup(rise,:)=0;%up
%volume=volumeup;
%v1=timetable(Date,volume);
bar(Date,volumeup,'g');
hold on
volumedown=volumeh;
volumedown(down,:)=0;%down
%volume=volumedown;
%v2=timetable(Date,volume);
bar(Date,volumedown,'r');
hold on
volumestay=volumeh;
volumestay(stay,:)=0;%stay
%volume=volumestay;
%v3=timetable(Date,volume);
bar(Date,volumestay,'w');
title('Volume','FontWeight','Bold', 'FontSize', 10)
xlim([thh1 thh2]);
grid on

%%
%%%%%%%%
%%%%%%%%%%%%MACD%%%%%%%%%%%%%%%%%%%
EMA1=movavg(closeh,'exponential',12);
EMA2=movavg(closeh,'exponential',26);
DIFF=EMA1-EMA2;
DEA=movavg(DIFF,'exponential',10);
MACD=2*(DIFF-DEA);

MACD_p = MACD;
MACD_n = MACD;
MACD_s = MACD;
MACD_p(MACD_p<0) = 0;
MACD_n(MACD_n>0) = 0;
MACD_s(MACD_s~=0) = 0;

figure(10)
bar(Date,MACD_p,'g','EdgeColor','w');
hold on

bar(Date,MACD_n,'r','EdgeColor','w');
hold on

bar(Date,MACD_s,'w','EdgeColor','w');
hold on 

h(7)=plot(Date,DIFF,'r','LineWidth',1);
hold on

h(8)=plot(Date,DEA,'b','LineWidth',1);

xlim([thh1 thh2]);
legend(h([7 8]), 'DIF', 'DEA')
grid on
title('MACD', 'FontWeight','Bold', 'FontSize', 12)

%%
%%%%%
%%%%%%%%RSI%%%%%%%%%
closeh=closed(903:length(closed));
openh=opend(903:length(opend));
highh=highd(903:length(highd));
lowh=lowd(903:length(lowd));
volumeh=ri.data(903:end,6);

RSIh1 = calc_RSI(closed(897:length(closed)),6);
RSIh2 = calc_RSI(closed(891:length(closed)),12);
RSIh3 = calc_RSI(closed(879:length(closed)),24);

figure(4)
%subplot(1,2,1)
plot(timeh,RSIh1,timeh,RSIh2,timeh,RSIh3);
hold on
plot([th1,th2],[20,20],'--',[th1,th2],[80,80],'--');
xlim([thh1 thh2]);
legend({'6-day','12-day','24-day','Oversold','Overbought'})
title('Relative Strength Index','FontWeight','Bold', 'FontSize', 12)
hold off
grid on

%%
%%%%
%%%%%%%KDJ%%%%%%%%%
[KValue,DValue,JValue]=KDJ(highh,lowh,closeh,9,3,3,3);
figure(11)
plot(Date,KValue,Date,DValue,Date,JValue);
hold on
plot([th1,th2],[20,20],'--',[th1,th2],[80,80],'--');
legend({'kdjk','kdjd','kdjj','Oversold','Overbought'},'location','southeast')
title('KDJ', 'FontWeight','Bold', 'FontSize', 12)

xlim([thh1 thh2]);
ylim([-10 100]);
grid on

                
%%MA model -- find best parameters
%haven't completed

LCOClose = close;
annualScaling = sqrt(250);
figure(100)
leadlag(LCOClose,1,20,annualScaling)

sharpes = nan(100,1);

for m = 1:100
[~,~,sharpes(m)] = leadlag(LCOClose,1,m);
end

[~,mxInd] = max(sharpes);
figure(101)
leadlag(LCOClose,1,mxInd,annualScaling)

sharpes = nan(100,100);

tic
for n = 1:100
for m = n:100
[~,~,sharpes(n,m)] = leadlag(LCOClose,n,m,annualScaling);
end
end
toc


figure(102)
sweepPlotMA(sharpes)
