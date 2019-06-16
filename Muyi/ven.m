t=datestr(719529+time/86400,'dd-mmm-yyyy HH:MM:SS') %更改时间

%% logreturns and returnsssss
returns      = diff(close)./close(1:end-1,:);
returns_12    =(close(13:end)-close(1:end-12))./close(1:end-12,:);
returns_24   =(close(25:end)-close(1:end-24))./close(1:end-24,:);
returns_168   =(close(169:end)-close(1:end-168))./close(1:end-168,:);

r = log(close(2:end)./close(1:end-1)); %log return
lreturns_12    =log(close(13:end))-log(close(1:end-12));
lreturns_24   =log(close(25:end))-log(close(1:end-24));
lreturns_168   =log(close(169:end))-log(close(1:end-168));


%% Aggregation of returns %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tau = 1*24; % Number of hours for aggregation of returns 

flag = 0; % Flag variable (set to 1 to aggregate returns over longer time scales)

if flag == 1
    
    aux = [];
   
    for t = 0:tau:length(r)-tau
       
        aux = [aux; sum(r(t+1:t+tau))];
        
    end
    
    r = aux;
    
end

%% plot the time series
figure,
plot(time,close),
xlabel('Time','fontsize',14)
ylabel('price','fontsize',14)
set(gca,'fontsize',14)
title('VEN hourly price vs time ','fontsize',14)
print('Dpriceswithtime.eps','-depsc')

figure, %plot log return
lreturns=diff(log(close));
plot(lreturns)
xlabel('Time','fontsize',14)
ylabel('log-return','fontsize',14)
set(gca,'fontsize',14)
title('VEN Hourly Log Return','fontsize',14)

% Plot Volumes
figure,
plot(time,total_volume_en),
title(' Total Volumes vs. Time')
xlabel('Time','fontsize',14)
ylabel('Total Volume','fontsize',14)
set(gca,'fontsize',14)
print('volumes.eps','-depsc')

figure,
plot(time(1:24),lreturns_24),
title('Ripple 1-day log-returns vs. Time')
xlabel('Dates','fontsize',14)
ylabel('log-return','fontsize',14)
set(gca,'fontsize',14)
print('dlog_returns1.eps','-depsc')


% Compute and print the values of the first four moments

N = length(r);
m= sum(r)/N; % Compute mean and store value in variable
s = sqrt(sum((r-m).^2)/N); % Compute std. deviation and store value in variable
fprintf('Std. deviation = %4.3f\n',s)
fprintf('Skewness = %4.3f\n',sum((r-m).^3)/(N*s^3))
fprintf('Excess kurtosis = %4.3f\n',sum((r-m).^4)/(N*s^4)-3)

%%% Plot of empirical PDF vs Gaussian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(min(r),max(r),100); % Point grid between min and max return
g = exp(-(x-m).^2/(2*s^2))/sqrt(2*pi*s^2); % Gaussian PDF on point grid

NB = 20;

figure(1)
subplot(1,2,1)
[b,a] = histnorm(r,NB); % Normalized histogram of returns with NB bins
semilogy(a,b,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
semilogy(x,g,'r','LineWidth',2)
xlim([-0.4 0.4])
ylim([0.03 10])
set(gca,'FontSize',20)
title('PDF')


%%% Plot of empirical CCDF vs Gaussian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = sort(r); % Returns sorted in ascending order
y = 1:1:length(r); 
y = 1 - y/(length(r)+1); % Calculating CCDF
 
c = 0.5*(1 - erf((x-m)/(s*sqrt(2)))); % Gaussian CCDF

subplot(1,2,2)
semilogy(x,y,'b','LineWidth',2)
hold on
semilogy(x,c,'r','LineWidth',2)
ylim([1e-4 1])
set(gca,'FontSize',20)
title('CCDF')

%% log return relative frequency distribution 
%compute and plot return distribution
[freq,bin]=hist(lreturns,100);
figure
bar(bin,freq/sum(freq))
hold on
%compare with normal distribution
m=mean(lreturns);
s=std(lreturns);
x=[min(lreturns):(max(lreturns)-min(lreturns))/1000:max(lreturns)];
plot(x,normpdf(x,m,s)*(bin(2)-bin(1)),'-m','linewidth',2)
xlim([-0.5,0.5])
ylim([0,max(freq/sum(freq))*1.1])
legend('log-return','normal')
title('VEN Log-Return relative frequency distribution','fontsize',14)
xlabel('Log-return','fontsize',14)
ylabel('relative frequency','fontsize',14)
set(gca,'fontsize',14)

%% complementary cumulative log-return distribution
figure
loglog(sort(lreturns(lreturns>0)),1-[1:(length(lreturns(lreturns>0)))]/length(lreturns(lreturns>0)),'+b')
hold on
loglog(sort(-lreturns(lreturns<0)),1-[1:(length(lreturns(lreturns<0)))]/length(lreturns(lreturns<0)),'xr')
x=[max(lreturns)/1000:max(lreturns)/1000:max(lreturns)];
loglog(x,1-(normcdf(x,m,s)-0.5)*2,'-m','linewidth',2)
xlim([1e-5,4])
ylim([1e-6,1.5])
legend({'pos ret','neg ret','normal'})
title('VEN Complementary Cumulative Log-Return distribution','fontsize',14)
xlabel('Log-Return','fontsize',14)
ylabel('Complementary Cumulative Distribution','fontsize',14)
set(gca,'fontsize',14)

%%qq plot(hourly log-returns)
figure,
qqplot(lreturns)
title('VEN q-q plot Log-Return Distribution','fontsize',14)

%%autocorrelation
figure, % 1 hour
[a1,lags1] = autocorr(lreturns,250);
plot(lags1,a1,'-r')
hold on
[a2,lags2] = autocorr(abs(lreturns),250);
plot(lags2,a2,'-m')
[a3,lags3] = autocorr(lreturns.^2,250);
plot(lags3,a3,'-b')
plot([0 300],[0 0],'-k')
axis([0 250 -0.2 1])
xlabel('lags (hours)','fontsize',14)
ylabel('autocorrelation','fontsize',14)
title('VEN autocorrelation of 1 hour log-returns','fontsize',14)
set(gca,'fontsize',14)
legend({'log-returns','|log-returns|','log-returns^2'})


%% empirical var and cvar
ys = sort(lreturns);
T = length(ys);

display('------------------------------')	
display('Ripple VaR Hourly')
VaR_Ripple95 = -ys(int16(ceil(0.05*T)));
VaR_Ripple99 = -ys(int16(ceil(0.01*T)));
display(sprintf('VaR Ripple 95: %.9f', VaR_Ripple95))
display(sprintf('VaR Ripple 99: %.9f', VaR_Ripple99))
display('------------------------------')

display('------------------------------')
display('Ripple CVaR Hourly')
CVaR_Ripple95 = -mean(ys(1:ceil(0.05*T)));
CVaR_Ripple99 = -mean(ys(1:ceil(0.01*T)));
display(sprintf('CVaR Ripple 95: %.9f', CVaR_Ripple95))
display(sprintf('CVaR Ripple 99: %.9f', CVaR_Ripple99))

ys24 = sort(lreturns_24);
T24 = length(ys24);

display('------------------------------')	
display('Ripple VaR Daily')
VaR_Ripple95_24 = -ys24(int16(ceil(0.05*T24)));
VaR_Ripple99_24 = -ys24(int16(ceil(0.01*T24)));
display(sprintf('Daily VaR Ripple 95: %.9f', VaR_Ripple95_24))
display(sprintf('Daily VaR Ripple 99: %.9f', VaR_Ripple99_24))
display('------------------------------')

display('------------------------------')	
display('Ripple CVaR Daily')
VaR_Ripple95_24 = -mean(ys24(1:ceil(0.05*T24)));
VaR_Ripple99_24 = -mean(ys24(1:ceil(0.01*T24)));
display(sprintf('Daily CVaR Ripple 95: %.9f', VaR_Ripple95_24))
display(sprintf('Daily CVaR Ripple 99: %.9f', VaR_Ripple99_24))
display('------------------------------')

%% parametric var and cvar %公式-mu + sigma*z
alpha=0.05;
loss=-lreturns;
m=mean(loss);
s=std(loss);
pd=fitdist(lreturns,'tlocationscale');
nu=pd.nu;
t=icdf('tLocationScale',1-alpha,0,1,nu);
var=-m+s*sqrt((nu-2)/nu)*t
cvar=-m+s*sqrt((nu -2)/nu)*pdf('tLocationScale',t,0,1,nu)*(nu+t.^2)/alpha/(nu-1)

alpha1=0.01;
loss=-lreturns;
m=mean(loss);
s=std(loss);
pd=fitdist(lreturns,'tlocationscale');
nu=pd.nu;
t1=icdf('tLocationScale',1-alpha1,0,1,nu);
var1=-m+s*sqrt((nu-2)/nu)*t1
cvar1=-m+s*sqrt(( nu -2)/nu)*pdf('tLocationScale',t1,0,1,nu)*(nu+t1.^2)/alpha1/(nu-1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Fitting right & left tail via Maximum Likelihood & Bootstrap %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = 0.1; % Defining tails as top p% of returns (both positive and negative)
bts = 0.8; % Fraction of data to be retained in each bootstrap sample
Nbts = 500; % Number of bootstrap samples
alpha = 0.9; % Significance level

figure,

%%% Right tail

r = sort(r); % Sorting returns
r_right = r(round((1-p)*length(r)):end); % Selecting top p% of returns

N = length(r_right); % Number of returns selected as right tail
alpha_right = N/sum(log(r_right/min(r_right))); % Maximum-likelihood estimate for right tail exponent

fprintf('Right tail exponent: %4.3f\n',alpha_right)

x_right = linspace(min(r_right),max(r_right),100);
y_right = alpha_right*(x_right/min(r_right)).^(-alpha_right-1)/min(r_right); % Power law distribution

[b_right,a_right] = histnorm(r_right,20);

subplot(1,2,1)
loglog(a_right,b_right,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
loglog(x_right,y_right,'r','LineWidth',2)
set(gca,'FontSize',20)
title('Right tail')

%%% Right tail with bootstrap

alpha_right_bts = []; % Vector to collect bootstrap estimates for right tail exponent

for i = 1:Nbts
   
    r_bts = r(randperm(length(r))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    r_bts = sort(r_bts); % Sorting bootstrapped returns
    r_right_bts = r_bts(round((1-p)*length(r_bts)):end); % Selecting top p% of returns
    
    N_bts = length(r_right_bts); % Number of bootstrapped returns
    
    alpha_right_bts = [alpha_right_bts; N_bts/sum(log(r_right_bts/min(r_right_bts)))];

end

alpha_right_bts = sort(alpha_right_bts); % Sorting bootstrap estimates for right tail exponent

fprintf('Right tail interval at %3.2f CL: [%4.3f; %4.3f] \n',alpha,alpha_right_bts(round(0.5*(1-alpha)*Nbts)),alpha_right_bts(round(0.5*(1+alpha)*Nbts)))
fprintf('\n')

%%% Left tail

r_left = r(1:round(p*length(r))); % Selecting bottom p% of returns
r_left = abs(r_left); % Converting negative returns to positive numbers

N = length(r_left); % Number of returns selected as left tail
alpha_left = N/sum(log(r_left/min(r_left))); % Maximum-likelihood estimate for left tail exponent

fprintf('Left tail exponent: %4.3f\n',alpha_left)

x_left = linspace(min(r_left),max(r_left),100);
y_left = alpha_left*(x_left/min(r_left)).^(-alpha_left-1)/min(r_left); % Power law distribution

[b_left,a_left] = histnorm(r_left,20);

subplot(1,2,2)
loglog(a_left,b_left,'ob','MarkerSize',8,'MarkerFaceColor','b')
hold on
loglog(x_left,y_left,'r','LineWidth',2)
set(gca,'FontSize',20)
title('Left tail')

%%% Left tail with bootstrap

alpha_left_bts = []; % Vector to collect bootstrap estimates for left tail exponent

for i = 1:Nbts
   
    r_bts = r(randperm(length(r))); % Random permutation of returns
    r_bts = r_bts(1:round(bts*length(r_bts))); % Bootstrapping bts% of returns 
    r_bts = sort(r_bts); % Sorting bootstrapped returns
    r_left_bts = r_bts(1:round(p*length(r_bts))); % Selecting bottom p% of returns
    r_left_bts = abs(r_left_bts); % Converting returns to positive
    
    N_bts = length(r_left_bts); % Number of bootstrapped returns
    
    alpha_left_bts = [alpha_left_bts; N_bts/sum(log(r_left_bts/min(r_left_bts)))];

end

alpha_left_bts = sort(alpha_left_bts); % Sorting bootstrap estimates for right tail exponent

fprintf('Left tail interval at %3.2f CL: [%4.3f; %4.3f] \n',alpha,alpha_left_bts(round(0.5*(1-alpha)*Nbts)),alpha_left_bts(round(0.5*(1+alpha)*Nbts)))
fprintf('\n')

%% quantile
lretSorted = sort(lreturns); %sorts log returns
rank = transpose(1:length(lreturns)); % create rank of log returns

%Creates table to help see what is going on
%and to check quantiles produced look OK
% column 1: rank
% column 2: percentile as a fraction (using N not N+1 to begin with)
% column 3: The sorted log returns
lretRank = [rank rank/length(lreturns) lretSorted]; 

T = length(lretRank); 
Perc_01 = lretSorted(ceil(0.01*T)); 
Perc_05 = lretSorted(ceil(0.05*T)); 
Perc_10 = lretSorted(ceil(0.10*T)); 
Perc_25 = lretSorted(ceil(0.25*T)); 
Perc_50 = lretSorted(ceil(0.50*T)); 
Perc_75 = lretSorted(ceil(0.75*T)); 
Perc_90 = lretSorted(ceil(0.90*T)); 
Perc_95 = lretSorted(ceil(0.95*T)); 
Perc_99 = lretSorted(ceil(0.99*T)); 

Perc_01
Perc_05
Perc_10
Perc_25
Perc_50
Perc_75
Perc_90
Perc_95
Perc_99

