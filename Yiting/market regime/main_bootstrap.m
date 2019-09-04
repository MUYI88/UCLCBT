

rb=rmrb;
rs=rmrs;
r = [0;log(close22(2:end)./close22(1:end-1))];
m=bootstrp(10000,@mean,r);
m1=bootstrp(10000,@mean,rb);
mn=m1-m;
mean1=mean(mn);

CI= prctile(mn,[2.5 97.5])

figure(39)
h1=histogram(mn,800);
hold on 
h11=plot([mean1,mean1],get(gca,'YLim'),'LineWidth',2)
hold on
h12=plot([CI(1),CI(1)],get(gca,'YLim'),'y','LineWidth',2)
hold on
h13=plot([0,0],get(gca,'YLim'),'--');
hold on
h14=plot([CI(2),CI(2)],get(gca,'YLim'),'y','LineWidth',2)

legend([h11,h12,h13],'Average Excess Return','95% Confidence Interval','Zero','Location','Best')


figure(40)
m2=bootstrp(10000,@mean,rs);

mn2=m2-m;
mean2=mean(mn2);

CI2= prctile(mn2,[2.5 97.5])

h1=histogram(mn2,800);
hold on 
h11=plot([mean2,mean2],get(gca,'YLim'),'LineWidth',2)
hold on
h12=plot([CI2(1),CI2(1)],get(gca,'YLim'),'y','LineWidth',2)
hold on
h13=plot([0,0],get(gca,'YLim'),'--');
hold on
h14=plot([CI2(2),CI2(2)],get(gca,'YLim'),'y','LineWidth',2)

legend([h11,h12,h13],'Average Excess Return','95% Confidence Interval','Zero','Location','Best')



