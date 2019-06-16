n=100;
m=340;
deltaxi=0.01;
nblocks=1000;
nsample=1000;
xi=-pi:deltaxi:pi;
phi=((exp(1i*xi)+exp(1i*2*xi)+exp(1i*3*xi)+exp(1i*4*xi)+exp(1i*5*xi)+exp(1i*6*xi))/6).^n;
disp(trapz(real(exp(-1i*m*xi).*phi))*deltaxi/(2*pi))
disp(trapz(imag(exp(-1i*m*xi).*phi))*deltaxi/(2*pi))
s=zeros(nblocks,nsample);
for i=1:nblocks
    s(i,:)=sum(randi(6,n,nsample));
end
