
!bart phantom -k -3 -x 256 phantom

fov = [12,12,12];

res = fov/256;

j = -128:127;
[mx,my,mz] = ndgrid(j,j,j);

sx = 0.5; %rad/mm
sy = 0.2;
sz = 0.9;

px = (sx * res(1)) * mx;
py = (sy * res(2)) * my;
pz = (sz * res(3)) * mz;

x = readcfl("phantom");

nstd = 1e-8;
noise = complex(normrnd(0,nstd,size(x)),normrnd(0,nstd,size(x)));
x = x + noise;

x = fftshift(ifftn(ifftshift(x)));
x = x .* exp(1i * (px + py + pz) );

disp_vol_center(angle(x));

x = fftshift(fftn(ifftshift(x)));

disp_vol_center(abs(x));




msk = abs(mx) <= 128 & abs(my) <= 3 & abs(mz) <= 3;



x = x(abs(j) < 128,abs(j) <= 3,abs(j) <= 3);

x = reshape(x,size(x,1),[]);
size(x)

writecfl("phant_test",x);