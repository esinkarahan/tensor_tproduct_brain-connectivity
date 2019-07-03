function C = calculateAutoCov(X,R)
% Calculates the first block row of the autocovariance tensor of a 
% multivariate Gaussian process
% INPUTS
% X    : Nvoxel x Ntime
% R    : Nlag
% OUTPUT
% C    : Nvoxel x Nvoxel x Nlag 
% C = H*H' / Ntime
% H is a block toeplitz matrix formulated as:
%|x(1),...,x(Ntime) 0 0 0 ... 0| 
%|0,x(1),...,x(Ntime) 0 0 ... 0| 
%|0,0,x(1)...,x(Ntime)  0 ... 0| 
%|...                          |
%|0,...,0,...,0x(1)...,x(Ntime)| 
% where x(1) is Nv by 1 vector
%
% Version 1 - May 2015

[Nv,Nt] = size(X);
H = zeros(Nv*(R+1),Nt+R);

for i = 1 : R+1
    H((i-1)*Nv+1:i*Nv,i:Nt+i-1)= X;
end

C = H(1:Nv,:)*H'/Nt;
C = reshape(C,Nv,Nv,R+1);
end
