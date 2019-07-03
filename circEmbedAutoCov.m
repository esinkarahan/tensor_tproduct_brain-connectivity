function S = circEmbedAutoCov(C)
% Calculates the Circulant Embedding of the Covariance Tensor 
% INPUT
% C : Covariance tensor
% OUTPUT
% S : Circulant embedding of the covariance
% Reference: CR Dietrich, GN Newsam,"Fast and Exact Simulation of Stationary 
% Gaussian Processes through Circulant Embedding of the Covariance Matrix", 
% SIAM J.on Sci.Comp.
%
% Version 1 - May 2015

[Nv,~,R] = size(C); 
S = zeros(Nv,Nv,2*R);
S(:,:,1:R) = C;
S(:,:,R+1) = 0.5.*(C(:,:,R) + C(:,:,R)');
j = R;
for i = 1:R-1
    S(:,:,i+R+1)= C(:,:,j)';
    j=j-1;
end
end