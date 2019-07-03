% This function is the caller function for Granger Causality analysis with
% t-product
%
% References:
% o Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-Order
% Tensors as Operators on Matrices: A Theoretical and Computational
% Framework with Applications in Imaging. SIAM Journal on Matrix Analysis
% and Applications, 34(1), 148–172. 
% o Dietrich, C. R., & Newsam, G. N. (1997). Fast and Exact Simulation of Stationary
% Gaussian Processes through Circulant Embedding of the Covariance Matrix.
% SIAM Journal on Scientific Computing. 
% o Chi, E. C., & Lange, K. (2014). Stable estimation of a covariance matrix 
% guided by nuclear norm penalties. Computational Statistics and Data Analysis, 80, 117–128.
%
% Version 1 - May 2015
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the data X
load Y
% normalize the data to have zero mean and unit standard deviation
Y = zscore(Y);
Y = Y';
[Nv,Nt] = size(Y);
% Determine the model order of the AR
Nlag = 5;
p = (Nlag-1)*2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the first column of the autocovariance
C = calculateAutoCov(Y,Nlag);

% Circulant embedding of Autocovariance tensor
R = circEmbedAutoCov(C(:,:,1:Nlag));
r = circEmbedAutoCov(C(:,:,2:Nlag+1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set alpha according to Chi et. al.
% Eq. 5
trC=0;
for i=1:Nlag+1
    trC = trC+trace(C(:,:,i));
end
alpha=1/(1+(trC/(Nv*(Nlag+1))).^2);

lambda = 1;

tic;
% Find the inverse of R through nuclear norm penalization
invR = tinv_NN(R,lambda,alpha);
% Calculate the AR coefficients
A    = fasttproduct(invR,r);
A = A(:,:,1:Nlag);
toc;

% Plot the results
imagesc(reshape(A,[Nv,Nv*(Nlag)]))
figure,imagesc(reshape(abs(A),[Nv,Nv*(Nlag)]))
colormap hot
xx=reshape(abs(A),[Nv,Nv*(Nlag)]);
figure,plot(xx')



