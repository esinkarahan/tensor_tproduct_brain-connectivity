function invA = tinv_NN(A,lambda,alpha)
% Find the inverse of a (covariance) tensor by using tensor nuclear norm(TNN) 
% INPUTS
% A     : Third order tensor
% lambda: Regularization parameter of TNN
% alpha : Mixture constant
% 
% OUTPUTS
% invA  : third order tensor, inverse of the A found by applying TNN on t-SVD
% 
% Copyright (C) 2015 Esin Karahan*, Pedro Ariel Rojas-Lopez', Pedro A. Valdes-Sosa'.
% * Bogazici University, Istanbul, Turkey, ' Cuban Neuroscience Center, Havana, Cuba
% contact: esin.karahan@gmail.com, pedro.rojas@cneuro.com, peter@cneuro.com
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Version 1 - May 2015
% 

[l,m,n]=size(A);
Ahat = fft(A ,[],3);
Uhat= zeros(l,l,n);
Shat= zeros(l,m,n);
Vhat= zeros(m,m,n);
C   = zeros(m,l,n); 
nn  = max(l,m);

% Loop over tubals of Ahat
for k=1:n
    [Uhat(:,:,k),Shat(:,:,k),Vhat(:,:,k)]=svd(Ahat(:,:,k));

    % nuclear norm on S
    di = diag(Shat(:,:,k));
    di = di(di~=0);
    di = di.^2; % eigenvalue
    % shrink the eigenvalues
    ei = (-nn + sqrt(nn^2+(4*lambda*alpha).*(nn.*di + lambda*(1-alpha))))./(2*lambda*alpha);
    ei = flipud(1./sqrt(ei));%back to singular value
    ei(ei<0)=0;
    Shat(:,:,k)=full(spdiags(ei,0,Shat(:,:,k)));
    % inverse
    C(:,:,k) = Vhat(:,:,k)*Shat(:,:,k)'*Uhat(:,:,k)';
end


invA=ifft(C,[],3);