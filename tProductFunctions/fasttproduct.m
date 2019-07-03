function result = fasttproduct(A,B)
% Cosine based tensor product of Kernfeld, Kilmer, Aeron
% INPUTS: A lxmxn B mxpxn
% OUTPUT: lxpxn
% Reference: E Kernfeld, S Aeron, ME Kilmer, "Clustering multi-way data: 
% a novel algebraic approach", ArXiv, 2014

dimsA = size(A);
dimsB = size(B);
if (dimsA(2)~=dimsB(1)) || (dimsA(3)~=dimsB(3))
    error('tensors not commensurate')
end
Ahat = fft( A,[],3 );
Bhat = fft( B,[],3 );
result=ifft( face_mult(Ahat,Bhat),[],3 );
end

