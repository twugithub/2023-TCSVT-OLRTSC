function [ D ] = OTLRR_solve_D( D, M, A, B, lambda1, lambda3 )

[~,d,n3] = size(D);

I = eye(d,d);

Afft = fft(A,[],3);
Bfft = fft(B,[],3);
Mfft = fft(M,[],3);

A_hat = zeros(size(A));
B_hat = zeros(size(B));

for i = 1:n3
    A_hat(:,:,i) = lambda1 * Afft(:,:,i) + lambda3 * I;
    B_hat(:,:,i) = lambda1 * Bfft(:,:,i) + lambda3 * Mfft(:,:,i);
end

Dfft = fft(D,[],3);

for k = 1:n3
    for j = 1:d
        
        djk = Dfft(:,j,k);
        ajk = A_hat(:,j,k);
        bjk = B_hat(:,j,k);
        
        tmp = djk - (Dfft(:,:,k) * ajk - bjk) / A_hat(j,j,k);
        Dfft(:,j,k) = tmp;
    end
end

D = ifft(Dfft,[],3);

end