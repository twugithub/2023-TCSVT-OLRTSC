function [ v, e ] = OTLRR_solve_ve( z, D, lambda1, lambda2 )

[n1,d,n3] = size(D);
I = eye(d,d);

Dfft = fft(D,[],3);
aux = zeros(d,n1,n3);
for i = 1:n3
    aux(:,:,i) = (Dfft(:,:,i)' * Dfft(:,:,i) + n3/lambda1 * I) \ Dfft(:,:,i)';
end

thd = lambda2 / lambda1;

v = zeros(d,1,n3);
e = zeros(n1,1,n3);

eps = 1e-3;
maxIter = 1e3;

converge = false;
iter = 0;

while ~converge
    iter = iter + 1;
    
    oldv = v;
    olde = e;
    
    a = z - e;
    afft = fft(a,[],3);
    vfft = zeros(d,1,n3);
    for i = 1:n3
        vfft(:,:,i) = aux(:,:,i)*afft(:,:,i);
    end
    v = ifft(vfft,[],3);
    e = sign(z - tprod(D, v)) .* max(abs(z - tprod(D, v)) - thd, 0);
    
    stopc = max(sqrt(sum(sum((v - oldv).^2)))/sqrt(sum(sum(v.^2))), sqrt(sum(sum((e - olde).^2)))/sqrt(sum(sum(e.^2))));
    
    if stopc < eps || iter > maxIter
        converge = true;
    end
end

end