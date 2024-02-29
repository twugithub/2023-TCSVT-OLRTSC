function [ h, v, e ] = OTLRR_solve_missing_ve( z, D, lambda1, lambda2 )

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
y = zeros(n1,1,n3);

zsq = squeeze(z);
Indicator = ~isnan(zsq(:));
msq = zeros(size(zsq));
msq(Indicator) = zsq(Indicator);
t = reshape(msq, [size(z,1), 1, size(z,3)]);

eps = 1e-3;
maxIter = 1e3;
mu = 0.1;

converge = false;
iter = 0;

while ~converge
    iter = iter + 1;
    
    oldv = v;
    olde = e;
    
    h = (lambda1* (tprod(D, v) + e) + mu*t - y) / (lambda1 + mu);
    
    a = h - e;
    afft = fft(a,[],3);
    vfft = zeros(d,1,n3);
    for i = 1:n3
        vfft(:,:,i) = aux(:,:,i)*afft(:,:,i);
    end
    v = ifft(vfft,[],3);
    e = sign(h - tprod(D, v)) .* max(abs(h - tprod(D, v)) - thd, 0);
    
    t = h + y/mu;
    msq = squeeze(t);
    msq(Indicator) = zsq(Indicator);
    t = reshape(msq, [size(z,1), 1, size(z,3)]);
    
    y = y + mu*(h - t);
    
    stopc = max([sqrt(sum(sum((h - t).^2))), sqrt(sum(sum((v - oldv).^2)))/sqrt(sum(sum(v.^2))), sqrt(sum(sum((e - olde).^2)))/sqrt(sum(sum(e.^2)))]);
    
    if stopc < eps || iter > maxIter
        converge = true;
    end
end

end