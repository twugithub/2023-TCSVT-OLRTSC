function [ A ] = orth_tensor( A )

Afft = fft(A,[],3);
for i = 1:size(A,3)
    Afft(:,:,i) = orth(Afft(:,:,i));
end

A = ifft(Afft,[],3);

end