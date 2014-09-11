function S = stim(n, nt, ns)
    mags = [2, 5, 8];
    pulses = [-fliplr(mags) 0 mags];
    inds = randi(numel(pulses), n, nt, 1);
    St = pulses(inds);
    S = zeros(n, nt, ns);
    for ii = 1:numel(St)
        aa = fix((ii-1)/nt)+1;
        bb = mod(ii-1,nt)+1;
        np = St(aa, bb);
        row = [zeros(ns-abs(np), 1); ones(abs(np), 1)];
        S(aa, bb, :) = (2*(np>0) - 1)*row(randperm(ns));
    end
end
