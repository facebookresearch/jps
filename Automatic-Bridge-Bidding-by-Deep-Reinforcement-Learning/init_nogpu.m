function [WW, BB, dW, dB] = init_nogpu(isize, osize, lsize, depth)
    WW{1} = rand(lsize, isize) * 0.2 - 0.1;
    BB{1} = rand(lsize, 1) * 0.2 - 0.1;
    dW{1} = zeros(lsize, isize);
    dB{1} = zeros(lsize, 1);
    for i = 2:depth
        WW{i} = rand(lsize, lsize) * 0.2 - 0.1;
        BB{i} = rand(lsize, 1) * 0.2 - 0.1;
        dW{i} = zeros(lsize, lsize);
        dB{i} = zeros(lsize, 1);
    end
    WW{depth + 1} = rand(osize, lsize) * 0.2 - 0.1;
    BB{depth + 1} = rand(osize, 1) * 0.2 - 0.1;
    dW{depth + 1} = zeros(osize, lsize);
    dB{depth + 1} = zeros(osize, 1);
    for i = 1:depth + 1
        WW{i} = single(WW{i});
        BB{i} = single(BB{i});
    end
    %for i = 1:depth + 1
    %    WW{i} = gpuArray(WW{i});
    %    BB{i} = gpuArray(BB{i});
    %    dW{i} = gpuArray(dW{i});
    %    dB{i} = gpuArray(dB{i});
    %end
end
