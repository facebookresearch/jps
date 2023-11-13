function [sW, sB] = sinit_nogpu(train_data, train_ref, WW, BB, output)
    lala = randperm(size(train_data, 2), 1000);
    RRR = 1:output;
    ref = (bsxfun(@eq, RRR.', train_ref(lala.').'));
    AA = update((train_data(:, lala)), WW, BB);
    a = size(WW, 2);
    DD{a} = AA{a + 1} .* (1 - AA{a + 1}) .* (AA{a + 1} - ref);
    for i = 1:a - 1
        DD{a - i} = (AA{a - i + 1} .* (1 - AA{a - i + 1}) .* (WW{a - i + 1}.' * DD{a - i + 1}));
    end
    for i = 1:a
        sW{i} = abs(DD{i} * AA{i}.');
        sB{i} = abs(sum(DD{i}, 2));
    end
end

function  AA = update(XX, WW, BB)
    n = size(WW, 2);
    AA{1} = XX;
    for i = 1:n
        %AA{i + 1} = sigmf(bsxfun(@plus, WW{i} * AA{i}, BB{i}), [1 0]) * drop;
        QQ = bsxfun(@plus, WW{i} * AA{i}, BB{i});
        if i ~=n
            AA{i + 1} = QQ .* ((QQ < 0) * 0.2 + (QQ > 0) * 1);
        else
            AA{i + 1} = QQ .* ((QQ < 0) * 0.2 + (QQ > 0) * 1);
        end
    end

end