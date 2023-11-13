function  AA = update_dnn(XX, WW, BB)
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