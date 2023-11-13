function [feature, feature_v] = get_feature(data,data_validate)
    feature = cell(2,1);
    feature{1} = zeros(5,size(data,2));
    feature{2} = zeros(5,size(data,2));
    feature_v = cell(2,1);
    feature_v{1} = zeros(5,size(data_validate,2));
    feature_v{2} = zeros(5,size(data_validate,2));
    % mask is an vector to get the high-card points for each hand
    mask = [zeros(1,9),1,2,3,4,zeros(1,9),1,2,3,4,zeros(1,9),1,2,3,4,zeros(1,9),1,2,3,4];
    %calculate the number of cards for spade,heart,diamond,clubs and
    %high-card points   
    for ii = 1:4
       for iii = 1:2
           feature{iii}(ii,:) = sum(data(ii*13-12+(iii-1)*52:ii*13+(iii-1)*52,:));
           feature_v{iii}(ii,:) = sum(data_validate(ii*13-12+(iii-1)*52:ii*13+(iii-1)*52,:));
       end
    end
    for iii = 1:2
        feature{iii}(5,:) = mask * data(iii*52-51:iii*52,:);
        feature_v{iii}(5,:) = mask * data_validate(iii*52-51:iii*52,:);
    end
end