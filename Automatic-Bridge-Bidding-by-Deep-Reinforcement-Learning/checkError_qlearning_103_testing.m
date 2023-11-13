n = size(data_t, 2);
m = floor(n/batchsizev);
totalcost=0;
Data = cell(1,totalbid);
AAv = cell(1,totalbid);
Iminv = cell(1,totalbid);
finalbid = cell(1,totalbid);
finaladdbid = cell(1,totalbid);
pass = cell(1,totalbid);
passadd = cell(1,totalbid);
for k = 1:m
    bid=1;
    a = floor(n * (k - 1) / m) + 1;
    b = floor(n * k / m);
    temp_batchsize = b-a+1;
    % lastfinal is a vector where the finalbid of the bidding is recorded
    lastfinal = zeros(1,temp_batchsize);
    Data{1} = data_t(1:52, a:b);
    % the hand player 1 sees
    Data1 = (data_t(1:52, a:b));
    % the hand player 2 sees
    Data2 = (data_t(53:104, a:b));
    AAv{1} = update_dnn(Data{1}, WW_qlearning{1}, BB_qlearning{1});
    [~,Iminv{1}]= min(AAv{1}{size(AAv{1},2)});
    finalbid{1} = Iminv{1};
    finaladdbid{1} = zeros(36,batchsizev);
    finaladdbid{bid} (sub2ind(size(finaladdbid{bid}),finalbid{bid},1:batchsizev))=1;
    finalbid{1} = Iminv{1};
    Ref = (cost_t(:, a:b));
    passstage = [];
    for bid = 2:totalbid
        odd = mod(bid,2);
        if odd ==1
            Data{bid}=Data1;
        else
            Data{bid}=Data2;
        end
        pass{bid-1} = zeros(1,batchsizev);
        passadd{bid-1} = zeros(1,batchsizev);
        dummy = ones(5,batchsizev);
        tempData = zeros(36,temp_batchsize);
        tempData(sub2ind(size(tempData),finalbid{bid-1},1:temp_batchsize))=1;
        if bid>=2
            Data{bid} = [Data{bid};finaladdbid{bid-1};dummy];
        end
        if bid>2
            pass{bid-1}(finalbid{bid-1}==finalbid{bid-2} & passadd{bid-2}==0 )=1;
            passadd{bid-1} = pass{bid-1}+passadd{bid-2};
            Reftemp = Ref(:,pass{bid-1}==1);
            [cost_for_batch]=sum(Reftemp(sub2ind(size(Reftemp), finalbid{bid-1}(pass{bid-1}==1),1:sum(pass{bid-1}==1))),2);
            %lastfinal(pass{bid-1}==1) = gather(finalbid{bid-1}(pass{bid-1}==1));
            totalcost = totalcost + (cost_for_batch)/(temp_batchsize);
        end
        AAv{bid} = update_dnn(Data{bid}, WW_qlearning{bid}, BB_qlearning{bid});
        for count=1:36
            AAv{bid}{size(AAv{bid},2)}(count,(finalbid{bid-1}>repmat(count,1,batchsizev)))=1.1;
        end
        [~,Iminv{bid}]= min(AAv{bid}{size(AAv{bid},2)});
        finalbid{bid} = Iminv{bid};
        finaladdbid{bid} = finaladdbid{bid-1};
        finaladdbid{bid} (sub2ind(size(finaladdbid{bid}),finalbid{bid},1:batchsizev))=1;
        if bid == totalbid
            Reftemp2 = Ref(:,passadd{bid-1}==0);
            [cost_for_batch]=sum(Reftemp2(sub2ind(size(Reftemp2), finalbid{bid}(:,passadd{bid-1}==0),1:sum(passadd{bid-1}==0))),2);
            totalcost = totalcost + (cost_for_batch)/(temp_batchsize);
            %lastfinal(:,passadd{bid-1}==0) = gather(finalbid{bid}(:,passadd{bid-1}==0));
        end
    end
end
average_cost = totalcost / m;
fprintf('maximum allowed bid is = %.0f  \n', totalbid);
fprintf('total testing cost = %.4f  \n', average_cost*25);
% the final contract for the bids are saved in lastfinal
