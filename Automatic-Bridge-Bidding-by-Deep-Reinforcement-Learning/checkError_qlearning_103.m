n = size(data_validate, 2);
m = floor(n/batchsizev);
totalcost=0;
totalcount =0;
Datav = cell(1,totalbid);
AAv = cell(1,totalbid);
Iminv = cell(1,totalbid);
finalbidv = cell(1,totalbid);
finaladdbidv = cell(1,totalbid);
randbidv = cell(1,totalbid);
passv = cell(1,totalbid);
passaddv = cell(1,totalbid);
for k = 1:m
    bid=1;
    a = floor(n * (k - 1) / m) + 1;
    b = floor(n * k / m);
    temp_batchsize = b-a+1;
    % lastfinal is a vector where the finalbid of the bidding is recorded
    lastfinal = zeros(1,temp_batchsize);
    Datav{bid} = data_validate(1:52, a:b);
    % the hand player 1 sees
    Datav1 = (data_validate(1:52, a:b));
    % the hand player 2 sees
    Datav2 = (data_validate(53:104, a:b));
    AAv{bid} = update_dnn(Datav{bid}, WW_qlearning{bid}, BB_qlearning{bid});
    keyboard()
    [~,Iminv{bid}]= min(AAv{bid}{size(AAv{bid},2)});
    % the first bid (finalbidv{1}) is determined by the one with the lowest cost
    finalbidv{bid} = Iminv{bid};
    % the total bidding histoty until bid i is recorded in finaladdbidv{i}
    finaladdbidv{bid} = zeros(36,batchsizev);
    % update finaladdbidv{1} by finalbidv{1}
    finaladdbidv{bid}(sub2ind(size(finaladdbidv{bid}),finalbidv{bid},1:batchsizev))=1;
    finalbidv{bid} = Iminv{bid};
    % Ref is the cost array for the current batch
    % Ref = (cost_validate(:, a:b));
    passstage = [];
    for bid = 2:totalbid
        % if odd==1, then player 1 is bidding
        % if odd==0, then player 2 is bidding
        odd = mod(bid,2);
        if odd ==1
            Datav{bid}=Datav1;
        else
            Datav{bid}=Datav2;
        end
        % passv{bid-1} records the bids that first passed in {bid-1} for bid >2
        % the pass here refers to pass so the bidding stops
        % so the first bid pass by player 1 doesnt count here
        % passaddv{bid-1} records all bids that passed from 1 to bid-1
        passv{bid-1} = zeros(1,batchsizev);
        passaddv{bid-1} = zeros(1,batchsizev);
        % dummy is 5 ones in the input dimension, only for the update_dnn
        % function
        dummy = ones(5,batchsizev);
        if bid>=2
            Datav{bid} = [Datav{bid};finaladdbidv{bid-1};dummy];
        end
        if bid>2
            % update passv and passaddv
            passv{bid-1}(finalbidv{bid-1}==finalbidv{bid-2} & passaddv{bid-2}==0 )=1;
            passaddv{bid-1} = passv{bid-1}+passaddv{bid-2};
            % record the cost for bids that have passed, so that the total
            % cost can be calculated
            % Ref is the cost array, and Reftemp is the cost array for
            % bids that have passed in bid-1
            % Reftemp = Ref(:,passv{bid-1}==1);
            % [cost_for_batch]=sum(Reftemp(sub2ind(size(Reftemp), finalbidv{bid-1}(passv{bid-1}==1),1:sum(passv{bid-1}==1))),2);
            %lastfinal(passv{bid-1}==1) = gather(finalbidv{bid-1}(passv{bid-1}==1));
            % add the cost of bids where first pass that ends the bidding
            % happened in bid-1 to totalcost
            % also record the totalcount to assure the calculated bids are
            % right
            % totalcost = totalcost + (cost_for_batch)/(temp_batchsize);
            % totalcount = totalcount + sum(passv{bid-1}==1);
        end
        % get the bid by calling the update_dnn function with the trainined
        % model WW_qlearning and BB_qlearning
        AAv{bid} = update_dnn(Datav{bid}, WW_qlearning{bid}, BB_qlearning{bid});
        keyboard()
        % disallow bids against bridge rules
        for count=1:36
            AAv{bid}{size(AAv{bid},2)}(count,(finalbidv{bid-1}>repmat(count,1,batchsizev)))=Inf;
        end
        [~,Iminv{bid}]= min(AAv{bid}{size(AAv{bid},2)});
        finalbidv{bid} = Iminv{bid};
        finaladdbidv{bid} = finaladdbidv{bid-1};
        finaladdbidv{bid} (sub2ind(size(finaladdbidv{bid}),finalbidv{bid},1:batchsizev))=1;
        % calculate the cost of all bids when the final allowed bid has
        % been reached
        %if bid == totalbid
        %    Reftemp2 = Ref(:,passaddv{bid-1}==0);
        %    [cost_for_batch]=sum(Reftemp2(sub2ind(size(Reftemp2), finalbidv{bid}(:,passaddv{bid-1}==0),1:sum(passaddv{bid-1}==0))),2);
        %    totalcost = totalcost + (cost_for_batch)/(temp_batchsize);
        %    %lastfinal(:,passaddv{bid-1}==0) = gather(finalbidv{bid}(:,passaddv{bid-1}==0));
        %    totalcount = totalcount + sum(passaddv{bid-1}==0);
        %    keyboard()
        %end
    end
end
% confirm that the number of cost calculated is equal to the number of
% totalbids
assert(totalcount ==  size(data_validate, 2));
% get the average IMP for the training
average_cost = totalcost / m;
fprintf('maximum allowed bid is = %.0f  \n', totalbid);
fprintf('total validation cost = %.4f  \n', average_cost*25);
% the final contract for the bids are saved in lastfinal
