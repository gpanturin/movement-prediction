function [X, Y ,tp , fp, fn] = event_triggered_feature_extraction(mainpath)
    
    F_s = 25; %Hz    
    
    % reading data from data folder
    path_split = split(mainpath,'\');
    data = dir([mainpath filesep '..' filesep char(path_split(end)) filesep]);
    data(1:2) = [];
    data_files = fullfile({data.folder}, {data.name});
    num_of_std_for_tresh = 5;
    
    Acc_files_count = 1;
    Gyro_files_count = 1;
    Label_files_count = 1;

    for file_num = 1:length(data)
        if isempty(strfind(data(file_num).name, 'Acc')) == 0
            Acc_files(Acc_files_count) =  data_files(file_num);
            Acc_files_count = Acc_files_count + 1;
        end
        if isempty(strfind(data(file_num).name, 'Gyro')) == 0
            Gyro_files(Gyro_files_count) =  data_files(file_num);
            Gyro_files_count = Gyro_files_count + 1;
        end
        if isempty(strfind(data(file_num).name, 'Label')) == 0
            Label_files(Label_files_count) =  data_files(file_num);
            Label_files_count = Label_files_count + 1;
        end
    end
   

    X = [];     % features matrix
    Y = [];     % Label vector
    event_count = 0; % num of tresh crossing
    tot_labels_to_recognaize = 0;

    tp=0;
    fp=0;
    tn=0;
    fn=0;

    for n = 1:length(Acc_files)
        A = readtable(Acc_files{1,n});     % Acc data
        G = readtable(Gyro_files{1,n});    % Gyro data
        L = readtable(Label_files{1,n});   % Label data
        % for fixing capital letter issues 
        L.Properties.VariableNames = lower(L.Properties.VariableNames); 

%         % extracting filtered data into variabels 
%         Acc_x = lowpass(A.x_axis_g_,1,F_s);
%         Acc_y = lowpass(A.y_axis_g_,1,F_s);
%         Acc_z = lowpass(A.z_axis_g_,1,F_s);
%         Gyro_x = lowpass(G.x_axis_deg_s_,1,F_s);
%         Gyro_y = lowpass(G.y_axis_deg_s_,1,F_s);
%         Gyro_z = lowpass(G.z_axis_deg_s_,1,F_s);

        % extracting filtered data into variabels 
        Acc_x = A.x_axis_g_;
        Acc_y = A.y_axis_g_;
        Acc_z = A.z_axis_g_;
        Gyro_x = G.x_axis_deg_s_;
        Gyro_y = G.y_axis_deg_s_;
        Gyro_z = G.z_axis_deg_s_;


        % equivalence of vectors length
        data_vec_size = min(length(Acc_x), length(Gyro_x));
        if (length(Acc_x) ~=  length(Gyro_x))
            Acc_x = Acc_x(1:data_vec_size); Acc_y = Acc_y(1:data_vec_size); Acc_z = Acc_z(1:data_vec_size);
            Gyro_x = Gyro_x(1:data_vec_size); Gyro_y = Gyro_y(1:data_vec_size); Gyro_z = Gyro_z(1:data_vec_size);
        end
        
        % parameters to define the event trigger window
        mean_Gyro_z = mean(abs(Gyro_z));
        std_Gyro_z = std(abs(Gyro_z));
        mean_Gyro_y = mean(abs(Gyro_y));
        std_Gyro_y = std(abs(Gyro_y));
        mean_Gyro_x = mean(abs(Gyro_x));
        std_Gyro_x = std(abs(Gyro_x));
        mean_Acc_y = mean(abs(Acc_y));
        std_Acc_y = std(abs(Acc_y));
        mean_Acc_x = mean(abs(Acc_x));
        std_Acc_x = std(abs(Acc_x));

        Acc_time = A.elapsed_s_;
        Gyro_time = G.elapsed_s_;
        label_times = L.secondsfromrecordingstart;

        tot_labels_to_recognaize = tot_labels_to_recognaize + sum(L.label~=6);  % amount of labels that have to be rocognazied


        % finding events in the signal with treshold of mean + 5 times std 
        event_triggered_samples = find(abs(Gyro_x)>mean_Gyro_x+num_of_std_for_tresh*std_Gyro_x|abs(Gyro_y)>mean_Gyro_y+num_of_std_for_tresh*std_Gyro_y|abs(Gyro_z)>mean_Gyro_z+num_of_std_for_tresh*std_Gyro_z);
        event_triggered_beginings = [event_triggered_samples(1)];

        for i=1:length(event_triggered_samples)-1
            if event_triggered_samples(i+1) > event_triggered_samples(i)+100   % creates max 4 sec gap between events 
               event_triggered_beginings = [event_triggered_beginings event_triggered_samples(i+1)]; % events starts in indexses 
            end
        end
        
        event_triggered_times_list = Gyro_time(event_triggered_beginings); %events starts in seconds 
        label_count=0;
        count_6 = 0;

        for event = 1:length(event_triggered_times_list)
            event_count = event_count+1;
            %creating window of 15 seconds around the time of the events 
            window = max(event_triggered_times_list(event)-7,1):min(event_triggered_times_list(event)+8, Gyro_time(end));
            window_in_indexes = max(event_triggered_beginings(event)-5*F_s,1):min(event_triggered_beginings(event)+10*F_s,data_vec_size);
            X(event_count,:) = extract_features(Acc_x(window_in_indexes),Acc_y(window_in_indexes),Acc_z(window_in_indexes),Gyro_x(window_in_indexes),Gyro_y(window_in_indexes),Gyro_z(window_in_indexes));
            false_count = 0;
            for j = 1:length(label_times)
                if window(1) <= label_times(j) && label_times(j) <= window(end) % if one of the labels happend in the time of the window so this is ture positive 
                    label = L.label(j);
                    tp = tp+1;
                   
                    if label >= 1 && label <=5
                        label_count = label_count +1;
                    else
                        count_6 = count_6+1;
                    end
                    Y(event_count) = label;
                else
                    label=0;
                    false_count = false_count +1;
                end
            end

            if false_count == length(label_times)
                fp = fp+1;
                 Y(event_count) = 0;
            end

        end

        % -------- check if the treshold missed == calculating fn
        if length(label_times) - label_count - count_6 > 0
            fn = fn + sum(L.label~=6) - label_count;
            
        end
        X = normalize(X, 'medianiqr');

    end