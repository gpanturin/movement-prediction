function [X_row] = extract_features(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
    X_row=zeros(1,177);
    % mean , std , max features for accelerometer
    X_row(1) = mean(acc_x);
    X_row(2) = std(acc_x); 
    X_row(3) = max(acc_x);
    X_row(4) = mean(acc_y);
    X_row(5) = std(acc_y); 
    X_row(6) = max(acc_y);
    X_row(7) = mean(acc_z);
    X_row(8) = std(acc_z);
    X_row(9) = max(acc_z);
    X_row(10) = mean(acc_x.*acc_y);
    X_row(11) = std(acc_x.*acc_y);
    X_row(12) = max(acc_x.*acc_y);

    % mean , std , max features for gyroscope
    X_row(13) = mean(gyro_x);
    X_row(14) = std(gyro_x);
    X_row(15) = max(gyro_x);
    X_row(16) = mean(gyro_y); 
    X_row(17) = std(gyro_y);
    X_row(18) = max(gyro_y);
    X_row(19) = mean(gyro_z);
    X_row(20) = std(gyro_z);
    X_row(21) = max(gyro_z);
    X_row(22) = mean(gyro_x.*gyro_y);
    X_row(23) = std(gyro_x.*gyro_y);
    X_row(24) = max(gyro_x.*gyro_y);

    % subtracting standard deviations features
    X_row(25) = std(acc_z)-std(acc_x);
    X_row(26) = std(acc_z)-std(acc_y);
    X_row(27) = std(acc_y)-std(acc_x);
    X_row(28) = std(gyro_z)-std(gyro_x);
    X_row(29) = std(gyro_z)-std(gyro_y);
    X_row(30) = std(gyro_y)-std(gyro_x);

    % linearity check features
    X_row(31) = sum(abs(acc_x-detrend(acc_x))); 
    X_row(32) = sum(abs(acc_y-detrend(acc_y)));
    X_row(33) = sum(abs(acc_z-detrend(acc_z)));
    X_row(34) = sum(abs(gyro_x-detrend(gyro_x)));
    X_row(35) = sum(abs(gyro_y-detrend(gyro_y)));
    X_row(36) = sum(abs(gyro_z-detrend(gyro_z)));

    % power spectral density  estimate features
    [~, f_axis] = pwelch(acc_x,25);
    [P, f] = pwelch(acc_x, 25, 1, f_axis, 25);    
    
    X_row(37) = mean(P);
    X_row(38) = abs(sum(P));
    
    [~, f_axis] = pwelch(acc_y,25);
    [P, f] = pwelch(acc_y, 25, 1,f_axis, 25);
    
    X_row(39) = mean(P);
    X_row(40) = abs(sum(P));
    
    [~, f_axis] = pwelch(acc_z,25);
    [P, f] = pwelch(acc_z, 25, 1, f_axis, 25);
   
    X_row(41) = mean(P);
    X_row(42) = abs(sum(P));
    
    [~, f_axis] = pwelch(gyro_x,25);
    [P, f] = pwelch(gyro_x, 25, 1, f_axis, 25);
 
    X_row(43) = mean(P);
    X_row(44) = abs(sum(P));
    
    [~, f_axis] = pwelch(gyro_y,25);
    [P, f] = pwelch(gyro_y, 25, 1, f_axis, 25);
  
    X_row(45) = mean(P);
    X_row(46) = abs(sum(P));
    
    [~, f_axis] = pwelch(gyro_z,25);
    [P, f] = pwelch(gyro_z, 25, 1, f_axis, 25);
  
    X_row(47) = mean(P);
    X_row(48) = abs(sum(P));

    % minimum features
    X_row(49) = min(acc_x);
    X_row(50) = min(acc_y);
    X_row(51) = min(acc_z);
    X_row(52) = min(gyro_x);
    X_row(53) = min(gyro_y);
    X_row(54) = min(gyro_z); 

    % subtracting min from max values features
    X_row(55) = max(acc_x) - min(acc_x);
    X_row(56) = max(acc_y) - min(acc_y); 
    X_row(57) = max(acc_z) - min(acc_z);
    X_row(58) = max(gyro_x) - min(gyro_x); 
    X_row(59) = max(gyro_y) - min(gyro_y);
    X_row(60) = max(gyro_z) - min(gyro_z);   

    % slope features
    dt = 1/25;
    t = 0:dt:(length(acc_x)-1)*dt;
    X_row(61) = mean(diff(acc_x)./diff(t)');
    t = 0:dt:(length(acc_y)-1)*dt;
    X_row(62) = mean(diff(acc_y)./diff(t)');
    t = 0:dt:(length(acc_z)-1)*dt;
    X_row(63) = mean(diff(acc_z)./diff(t)');
    t = 0:dt:(length(gyro_x)-1)*dt;
    X_row(64) = mean(diff(gyro_x)./diff(t)');
    t = 0:dt:(length(gyro_y)-1)*dt;
    X_row(65) = mean(diff(gyro_y)./diff(t)');
    t = 0:dt:(length(gyro_z)-1)*dt;
    X_row(66) = mean(diff(gyro_z)./diff(t)');

    % envelope integral features
    [u, l] = envelope(acc_x);
    X_row(67) = trapz(u-l);
    [u, l] = envelope(acc_y);
    X_row(68) = trapz(u-l);
    [u, l] = envelope(acc_z);
    X_row(69) = trapz(u-l);
    [u, l] = envelope(gyro_x);
    X_row(70) = trapz(u-l);
    [u, l] = envelope(gyro_y);
    X_row(71) = trapz(u-l);
    [u, l] = envelope(gyro_z);
    X_row(72) = trapz(u-l);
    

    % median features
    X_row(73) = median(acc_x);
    X_row(74) = median(acc_y);
    X_row(75) = median(acc_z);
    X_row(76) = median(gyro_x);
    X_row(77) = median(gyro_y);
    X_row(78) = median(gyro_z);

    % maximum diff between followed values features
    X_row(79) = max(diff(acc_x));
    X_row(80) = max(diff(acc_y));
    X_row(81) = max(diff(acc_z));
    X_row(82) = max(diff(gyro_x));
    X_row(83) = max(diff(gyro_y));
    X_row(84) = max(diff(gyro_z));

    % moving avarage features
    X_row(85) = max(movmean(acc_x, 3));
    X_row(86) = max(movmean(acc_y, 3));
    X_row(87) = max(movmean(acc_z, 3));
    X_row(88) = max(movmean(gyro_x, 3));
    X_row(89) = max(movmean(gyro_y, 3));
    X_row(90) = max(movmean(gyro_z, 3));

    % asymmetry of a distribution level features
    X_row(91) = skewness(acc_x);
    X_row(92) = skewness(acc_y);
    X_row(93) = skewness(acc_z);
    X_row(94) = skewness(gyro_x);
    X_row(95) = skewness(gyro_y);
    X_row(96) = skewness(gyro_z);
    
    % EMD features 
    [Acc_x_imf,Acc_x_res,Acc_x_info] = emd(acc_x);
    [Acc_y_imf,Acc_y_res,Acc_y_info] = emd(acc_y);
    [Acc_z_imf,Acc_z_res,Acc_z_info] = emd(acc_z);
    [Gyro_x_imf,Gyro_x_res,Gyro_x_info] = emd(gyro_x);
    [Gyro_y_imf,Gyro_y_res,Gyro_y_info] = emd(gyro_y);
    [Gyro_z_imf,Gyro_z_res,Gyro_z_info] = emd(gyro_z);

    % EMD features for Acc_x
    X_row(97) = mean(Acc_x_info.NumExtrema);
    X_row(98) = mean(Acc_x_info.NumZerocrossing);
    X_row(99) = mean(Acc_x_info.MeanEnvelopeEnergy);
    X_row(100) = mean(Acc_x_info.RelativeTolerance);
    X_row(101) = mean(Acc_x_res);
    X_row(102) =  mean(mean(Acc_x_imf));

    % EMD features for Acc_y
    X_row(98) = mean(Acc_y_info.NumExtrema);
    X_row(99) = mean(Acc_y_info.NumZerocrossing);
    X_row(100) = mean(Acc_y_info.MeanEnvelopeEnergy);
    X_row(101) = mean(Acc_y_info.RelativeTolerance);
    X_row(102) = mean(Acc_y_res);
    X_row(103) =  mean(mean(Acc_y_imf));

    % EMD features for Acc_z
    X_row(104) = mean(Acc_z_info.NumExtrema);
    X_row(105) = mean(Acc_z_info.NumZerocrossing);
    X_row(106) = mean(Acc_z_info.MeanEnvelopeEnergy);
    X_row(107) = mean(Acc_z_info.RelativeTolerance);
    X_row(108) = mean(Acc_z_res);
    X_row(109) =  mean(mean(Acc_z_imf));
     
    % EMD features for Gyro_x
    X_row(110) = mean(Gyro_x_info.NumExtrema);
    X_row(111) = mean(Gyro_x_info.NumZerocrossing);
    X_row(112) = mean(Gyro_x_info.MeanEnvelopeEnergy);
    X_row(113) = mean(Gyro_x_info.RelativeTolerance);
    X_row(114) = mean(Gyro_x_res);
    X_row(115) =  mean(mean(Gyro_x_imf));

    % EMD features for Gyro_y
    X_row(116) = mean(Gyro_y_info.NumExtrema);
    X_row(117) = mean(Gyro_y_info.NumZerocrossing);
    X_row(118) = mean(Gyro_y_info.MeanEnvelopeEnergy);
    X_row(119) = mean(Gyro_y_info.RelativeTolerance);
    X_row(120) = mean(Gyro_y_res);
    X_row(121) =  mean(mean(Gyro_y_imf));

    % EMD features for Gyro_z
    X_row(122) = mean(Gyro_z_info.NumExtrema);
    X_row(123) = mean(Gyro_z_info.NumZerocrossing);
    X_row(124) = mean(Gyro_z_info.MeanEnvelopeEnergy);
    X_row(125) = mean(Gyro_z_info.RelativeTolerance);
    X_row(126) = mean(Gyro_z_res);
    X_row(127) =  mean(mean(Gyro_z_imf));

    % more EMD features - numsifting 
    X_row(128) = mean(Acc_x_info.NumSifting);
    X_row(129) = mean(Acc_y_info.NumSifting);
    X_row(130) = mean (Acc_z_info.NumSifting); 
    X_row(131) = mean(Gyro_x_info.NumSifting);
    X_row(132) = mean(Gyro_y_info.NumSifting);
    X_row(133) = mean(Gyro_z_info.NumSifting);
    
    % peak2rms features
    X_row(134) = peak2rms(acc_x);
    X_row(135) = peak2rms(acc_y);
    X_row(136) = peak2rms(acc_z);
    X_row(137) = peak2rms(gyro_x);
    X_row(138) = peak2rms(gyro_y);
    X_row(139) = peak2rms(gyro_z);

    % mean SNR features 
    X_row(140) = mean(snr(acc_x));
    X_row(141) = mean(snr(acc_y));
    X_row(142) = mean(snr(acc_z));
    X_row(143) = mean(snr(gyro_x));
    X_row(144) = mean(snr(gyro_y));
    X_row(145) = mean(snr(gyro_z));

    % rms features
    X_row(146) = rms(acc_x);
    X_row(147) = rms(acc_y);
    X_row(148) = rms(acc_z);
    X_row(149) = rms(gyro_x);
    X_row(150) = rms(gyro_y);
    X_row(151) = rms(gyro_z);
    
    % rms features
    X_row(152) = iqr(acc_x);
    X_row(153) = iqr(acc_y);
    X_row(154) = iqr(acc_z);
    X_row(155) = iqr(gyro_x);
    X_row(156) = iqr(gyro_y);
    X_row(157) = iqr(gyro_z);
    
    % energy features
    X_row(158)=mean(sqrt(acc_x.^2+acc_y.^2+acc_z.^2));
    X_row(159)=mean(sqrt(gyro_x.^2+gyro_y.^2+gyro_z.^2));

    % wentropy
    X_row(160) = wentropy(acc_x,'shannon');
    X_row(161) = wentropy(acc_y,'shannon');
    X_row(162) = wentropy(acc_z,'shannon');
    X_row(163) = wentropy(gyro_x,'shannon');
    X_row(164) = wentropy(gyro_y,'shannon');
    X_row(165) = wentropy(gyro_z,'shannon');

    % sum diff between followed values features
    X_row(166) = sum(diff(acc_x));
    X_row(167) = sum(diff(acc_y));
    X_row(168) = sum(diff(acc_z));
    X_row(169) = sum(diff(gyro_x));
    X_row(170) = sum(diff(gyro_y));
    X_row(171) = sum(diff(gyro_z));

    % amplutude std features
    X_row(172) = std(20*log(abs(real(fft(acc_x)))));
    X_row(173) = std(20*log(abs(real(fft(acc_y)))));
    X_row(174) = std(20*log(abs(real(fft(acc_z)))));
    X_row(175) = std(20*log(abs(real(fft(gyro_x)))));
    X_row(176) = std(20*log(abs(real(fft(gyro_y)))));
    X_row(177) = std(20*log(abs(real(fft(gyro_z)))));
   


    








end