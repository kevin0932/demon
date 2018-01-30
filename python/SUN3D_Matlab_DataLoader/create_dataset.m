%% retrieve full dataset to get max frameid
% sequenceName = 'hotel_umd/maryland_hotel3';
% sequenceName = 'hotel_beijing/beijing_hotel_2';
% % sequenceName = 'mit_46_ted_lab1/ted_lab_2';
% sequenceName = 'mit_w85_lounge1/wg_gym_2';
% sequenceName = 'mit_w85k1/living_room_night';
% % sequenceName = 'mit_w16/kresge_stage';
% sequenceName = 'mit_32_123/classroom_32123_nov_2_2012_scan1_erika'
sequenceName = 'harvard_c4/hv_c4_1'

% [~, frameCount, ~, ~] = SUN3Dreader(sequenceName);
maxFrameID = GetFrameCount(sequenceName)
clearvars -except maxFrameID sequenceName

%% retrieve data for later storage
newStr = split(sequenceName,'/')
dsetName = strcat('/',newStr{1},'~',newStr{2})
% frame_ids = 1:30:round(maxFrameID/5);
frame_ids = 1:15:maxFrameID;
[Ks, frameCount, frameImages, frameDepths, frameExtrinsicsW2C] = SUN3Dreader(sequenceName,frame_ids);

%% create local directory for data storage
% destdirectory = '/home/kevin/ThesisDATA/SUN3D';
destdirectory = './SUN3D';
mkdir(destdirectory);   %create the directory
hdf5filename = strcat('GT_',newStr{1},'~',newStr{2},'.h5')
fulldestination = fullfile(destdirectory, hdf5filename); 


%% write data to HDF5 file
for frame_id = 1:length(frame_ids)
%     [K, frameCount, frameImage, frameDepth] = SUN3Dreader(sequenceName,frame_id);
    dset_camera = strcat(dsetName,'-',sprintf('%07d',frame_ids(frame_id)),'/camera');
    tmpRotMat = frameExtrinsicsW2C(:,1:3,frame_id);
    tmpRotVec = reshape(tmpRotMat,size(tmpRotMat,1)*size(tmpRotMat,2),1);
    tmpTransVec = frameExtrinsicsW2C(1:3,4,frame_id);
    cam_data = [Ks(1,1); Ks(2,2); Ks(1,2); Ks(1,3); Ks(2,3); tmpRotVec; tmpTransVec];
    h5create(fulldestination,dset_camera,[17]);
%     h5disp(fulldestination)
    h5write(fulldestination,dset_camera, cam_data);
    
    dset_image = strcat(dsetName,'-',sprintf('%07d',frame_ids(frame_id)),'/image');
    image_data = frameImages(:,:,:,frame_id);
    h5create(fulldestination,dset_image,size(image_data));
%     h5disp(fulldestination)
    h5write(fulldestination,dset_image, image_data);
    
    dset_depth = strcat(dsetName,'-',sprintf('%07d',frame_ids(frame_id)),'/depth');
    depth_data = frameDepths(:,:,frame_id);
    h5create(fulldestination,dset_depth,size(depth_data));
%     h5disp(fulldestination)
    h5write(fulldestination,dset_depth, depth_data);
%    hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
end

%hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
%%
test_dset = '/hotel_umd~maryland_hotel3-0000001/camera'

data = h5read(fulldestination,test_dset)
%%
test_dset = '/hotel_umd~maryland_hotel3-0000001/image'

data = h5read(fulldestination,test_dset)
%%
test_dset = '/hotel_umd~maryland_hotel3-0000001/depth'

data = h5read(fulldestination,test_dset)
%%
test_dset = '/rgbd_dataset_freiburg2_360_kidnap-0000605/frames/t0/v0/camera'

data = h5read('rgbd_10_to_20_handheld_train.h5',test_dset)

%%
for frame_id = 1:10:maxFrameID
%     [K, frameCount, frameImage, frameDepth] = SUN3Dreader(sequenceName,frame_id);
    dset_camera = strcat(dsetName,'-',str(frame_id),'/camera')
    h5write(fulldestination,dset_camera, K)
    dset_image = strcat(dsetName,'-',str(frame_id),'/image')
    h5write(fulldestination,dset_image, frameImage)
    dset_depth = strcat(dsetName,'-',str(frame_id),'/depth')
    h5write(fulldestination,dset_depth, frameDepth)
%    hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
end

%hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
