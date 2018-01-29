%% retrieve full dataset
sequenceName = 'hotel_umd/maryland_hotel3'
[K, frameCount, frameImage, frameDepth] = SUN3Dreader(sequenceName);

%% create local directory for data storage
destdirectory = '/home/kevin/ThesisDATA/SUN3D';
mkdir(destdirectory);   %create the directory

fulldestination = fullfile(destdirectory, 'myfile2.h5'); 

%% write data to HDF5 file

for frame_id = 1:frameCount
    hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
end

%hdf5write(fulldestination, dset_details, dset, attr_details, attr, 'WriteMode', 'append');
