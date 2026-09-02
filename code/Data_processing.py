### Data_processing ###
# Asume image of full tissue to be segmented is available


## training patches sampling + masks ##
# use downloaded tissue samples and masks (from data -> slides)
# use random augmented sampling (randow orientation, scaling, stain?)
# download and divide sampled patches and masks in (data -> images -> train) and (data -> masks -> train)
# put location, orientation, size, stain? in train.csv (with slide id and patch id)


## validation / testing patches sampling ##
# use full pathological images and masks
# put grid with no augmentation
# download all sampled all patches + masks in (data-> images -> val,test) and (data-> masks -> val,test)
# put location of patches in val.csv and test.csv (with slide id and patch id)


## turn saved images into pytorch datasets? ##