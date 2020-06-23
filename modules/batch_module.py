import numpy as np

batch_size = 0
input_img_size = 0

def getBatchSize(_batch_size):
    global batch_size
    batch_size = _batch_size
    print("Batch Size :", batch_size)

def generateBatchList(df, sample_limit=200):
    
    batch_list = []

    for begin in range(0, len(df), batch_size):

        end = min(begin+batch_size, len(df))-1
        n_sample = df['n_sample'].iloc[end]

        if ( n_sample > sample_limit):
            n_sample = sample_limit

        for s in range(n_sample):

            batch = []

            for b in range(end-begin+1):

                ## Do not shuffle
                nID = df.index[begin+b]
                coord = df['coord'].iloc[begin+b][0:n_sample]
                coord_x = coord[s][0]
                coord_y = coord[s][1]

                batch.append((nID, coord_x, coord_y))

            batch_list.append(batch)

    return batch_list

def groupBatch(sample_df, sample_limit=200):
    
    train_df = sample_df[sample_df['class']=='L']
    valid_df = sample_df[sample_df['class']=='V']
    
    batch_train = generateBatchList(train_df, sample_limit)
    batch_valid = generateBatchList(valid_df, sample_limit)
    
    n_train_steps = len(batch_train)
    n_valid_steps = len(batch_valid)

    n_train_total = (n_train_steps-1)*batch_size + len(batch_train[len(batch_train)-1])
    n_valid_total = (n_valid_steps-1)*batch_size + len(batch_valid[len(batch_valid)-1])

    print("Total Training images :", n_train_total)
    print("Total Validation images :", n_valid_total)
    print("Train steps:",n_train_steps)
    print("Valid steps:",n_valid_steps)
    
    return batch_train, batch_valid, n_train_steps, n_valid_steps

def generateBatchImages(im, batch, sample_df):
    
    x_batch = []
    y_batch = []

    for b in range(len(batch)):

        nID = batch[b][0]
        coord_x = batch[b][1]
        coord_y = batch[b][2]
        width = sample_df['resize_w'][nID]
        height = sample_df['resize_h'][nID]

        ## Select a sample from an total image
        crop_img, crop_mask = im.chooseCroppedRegions(nID, coord_x, coord_y, width, height)

        x_batch.append(crop_img)
        y_batch.append(crop_mask)

    x_batch = np.array(x_batch, np.float32) / 255
    y_batch = np.array(y_batch, np.float32)
    
    return x_batch, y_batch
