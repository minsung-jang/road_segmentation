import pandas as pd

def Resize(total_df, n_resize, input_img_size):
    
    ## Data frame
    add_df = pd.DataFrame(columns=["resize_w", "resize_h", "n_sample", "coord"])

    ## [Customize] Fundamentals
    height = n_resize * input_img_size
    width = n_resize * input_img_size
    n_windows = n_resize * n_resize
    n_sample = n_windows

    ## [Customize] Record the coordinates of samples
    for nID, row in total_df.iterrows():

        ## Coordiate per image
        crop_coord_list = []        

        for crop_id in range(n_windows):

            i = crop_id % n_resize
            j = crop_id // n_resize
            x = i * input_img_size
            y = j * input_img_size

            crop_coord_list.append((x,y))

        add_df= add_df.append({'resize_w': width, 'resize_h': height, \
                               'n_sample': n_sample, 'coord': crop_coord_list}, ignore_index=True)
    
    
    ## Common parts
    sample_df = [ total_df, add_df ]
    sample_df = pd.concat(sample_df, axis=1)
    sample_df.index = total_df.index
    sample_df=sample_df.sort_values(['n_sample'], ascending=False)
    
    return sample_df