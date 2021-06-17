def unpack_minibatch(minibatch):
    '''
    input_tensor[batch_n, 0, dim, dim] : drivable region layer
                [       , 1,         ] : road marking layer
                [       , 2, label_dim, label_dim] : drivable region label
                [       , 3,                     ] : trajectory mask label
                [       , 4,                     ] : dir 'x' label
                [       , 5,                     ] : dir 'y' label
    '''
    input_tensor = minibatch[:,0:2,:]
    label_tensor = minibatch[:,2:,:]
    label_tensor = label_tensor[:,:,:128,:128]
        
    return (input_tensor, label_tensor)
