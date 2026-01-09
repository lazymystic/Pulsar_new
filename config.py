
class CFG:
    debug = False
    batch_size = 64
    sequence_length = 80
    num_classes = 2
    num_feats = 3
    lr = 1e-1
    min_lr = 1e-5
    epochs = 1
    print_freq = 100
    resume = False

    model_type = "AAGCN" # AAGCN, LSTM, STCONV
    dataset = "US_PARK_FT_200"
    experiment_id = "AAAI24_small_stream_J_Adaptive_PU_run1"

    add_feats = False
    add_phi = False

    add_joints1 = True     #Abl
    add_joints2 = True     #Abl    
    add_joints_mode = "ori"
    sam = False             #Abl
    loss_fn = "Focal"       #Abl Focal, BCE, BCEWithLogits

    ## only one of the stream can be true
    joint_stream = True    #Abl
    bone_stream = False       #Abl
    vel_stream = False  #Abl
    acc_stream = False  #Abl


    experiment_name = f"{experiment_id}__2{model_type}_{dataset}_{loss_fn}_seqlen{sequence_length}_{'SAM_' if sam else ''}{'joints1_' if add_joints1 else ''}{'joints2_' if add_joints2 else ''}{'joint' if joint_stream else ''}{'bone' if bone_stream else ''}{'vel' if vel_stream else ''}{'acc' if acc_stream else ''}"

    plot_weights = True
    
    if add_feats:
        num_feats = 6
    
    stconv_spatial_channels = 16
    stconv_out_channels = 64

    lstm_num_layers = 2
    lstm_hidden_layers = 120
    
    classes = ["POSITIVE", "NEGATIVE"]
