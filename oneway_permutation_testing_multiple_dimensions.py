import numpy as np

def test_oneway(subject,frequency,output_path,condition_names,measure,list_of_data_ctb,p=0.05,permutations=1000): 
    #b are the dimensions distances are measured, resulting in one significance level
    #subject is a string given id of subject
    #frequency is a float in Hertz, for the save file
    #output_path is where you want to save the permutation results
    #condition_names is a list with two items which are strings describing the pair of conditions
    #measure is a string describing the dimensions measured
    #list_of_data_ctb are the scalar quantities over t time samples and b dimensions for n conditions
    #p is the significance level
    #permutations is the number of permutations in the statistical test
    output_file = 'pairwise_permutation_testing_%s_%s_%.3fHz_%s_vs_%s_p_%.2f_permutations_%d'%(subject,measure,frequency,condition_names[0],condition_names[1],p,permutations)
    sig_t = np.zeros((list_of_data_ctb[0].shape[1]),bool)
    cutoff_value,cutoff_sum = permutation_test_per_frequency_oneway(list_of_data_ctb,p,permutations)
    #print('cutoff_value',len(cutoff_value),cutoff_value)
    #print('cutoff_sum',cutoff_sum)
    #significance level over times
    sig_t = test_per_frequency_oneway(list_of_data_ctb,cutoff_value,cutoff_sum)
    #print('Saving',output_file+'.npy')
    np.save(output_path+output_file+'.npy',sig_t)
    return sig_t

def test_per_frequency_oneway(list_of_data_ctb,cutoff_values,cutoff_sum):
    means = []
    for l,data_ctb in enumerate(list_of_data_ctb): means += [data_ctb.mean(0)]
    mean_of_means  = np.array(means).mean(0)
    mean_distances = []
    for l,data_ctb in enumerate(list_of_data_ctb): mean_distances += [np.linalg.norm(data_ctb.mean(0)-mean_of_means,axis=-1)] #time
    mean_of_mean_distances = np.array(mean_distances).mean(0) #time
    count = 0
    summed_overs = 0
    significant = np.zeros((mean_of_mean_distances.shape[0]),bool)
    prev = False
    over_threshold = mean_of_mean_distances-cutoff_values
    #print('over_threshold',len(over_threshold),over_threshold)
    #print('summed_overs',end=' ')
    for t in range(mean_of_mean_distances.shape[0]):
        if over_threshold[t]<=0 and prev>0:
            if summed_overs>=cutoff_sum:
                significant[t-count+1:t+1] = True
                #print(summed_overs,end=', ')
            count = 0
            summed_overs = 0
        elif over_threshold[t]>0:
            summed_overs += over_threshold[t]+cutoff_values[t]
            count += 1
        prev = np.copy(over_threshold[t])
    #cover the exit case where there is a sum in cutoff that has not been used
    if over_threshold[t]>0 and summed_overs>=cutoff_sum:
        significant[t-count+1:t+1] = True
        #print(t-count+1,':',t+1)
        #print(summed_overs,end=', ')
    #print()
    return significant

def permutation_test_per_frequency_oneway(list_of_data_ctb,p=0.05,permutations=1000):
    value_cutoff_index = int(permutations*(1-p))
    data_shapes = []
    for data_ctb in list_of_data_ctb: data_shapes += [data_ctb.shape]
    perm_mean_distances = np.zeros((permutations,data_shapes[0][1])) #p,t
    rng = np.random.default_rng()
    means = []
    for l,data_ctb in enumerate(list_of_data_ctb): means += [data_ctb.mean(0)]
    mean_of_means  = np.array(means).mean(0)
    data_Ctb = np.concatenate(list_of_data_ctb,axis=0)
    for perm in range(permutations):
        mean_distances = []
        for l,data_ctb in enumerate(list_of_data_ctb):
            indexesl_c = np.random.choice(np.arange(data_Ctb.shape[0]),data_shapes[l][0],replace=True) #choose a random index from all cases to make up permuted group l
            permutedl_ctb = data_Ctb[indexesl_c,:,:]
            mean_distances += [np.linalg.norm(permutedl_ctb.mean(0)-mean_of_means,axis=-1)] #find the difference between the mean of the permuted group and the grand-mean; at each time but through dimensions b
        perm_mean_distances[perm,:] = np.array(mean_distances).mean(0)
    #print('mean_distances',mean_distances.shape)
    argsorted_distances = perm_mean_distances.argsort(0)
    distances_indexes = np.mgrid[0:permutations,0:data_shapes[0][1]] #p,t
    sorted_distances = perm_mean_distances[argsorted_distances,distances_indexes[1]]
    cutoff_values = sorted_distances[value_cutoff_index,:] #cut-off at each time
    over_threshold = perm_mean_distances-cutoff_values #time
    sequences = [] #sequences of frequency sums for consecutive significant pixels
    for perm in range(permutations):
        summed_overs = 0
        prev = -1
        for t in range(data_shapes[0][1]):
            if over_threshold[perm,t]<=0 and prev>0: #end summation process
                sequences += [summed_overs]
                summed_overs = 0
            elif over_threshold[perm,t]>0: summed_overs += over_threshold[perm,t]+cutoff_values[t] #get the actual frequency and add it to the sum over sequence
            prev = np.copy(over_threshold[perm,t])
        #cover the exit case where there is a sum in cutoff that has not been used
        if over_threshold[perm,t]>0: sequences += [summed_overs]
    sequences = np.array(sequences)
    #print(sequences)
    argsorted_sequences = sequences.argsort(0)
    sequence_cutoff_index = int(sequences.shape[0]*(1-p)) #threshold for 95% of sequences
    sorted_sequences = sequences[argsorted_sequences]
    cutoff_sum = sorted_sequences[sequence_cutoff_index]
    return cutoff_values,cutoff_sum
