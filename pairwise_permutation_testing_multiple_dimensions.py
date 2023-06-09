import numpy as np

def test_pairwise(subject,frequency,output_path,condition_names,measure,data1_ctb,data2_ctb,p=0.05,permutations=1000): 
    #b are the dimensions distances are measured, resulting in one significance level
    #subject is a string given id of subject
    #frequency is a float in Hertz, for the save file
    #output_path is where you want to save the permutation results
    #condition_names is a list with two items which are strings describing the pair of conditions
    #measure is a string describing the dimensions measured
    #data1_ctb are the scalar quantities over t time samples and b dimensions for condition 1
    #data2_ctb are the scalar quantities over t time samples and b dimensions for condition 2
    #p is the significance level
    #permutations is the number of permutations in the statistical test
    output_file = 'pairwise_permutation_testing_%s_%s_%.3fHz_%s_vs_%s_p_%.2f_permutations_%d'%(subject,measure,frequency,condition_names[0],condition_names[1],p,permutations)
    sig_t = np.zeros((data1_ctb.shape[1]),bool)
    cutoff_value,cutoff_sum = permutation_test_per_frequency_pairwise(data1_ctb,data2_ctb,p,permutations)
    #print('cutoff_value',len(cutoff_value),cutoff_value)
    #print('cutoff_sum',cutoff_sum)
    #significance level over times
    sig_t = test_per_frequency_pairwise(data1_ctb,data2_ctb,cutoff_value,cutoff_sum)
    #print('Saving',output_file+'.npy')
    np.save(output_path+output_file+'.npy',sig_t)
    return sig_t

def test_per_frequency_pairwise(data1_ctb,data2_ctb,cutoff_values,cutoff_sum):
    mean_distances = np.linalg.norm(data1_ctb.mean(0)-data2_ctb.mean(0),axis=-1) #time
    count = 0
    summed_overs = 0
    significant = np.zeros((mean_distances.shape[0]),bool)
    prev = False
    over_threshold = mean_distances-cutoff_values
    #print('over_threshold',len(over_threshold),over_threshold)
    #print('summed_overs',end=' ')
    for t in range(mean_distances.shape[0]):
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

def permutation_test_per_frequency_pairwise(data1_ctb,data2_ctb,p=0.05,permutations=1000):
    value_cutoff_index = int(permutations*(1-p))
    data1_shape = data1_ctb.shape
    data2_shape = data2_ctb.shape
    perm_mean_distances = np.zeros((permutations,data1_shape[1])) #p,t
    rng = np.random.default_rng()
    for perm in range(permutations):
        set1 = np.random.choice(np.arange(2),data1_shape[0],replace=True) + 1 #randomly choose (1,2) from either group to make permuted group1
        indexes11_c = np.random.choice(np.arange(data1_shape[0]),(set1==1).sum(),replace=True) #for choice==1, choose a random index from group1 to make up permuted group1
        indexes12_c = np.random.choice(np.arange(data2_shape[0]),(set1==2).sum(),replace=True) #for choice==2, choose a random index from group2 to make up permuted group1
        permuted1_ctb = np.concatenate((data1_ctb[indexes11_c,:,:],data2_ctb[indexes12_c,:,:]),axis=0)
        set2 = np.random.choice(np.arange(2),data2_shape[0],replace=True) + 1 #randomly choose (1,2) from either group to make permuted group2
        indexes21_c = np.random.choice(np.arange(data1_shape[0]),(set2==1).sum(),replace=True) #for choice==1, choose a random index from group1 to make up permuted group2
        indexes22_c = np.random.choice(np.arange(data2_shape[0]),(set2==2).sum(),replace=True) #for choice==2, choose a random index from group2 to make up permuted group2
        permuted2_ctb = np.concatenate((data1_ctb[indexes21_c,:,:],data2_ctb[indexes22_c,:,:]),axis=0)
        mean_distances = np.linalg.norm(permuted1_ctb.mean(0)-permuted2_ctb.mean(0),axis=-1) #find the difference between the means of the two permuted groups; at each time but through dimensions b
        perm_mean_distances[perm,:] = mean_distances
    #print('mean_distances',mean_distances.shape)
    argsorted_distances = perm_mean_distances.argsort(0)
    distances_indexes = np.mgrid[0:permutations,0:data1_shape[1]] #p,t
    sorted_distances = perm_mean_distances[argsorted_distances,distances_indexes[1]]
    cutoff_values = sorted_distances[value_cutoff_index,:] #cut-off at each time
    over_threshold = perm_mean_distances-cutoff_values #time
    sequences = [] #sequences of frequency sums for consecutive significant pixels
    for perm in range(permutations):
        summed_overs = 0
        prev = -1
        for t in range(data1_shape[1]):
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
