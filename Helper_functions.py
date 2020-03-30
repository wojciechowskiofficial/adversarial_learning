import numpy as np
import abc 

class Imbalance(abc.ABC):
    @staticmethod
    def get_set_distribution(labels):
        return dict(zip(*np.unique(labels, return_counts=True)))

    @staticmethod
    def get_set_parameters(labels):
        #returns tuple(no_parameters, mu if step_imbalance detected else None, rho)
        #mu = |{minority classes}| / no_classes
        #rho = max{|class_i|} / min{|class_i|}
        #dictionary.keys() are no instances of a class
        #dictionary.values() are no classes with no of instances equal to its corresponding key
        array_of_numbers = np.unique(labels, return_counts = True)
        dictionary = dict(zip(*np.unique(array_of_numbers[1], return_counts = True)))
        mu = None
        if len(dictionary) == 2:
            mu = dictionary[min(dictionary.keys())] / sum(dictionary.values())
        rho = max(dictionary.keys()) / min(dictionary.keys())
        return (mu, rho)

    @staticmethod
    def change_set_statistics(images: np.ndarray, labels: np.ndarray, mu, rho):
    #check tensor dimensions
    if images.shape[0] != labels.shape[0]:
        raise IndexError
    array_dict = dict()
    #set majority class size to the minimal avaliable class size
    majority_size = np.min(np.unique(labels, return_counts=True)[1])
    #extract every class from the images array, make it an array, shuffle it, 
    #and shorten in length such that len(every class) == majority_size
    for i in np.unique(labels):
        array_dict[i] = train_images[np.where(train_labels == i)]
        np.random.shuffle(array_dict[i])
        array_dict[i] = array_dict[i][:majority_size]
    #set minority class size
    minority_size = int(np.round(majority_size / rho))
    #if step imbalance variant chosen
    if mu is not None:
        #set numbers of minority and majroity classes
        minority_no = int(np.round(mu * len(array_dict)))
        majority_no = len(array_dict) - minority_no
        #create list of unique shuffled labels
        list_of_labels = np.unique(labels)
        np.random.shuffle(list_of_labels)
        #init output np arrays
        out_labels = np.empty(shape = tuple(labels.shape[i] if i != 0
                                            else 0
                                            for i in range(len(labels.shape))))
        out_images = np.empty(shape = tuple(images.shape[i] if i != 0 
                                            else 0
                                            for i in range(len(images.shape))))
        #add minority classes to concatenaded output tensors
        for i in range(minority_no):
            array_dict[list_of_labels[i]] = array_dict[list_of_labels[i]][:minority_size]
            tmp_labels = np.full(shape = tuple(labels.shape[i] if i != 0
                                            else minority_size
                                            for i in range(len(labels.shape))), 
                                 fill_value = list_of_labels[i])
            out_images = np.concatenate((out_images, array_dict[list_of_labels[i]]))
            out_labels = np.concatenate((out_labels, tmp_labels))
        for i in range(minority_no, len(array_dict), 1):
            tmp_labels = np.full(shape = tuple(labels.shape[i] if i != 0
                                            else majority_size
                                            for i in range(len(labels.shape))), 
                                 fill_value = list_of_labels[i])
            out_images = np.concatenate((out_images, array_dict[list_of_labels[i]]))
            out_labels = np.concatenate((out_labels, tmp_labels))
        return (out_images, out_labels)
    #todo: comment i jeszcze poshufflowac te out arrays np.permutations
    #todo: mu is None    
