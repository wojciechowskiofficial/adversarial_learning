import numpy as np
import abc 
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

class Imbalance(abc.ABC):
    #mu = |{minority classes}| / no_classes
    #rho = max{|class_i|} / min{|class_i|}

    @abc.abstractmethod
    def get_set_distribution(labels):
        return dict(zip(*np.unique(labels, return_counts=True)))

    @abc.abstractmethod
    def get_set_parameters(labels):
        #returns tuple(no_parameters, mu if step_imbalance detected else None, rho)
        #dictionary.keys() are no instances of a class
        #dictionary.values() are no classes with no of instances equal to its corresponding key
        array_of_numbers = np.unique(labels, return_counts = True)
        dictionary = dict(zip(*np.unique(array_of_numbers[1], return_counts = True)))
        mu = None
        if len(dictionary) == 2:
            mu = dictionary[min(dictionary.keys())] / sum(dictionary.values())
        rho = max(dictionary.keys()) / min(dictionary.keys())
        return (mu, rho)

    @abc.abstractmethod
    def change_set_statistics(images: np.ndarray, labels: np.ndarray, mu, rho):
        #check tensor dimensions
        if images.shape[0] != labels.shape[0]:
            raise IndexError('tensor dimensions are not matching')
        #check parameter values
        if mu is not None and (mu <= 0 or mu >= 1):
            raise ValueError('mu is a real number in range (0; 1)')
        if rho < 1:
            raise ValueError('rho is a greater or equal than 1 real number')
        array_dict = dict()
        #set majority class size to the minimal avaliable class size
        majority_size = np.min(np.unique(labels, return_counts=True)[1])
        #extract every class from the images array, make it an array, shuffle it, 
        #and shorten in length such that len(every class) == majority_size
        for i in np.unique(labels):
            array_dict[i] = images[np.where(labels == i)]
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
                                                for i in range(len(labels.shape))), 
                                  dtype = np.int32)
            out_images = np.empty(shape = tuple(images.shape[i] if i != 0 
                                                else 0
                                                for i in range(len(images.shape))))
            #add minority classes to concatenaded output tensors
            for i in range(minority_no):
                array_dict[list_of_labels[i]] = array_dict[list_of_labels[i]][:minority_size]
                tmp_labels = np.full(shape = tuple(labels.shape[i] if i != 0
                                                else minority_size
                                                for i in range(len(labels.shape))), 
                                     fill_value = list_of_labels[i], 
                                     dtype = np.int32)
                out_images = np.concatenate((out_images, array_dict[list_of_labels[i]]))
                out_labels = np.concatenate((out_labels, tmp_labels))
            for i in range(minority_no, len(array_dict), 1):
                tmp_labels = np.full(shape = tuple(labels.shape[i] if i != 0
                                                else majority_size
                                                for i in range(len(labels.shape))), 
                                     fill_value = list_of_labels[i], 
                                     dtype = np.int32)
                out_images = np.concatenate((out_images, array_dict[list_of_labels[i]]))
                out_labels = np.concatenate((out_labels, tmp_labels))
        #if linear imabalnce variant chosen
        else:
            #fromalize namespace
            biggest_size = majority_size
            smallest_size = minority_size
            #create list of unique shuffled labels
            list_of_labels = np.unique(labels)
            np.random.shuffle(list_of_labels)
            #init output np arrays
            out_labels = np.empty(shape = tuple(labels.shape[i] if i != 0
                                                else 0
                                                for i in range(len(labels.shape))), 
                                  dtype = np.int32)
            out_images = np.empty(shape = tuple(images.shape[i] if i != 0 
                                                else 0
                                                for i in range(len(images.shape))))
            class_sizes = list(np.linspace(smallest_size, biggest_size, len(array_dict), dtype = np.int32))
            for (i, class_size) in zip(range(len(array_dict)), class_sizes):
                array_dict[list_of_labels[i]] = array_dict[list_of_labels[i]][:class_size]
                tmp_labels = np.full(shape = tuple(labels.shape[i] if i != 0
                                                else class_size
                                                for i in range(len(labels.shape))), 
                                     fill_value = list_of_labels[i], 
                                     dtype = np.int32)
                out_images = np.concatenate((out_images, array_dict[list_of_labels[i]]))
                out_labels = np.concatenate((out_labels, tmp_labels))
        #shuffle both of the final tensors in coequal manner
        permutation = np.random.permutation(np.arange(out_images.shape[0]))
        out_images = out_images[permutation]
        out_labels = out_labels[permutation]
        return (out_images, out_labels)

    @abc.abstractmethod
    def plot_set_distribution(images: np.ndarray, labels: np.ndarray):
        dictionary = Imbalance.get_set_distribution(labels)
        params = Imbalance.get_set_parameters(labels)
        if Imbalance.get_set_parameters(labels)[0] is not None:
            plt.bar(dictionary.keys(), dictionary.values())
            plt.show()
            print(f'mu = {params[0]}, rho = {params[1]}')
        else:
            print('sorted')
            (x, y) = (list(range(len(dictionary))), list(dictionary.values()))
            y.sort()
            plt.bar(x, y)
            plt.show()
            print('real')
            plt.bar(dictionary.keys(), dictionary.values())
            plt.show()
            print(f'rho = {params[1]}')

#use on tensorflow.tensor datatype 
class Adversarial(abc.ABC):
    @abc.abstractmethod
    def create_adversarial_masks(images, labels, loss_object, model):
        if images.shape[0] != labels.shape[0]:
            raise IndexError('tensor dimensions are not matching')
        model.trainable = False
        with tf.GradientTape() as tape:
            tape.watch(images)
            pred = model(images)
            loss = loss_object(labels, pred)
        signed_grad = tf.sign(tape.gradient(loss, images))
        return signed_grad
    
    @abc.abstractmethod
    def apply_masks(images, masks, epsilon):
        return tf.clip_by_value(images + epsilon * masks, 0, 1)

    @abc.abstractmethod
    def plot_epsilon_effectiveness(images, labels, loss_object, model, epsilons):
        def binary_diff():
            count = 0
            for i in np.arange(images.shape[0]):
                if np.argmax(predictions[i]) != np.argmax(adversarial_predictions[i]):
                    count += 1
            return count / images.shape[0]
        masks = Adversarial.create_adversarial_masks(images, labels, loss_object, model)
        output = dict()
        predictions = model.predict(images)
        for e in tqdm.tqdm(epsilons):
            adversarial_images = Adversarial.apply_masks(images, masks, e)
            adversarial_predictions = model.predict(adversarial_images)
            output[e] = binary_diff()
        plt.plot(list(output.keys()), list(output.values()))

    #takes numpy returns tensor!
    @abc.abstractmethod
    def oversample(images, labels, loss_object, model, eps, data_type = None):
        if data_type is None:
            raise AttributeError('specify imbalance type {"step"; "linear"}')
        elif data_type == 'step':
            images = np.true_divide(images, 255.0)
            distribution = Imbalance.get_set_distribution(labels)
            sorted_labels = [k for (k,_) in sorted(distribution.items(), key = lambda x:x[1])]
            tmp = distribution[sorted_labels[0]]
            minority_no = 0
            for k in sorted_labels:
                if distribution[k] == tmp:
                    minority_no += 1
                else:
                    break
            minority_size = distribution[sorted_labels[0]]
            majority_size = distribution[sorted_labels[-1]]
            labels_to_oversample = sorted_labels[:minority_no]
            for label in labels_to_oversample:
                current_images = tf.cast(images[np.where(labels == label)], dtype = tf.float32)
                current_labels = labels[np.where(labels == label)]
                masks = Adversarial.create_adversarial_masks(current_images, current_labels, loss_object, model).numpy()
                current_images = current_images.numpy()
                current_len = current_images.shape[0]
                masked_images = np.empty(shape = tuple(current_images.shape[i] if i != 0
                                                else majority_size - minority_size
                                                for i in range(len(current_images.shape))), 
                                  dtype = np.float32)
                masked_labels = np.full(shape = tuple(current_labels.shape[i] if i != 0
                                                else majority_size - minority_size
                                                for i in range(len(current_labels.shape))), 
                                     fill_value = label, 
                                     dtype = np.int32)
                i = 0
                while i < majority_size - minority_size:
                    masked_images[i] = np.clip(current_images[i % current_len] + np.random.choice(eps) * masks[i % current_len], 0, 1)
                    i += 1
                images = np.concatenate((images, masked_images))
                labels = np.concatenate((labels, masked_labels))
        permutation = np.random.permutation(np.arange(images.shape[0]))
        out_images = images[permutation]
        out_labels = labels[permutation]
        return (out_images * 255.0, out_labels)
