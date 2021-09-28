import os
import numpy as np
from Data_Generators.Image_Processors_Module.src.Processors.TFRecordWriter import dictionary_to_tf_record

image_array = np.random.random((10, 128, 128)) * 10
annotation_array = np.zeros((10, 128, 128))
annotation_array[:, 30:90, 30:90] = 1
example = {
    'Patient_Image': image_array,
    'Patient_Mask': annotation_array,
    'Patient_Name': 'John Doe',
    'Patient_Age': 53,
    'Patient_Weight': 123.4,
    'Patient_Location': 'Houston/San Diego'
}

dictionary_to_tf_record(os.path.join('.', 'TFRecords', 'Basic',
                                     'base_example.tfrecord'),
                        example)

image_array = np.random.random((5, 128, 128)) * 10
annotation_array = np.zeros((5, 128, 128))
annotation_array[:, 30:90, 30:90] = 1
example = {
    'Patient_Image': image_array,
    'Patient_Mask': annotation_array,
    'Patient_Name': 'Jane Doe',
    'Patient_Weight': 83.4,
    'Patient_Age': 34,
    'Patient_Location': 'LA'
}

dictionary_to_tf_record(os.path.join('.', 'TFRecords', 'Basic',
                                     'base_example2.tfrecord'),
                        example)
