import os
import numpy as np
from Data_Generators.Image_Processors_Module.src.Processors.TFRecordWriter import dictionary_to_tf_record


example = {
    'Patient_Image': np.random.random((10, 512, 512)),
    'Patient_Mask': np.random.random((10, 512, 512)).astype('int'),
    'Patient_Name': 'John Doe',
    'Patient_Age': 53,
    'Patient_Weight': 123.4,
    'Patient_Location': 'Houston/San Diego'
}

dictionary_to_tf_record(os.path.join('.', 'TFRecords', 'Basic',
                                     'base_example.tfrecord'),
                        example)

example = {
    'Patient_Image': np.random.random((5, 128, 128)),
    'Patient_Mask': np.random.random((5, 128, 128)).astype('int'),
    'Patient_Name': 'Jane Doe',
    'Patient_Weight': 83.4,
    'Patient_Age': 34,
    'Patient_Location': 'LA'
}

dictionary_to_tf_record(os.path.join('.', 'TFRecords', 'Basic',
                                     'base_example2.tfrecord'),
                        example)
