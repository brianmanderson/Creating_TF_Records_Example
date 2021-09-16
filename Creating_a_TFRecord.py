from PlotScrollNumpyArrays import plot_scroll_Image
import os
import numpy as np
import SimpleITK as sitk
import Data_Generators.Image_Processors_Module.src.Processors.MakeTFRecordProcessors as Processors
from Data_Generators.Image_Processors_Module.src.Processors.TFRecordWriter import dictionary_to_tf_record


# Processors.LoadNifti(nifti_path_keys=('image_path', 'annotation_path'),
#                                          out_keys=('image_handle', 'annotation_handle')),
image_array = np.random.random((10, 128, 128)) * 10
annotation_array = np.zeros((10, 128, 128))
annotation_array[3:8, 60:90, 60:90] = 1

image_handle = sitk.GetImageFromArray(image_array)
image_handle.SetSpacing((1, 1, 1))

annotation_handle = sitk.GetImageFromArray(annotation_array)
annotation_handle.SetSpacing((1, 1, 1))

image_processors = [
    Processors.ResampleSITKHandles(resample_keys=('image_handle', 'annotation_handle'),
                                   resample_interpolators=('Linear', 'Nearest'),
                                   desired_output_spacing=(2, 2, 2)),
    Processors.NiftiToArray(nifti_keys=('image_handle', 'annotation_handle'),
                            out_keys=('image_array', 'annotation_array'),
                            dtypes=('float32', 'int8')),
    Processors.AddByValues(image_keys=('image_array',), values=(-5,)),
    Processors.DivideByValues(image_keys=('image_array',), values=(5,))
]

example_dictionary = {'image_handle': image_handle, 'annotation_handle': annotation_handle}
for processor in image_processors:
    processor.pre_process(example_dictionary)

dictionary_to_tf_record(filename=os.path.join('.', 'TFRecords', 'example.tfrecord'), input_features=example_dictionary)