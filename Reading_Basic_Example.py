from Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
import Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors as Processors

train_paths = [r'C:\Users\b5anderson\Desktop\Modular_Projects\TeachingTFRecords\TFRecords\Basic']
data_generator = DataGeneratorClass(record_paths=train_paths)
data_iterator = iter(data_generator.data_set)
data = next(data_iterator)
data2 = next(data_iterator)
xxx = 1
processors = [
    Processors.ExpandDimension(axis=-1, image_keys=('image_array', 'annotation_array'),),
    Processors.ReturnOutputs(input_keys=('image_array',), output_keys=('annotation_array',)),
    {'batch': 1},
    {'repeat'}
]
