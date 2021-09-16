from Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
import Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors as Processors
from PlotScrollNumpyArrays import plot_scroll_Image

train_paths = [r'C:\Users\b5anderson\Desktop\Modular_Projects\TeachingTFRecords\TFRecords']

data_generator = DataGeneratorClass(record_paths=train_paths)

processors = [
    Processors.ExpandDimension(axis=-1, image_keys=('image_array', 'annotation_array'),),
    Processors.ReturnOutputs(input_keys=('image_array',), output_keys=('annotation_array',)),
    {'batch': 1},
    {'repeat'}
]

data_generator.compile_data_set(image_processors=processors)

x, y = next(iter(data_generator.data_set))
xxx = 1