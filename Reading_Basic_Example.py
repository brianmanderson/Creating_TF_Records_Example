from Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
import Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors as Processors
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

train_paths = [r'C:\Users\b5anderson\Desktop\Modular_Projects\TeachingTFRecords\TFRecords\Basic']
data_generator = DataGeneratorClass(record_paths=train_paths)
processors = [
    Processors.ExpandDimension(axis=-1, image_keys=('Patient_Image', 'Patient_Mask'),),
    Processors.ReturnOutputs(input_keys=('Patient_Image',), output_keys=('Patient_Mask',)),
    # {'repeat'}
]
data_generator.compile_data_set(image_processors=processors, debug=True)
data_iterator = iter(data_generator.data_set)
data = next(data_iterator)
data2 = next(data_iterator)
xxx = 1

