# suppress informational messages from TF
export TF_CPP_MIN_LOG_LEVEL=2 



python train.py --epochs=30 --run_test_set=True  
# python quantize.py  --tfl_file_name=trained_models/kws_model.tflite 
# python quantize_custom.py
# python eval_quantized_model.py --tfl_file_name=trained_models/kws_model.tflite 

