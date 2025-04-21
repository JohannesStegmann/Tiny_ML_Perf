import keras_model as models
import kws_util

Flags, unparsed = kws_util.parse_command()

model_settings = models.prepare_model_settings(12,Flags)

print(model_settings)