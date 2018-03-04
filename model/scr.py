import Model
model = Model.Model()

if model.configuration.prediction_mode:
    model.predict(model.configuration.input_path,
                  model.configuration.output_path,
                  model.configuration.suffix)
else:
    # Pretrain
    model.train('Synth')

    # fine-tuning
    model.train('train')
