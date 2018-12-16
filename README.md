# Machine Translation using LSTM Encoder Decoder

Developed from scratch using just Numpy

Preprocessing recommended:
1. Use fixed sized sentences
2. Append <eos> to every sentence

## Execution
To execute by loading existing models, use:
python lstmimpl.py 10 Y models/encoder_0.model models/decoder_0.model

To execute without loading existing models, use:
python lstmimpl.py 10 N

Arguments:
1. Number of epochs
2. Load existing models? (Y/N)
3. Path of existing encoder model
4. Path of existing decoder model

## Screenshots
![Screenshot](https://github.com/kunwardeeps/machine-translation/blob/master/translations.png)

![Screenshot](https://github.com/kunwardeeps/machine-translation/blob/master/probabilities.png)
