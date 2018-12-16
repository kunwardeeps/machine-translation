# Machine Translation using just Numpy

LSTM Encoder Decoder model is used to implement.

Preprocessing recommended:
-Use fixed sized sentences
-Append <eos> to every sentence

To execute by loading existing models, use:
python lstmimpl.py 10 Y models/encoder_0.model models/decoder_0.model

To execute without loading existing models, use:
python lstmimpl.py 10 N

Arguments:
1. Number of epochs
2. Load existing models? (Y/N)
3. Path of existing encoder model
4. Path of existing decoder model