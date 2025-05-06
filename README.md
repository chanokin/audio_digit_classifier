## Training a recurrent neural network (RNN) to categorize audio clips for spoken digits

### Data

The [free spoken digits](https://github.com/Jakobovski/free-spoken-digit-dataset) dataset consists of 3000 recordings of 6 participants speaking the numbers 0 to 9 in English; the set is comprised of .wav audio files.

I used a [proxy](https://github.com/eonu/torch-fsdd/) for this dataset which creates a *PyTorch* `Dataset` object from the recordings. While this was an invaluable bootstrap for the task, I ended [modifying](https://github.com/chanokin/torch-fsdd/) it due to version incompatibilities.

#### Preprocessing

As I had not worked with audio data directly, I searched the internet for prior work on this subject. The main takeaways were:

 1. A spectrogram is not the best transform for voice.
 2. The Mel-Frequency Cepstral Coefficients (MFCC) describes the "envelope" of the spectrum in a compact way and is preferred in speech recognition.
 3. A way to add noise to the input is to occlude an MFCC "channel" throughout the sample or drop a frame (time step) over all channels.

I split the data in train, validation and test subsets with a 70%, 15%, 15% proportion. All samples were transformed by trimming silences and the MFCC transform from the `torchaudio`library. Additionally, for the training data I added noisy via channel/step dropping; this was done with the purpose of enhancing the robustness of the network.

### Network architecture

Throughout the work I used *PyTorch* for modelling and *Pytorch-Lightning* for training. The network is composed of a `LSTM` layer and a `Linear` layer.

I chose an `LSTM` layer because it is recurrent, which satisfies the requirement of the task, and has been used in many time series problems in literature. Furthermore, the recurrence provides the layer a "memory" of past events (frames of the MFCC) which is used to produce an embedding which represents the sequence. The `Linear` layer maps the embedding to the class via an all-to-all connection.

### Classification

The baseline network has an `LSTM` with 128 hidden units; this results in an average of 96% test accuracy.
