## Ex 8-1

a) 

Unit test if `fit_linear` works properly.

b) 

Why reproducibility is important?

- Reproducebility is the key difference between science and alchemy.

How to ensure reproducibility?

- Do not use a GPU :(
  + Use `tensorflow` instead of `tensorflow-gpu`
  + Set `CPUDA_VISIBLE_DEVICES=""`
  + Use `tf.device()` blocks
    ```python
    with tf.device("/cpu:0"):
        ... # create ops
    ```
- Run single-threaded :(
  ```python
  config = tf.ConfigProto(intra_op_parallelism_threads=1,
           inter_op_parallelism_threads=1)
  with tf.Session(config=config) as sess:
      ...
  ```
- Set all random seeds!
  ```python
  random.seed(42)
  np.random.seed(42)
  # at the beginning or before tf.reset_default_graph() (i.e. before first random op is created)
  tf.set_random_seed(42)
  config = tf.estimator.RunConfig(tf_random_seed=42)
  dnn_clf = tf.estimator.DNNClassifier(..., config=config)
  input_fn = tf.estimator.inputs.numpy_input_fn(x={"X": xtrain, ..., shuffle=False})
  ```
- Eliminate any other source of variability

  ```python
  # results depends of OS
  files = os.listdir()
  files.sort() # should sort before use
  ```

c)

Loss is NaN.

d)

```python
from tensorflow.python import debug as tf_debug

...

if debug:
    session = tf_debug.LocalCLIDebugWrapperSession(session)
```

e)

```python
output = dense_layer(h, 'output_layer', 1)
output = sigmoid(output) # here cause negative outputs
```

## Ex 8-2

a)

- create summaries, `tf.summary.scalar()` `tf.summary.histogram()`
- merge summaries, `tf.summary.merge_all()`
- store summary on disk, `tf.summary.FileWriter()`
- use merged ops to run session

b)

...


c)

...

d)

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    experiment_dir, write_graph=True, write_images=True,
    histogram_freq=1)
callbacks = [tensorboard_callback]
```