import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Importing Training set
path = 'heart.csv'
df = pd.read_csv(path)

# Converting Data Set into NumPy Array
df_features = df.copy()
df_labels = df_features.pop('target')
X = np.array(df_features)
Y = np.array(df_labels)

# Normalization
normalize = tf.keras.layers.Normalization(axis=-1)
normalize.adapt(X)
Xn = normalize(X)

# Sizing the Training Set
Xt = np.repeat(Xn, 1000, axis=0)
Yt = np.repeat(Y, 1000)

# Model Initiation
np.random.seed(42)
tf.random.set_seed(42)
model = Sequential(
    [
        Dense(3,activation ='sigmoid'),
        Dense(1,activation ='sigmoid')
    ]
)

# Model Compilation
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)



# Model Training
model.fit(
    Xt, Yt,
    epochs=10,

)

# Prediction (Validation)
X_Val = np.array([[63.0,1.0,3.0,145.0,200.0,1.0,0.,150.0,0.,2.3,0.,0.,1.0]])
X_test = normalize(X_Val)
predictions = model.predict(X_test)
print(predictions)




