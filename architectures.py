import numpy as np
import tensorflow as tf


class KAutomation(tf.keras.Model):
    def __init__(self, option, crop, it_lim=10, gamma=1.):
        super().__init__()
        self.it_lim = it_lim
        self.gamma = gamma
        self.option = option
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.CROP = crop

        input_shape = (crop, crop, 1)

        self.kappa = self.get_kappa()
        self.differential_operator = self.get_differential_operator()

        outputs = tf.keras.Input(shape=input_shape)
        grad_x, grad_y = self.differential_operator(outputs)

        deltaS, deltaE = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(outputs)
        E = tf.keras.layers.multiply((grad_y, deltaE))
        S = tf.keras.layers.multiply((grad_x, deltaS))

        NS = S
        EW = E
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-3)
        NS = tf.keras.layers.Concatenate(axis=1)([zeros_x, NS])
        EW = tf.keras.layers.Concatenate(axis=2)([zeros_y, EW])
        NS = tf.keras.layers.Lambda(lambda z: z[:, 1:] - z[:, :-1])(NS)
        EW = tf.keras.layers.Lambda(lambda z: z[:, :, 1:] - z[:, :, :-1])(EW)

        mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(gamma, dtype=tf.float32), z))(tf.ones_like(NS))

        adding = tf.keras.layers.add([NS, EW])
        adding = tf.keras.layers.multiply((mult, adding))

        self.diffusor = tf.keras.Model(outputs, adding, name='diffusing')

        inputs = tf.keras.Input(shape=input_shape)
        outputs = inputs

        for num_it in range(self.it_lim):
            adding = self.diffusor(outputs)
            outputs = tf.keras.layers.add((outputs, adding))

        self.denoiser = tf.keras.Model(inputs, outputs)

    def call(self, inputs, **kwargs):
        return self.denoiser(inputs, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self.denoiser(x)
            loss = tf.keras.losses.mean_squared_error(outputs, y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):

        x, y = data
        outputs = self.denoiser(x)
        loss = tf.keras.losses.mean_squared_error(outputs, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def get_kappa(self, kernel_size=3, pool_size=3, latent_size=1024):
        inputs = tf.keras.Input(shape=(self.crop, self.crop, 1))
        x = tf.keras.layers.Conv2D(32, kernel_size, strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(64, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        previous_block_activation = x

        for size in [16, 32, 64, 128]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(pool_size, strides=2, padding="same")(x)

            residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = tf.keras.layers.add([x, residual])
            previous_block_activation = x

        x = tf.keras.layers.SeparableConv2D(latent_size, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)

        return tf.keras.Model(inputs, x, name='kappa')

    def get_differential_operator(self):

        inputs = tf.keras.Input(shape=(self.crop, self.crop, 1))
        k = self.kappa(inputs)
        k = tf.keras.layers.Lambda(lambda z: 1. / (1. + tf.pow(z, 2.)))(k)
        k = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z, 1), 1))(k)
        grad_x, grad_y = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(inputs)
        grad_x = tf.keras.layers.Lambda(lambda z: tf.pow(z, 2))(grad_x)
        grad_y = tf.keras.layers.Lambda(lambda z: tf.pow(z, 2))(grad_y)

        grad_x = tf.keras.layers.multiply((k, grad_x))
        grad_y = tf.keras.layers.multiply((k, grad_y))
        if self.option == 1:
            grad_x = tf.keras.layers.Lambda(lambda z: tf.exp(-z))(grad_x)
            grad_y = tf.keras.layers.Lambda(lambda z: tf.exp(-z))(grad_y)
        if self.option == 2:
            grad_x = tf.keras.layers.Lambda(lambda z: 1. / (1. + z))(grad_x)
            grad_y = tf.keras.layers.Lambda(lambda z: 1. / (1. + z))(grad_y)

        return tf.keras.Model(inputs, [grad_x, grad_y], name='differential_operator')


class FoE(tf.keras.Model):
    def __init__(self, typee, degree, num_filters, crop, it_lim=10, num_classes=20, gamma=1.):
        super().__init__()
        self.it_lim = it_lim
        self.gamma = gamma
        self.typee = typee
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.degree = degree
        self.crop = crop

        input_shape = (crop, crop, 1)

        self.filter_experts = self.get_experts(input_shape)
        self.model_functions = self.get_model_functions(input_shape)
        self.differential_operator = self.get_differential_operator()

        outputs = tf.keras.Input(shape=input_shape)
        experts = tf.keras.Input(shape=(crop, crop, self.num_filters, 1))
        g = self.differential_operator([outputs, experts])

        deltaS, deltaE = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(outputs)
        E = tf.keras.layers.multiply((g, deltaE))
        S = tf.keras.layers.multiply((g, deltaS))

        NS = S
        EW = E
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-3)
        NS = tf.keras.layers.Concatenate(axis=1)([zeros_x, NS])
        EW = tf.keras.layers.Concatenate(axis=2)([zeros_y, EW])
        NS = tf.keras.layers.Lambda(lambda z: z[:, 1:] - z[:, :-1])(NS)
        EW = tf.keras.layers.Lambda(lambda z: z[:, :, 1:] - z[:, :, :-1])(EW)

        mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(gamma, dtype=tf.float32), z))(tf.ones_like(NS))

        adding = tf.keras.layers.add([NS, EW])
        adding = tf.keras.layers.multiply((mult, adding))

        self.diffusor = tf.keras.Model([outputs, experts], adding, name='diffusing')

        inputs = tf.keras.Input(shape=input_shape)
        outputs = inputs

        for num_it in range(self.it_lim):
            experts = self.filter_experts(outputs)
            adding = self.diffusor([outputs, experts])
            outputs = tf.keras.layers.add((outputs, adding))

        self.denoiser = tf.keras.Model(inputs, outputs)

    def call(self, inputs, **kwargs):

        return self.denoiser(inputs, **kwargs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            outputs = self.denoiser(x)
            loss = tf.keras.losses.mean_squared_error(outputs, y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):

        x, y = data

        outputs = self.denoiser(x)
        loss = tf.keras.losses.mean_squared_error(outputs, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def get_experts(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        experts = tf.keras.layers.Lambda(lambda z: tf.image.per_image_standardization(z))(inputs)
        experts = tf.keras.layers.Conv2D(self.num_filters, (self.degree, self.degree), padding='same',
                                         use_bias=False, activation='sigmoid', name='experts')(experts)

        experts = tf.keras.layers.Reshape((self.CROP, self.CROP, self.num_filters, 1))(experts)

        return tf.keras.Model(inputs, experts, name='filter_experts')

    def get_model_functions(self, input_shape):
        inp_ones = tf.keras.Input(shape=input_shape)
        crop_ones = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=1))(inp_ones)
        crop_ones = tf.keras.layers.add(crop_ones)
        crop_ones = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=1))(crop_ones)
        crop_ones = tf.keras.layers.add(crop_ones)
        crop_ones = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z, -1), -1))(crop_ones)
        crop_ones = tf.stop_gradient(tf.keras.layers.Lambda(lambda z: tf.ones_like(z))(crop_ones))

        if (self.typee == 'splines') or (self.typee == 'decreasing'):
            crop_zeros = tf.stop_gradient(tf.keras.layers.Lambda(lambda z: tf.zeros_like(z))(crop_ones))
            functions_m = tf.keras.layers.Conv2D(self.num_classes * self.num_filters, 1, padding='same',
                                                 use_bias=False)(crop_ones)
            functions_n = tf.keras.layers.Conv2D(self.num_filters, 1, padding='same')(crop_zeros)
            functions_n = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1))(functions_n)
            model_functions = tf.keras.Model(inp_ones, [functions_m, functions_n], name='getting_functions')

        elif self.typee == 'monomials':
            crop_zeros = tf.stop_gradient(tf.keras.layers.Lambda(lambda z: tf.zeros_like(z))(crop_ones))
            functions_m = tf.keras.layers.Conv2D(self.num_filters,
                                                 1, padding='same', use_bias=False, )(crop_ones)
            functions_n = tf.keras.layers.Conv2D(self.num_filters,
                                                 1, padding='same', )(crop_zeros)
            model_functions = tf.keras.Model(inp_ones, [functions_m, functions_n], name='getting_functions')
        elif self.typee == 'RothBlack':
            functions_m = tf.keras.layers.Conv2D(self.num_filters, 1, padding='same', use_bias=False,
                                                 activation='linear')(crop_ones)
            functions_m = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.exp(z), axis=-1))(functions_m)
            model_functions = tf.keras.Model(inp_ones, functions_m, name='getting_functions')

        return model_functions

    def get_differential_operator(self):
        inputs = tf.keras.Input(shape=(self.crop, self.crop, 1))

        if (self.typee == 'splines') or (self.typee == 'decreasing'):
            experts = tf.keras.Input(shape=(self.crop, self.crop, self.num_filters, 1))
            f, g = self.model_functions(inputs)
            f = tf.keras.layers.Reshape((1, 1, self.num_filters, self.num_classes))(f)
            g = tf.keras.layers.Lambda(lambda z: tf.pow(z, 2), name='g')(g)
            if self.typee == 'Decreasing':
                f = tf.keras.layers.Lambda(lambda z: -tf.pow(z, 2), name='g')(f)
            h = tf.keras.layers.Lambda(lambda z: z / self.num_classes)(f)

            partition_low = tf.constant(np.power(np.linspace(0, 1, self.num_classes + 1), 1)[:-1])
            partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low, 0), 0), 0)
            partition_low = tf.expand_dims(partition_low, 0)
            partition_low = tf.cast(partition_low, tf.float32)
            partition_up = tf.constant(np.power(np.linspace(0, 1, self.num_classes + 1), 1)[1:])
            partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up, 0), 0), 0)
            partition_up = tf.expand_dims(partition_up, 0)
            partition_up = tf.cast(partition_up, tf.float32)

            ineq1 = tf.keras.layers.Lambda(lambda z: tf.greater(z, partition_low))(experts)
            ineq2 = tf.keras.layers.Lambda(lambda z: tf.less_equal(z, partition_up))(experts)
            interval = tf.keras.layers.Lambda(lambda z: tf.cast(tf.math.logical_and(z[0], z[1]), tf.float32))(
                [ineq1, ineq2])

            inputs_mod = tf.keras.layers.Lambda(lambda z: tf.math.floormod(z, 1 / self.num_classes))(experts)
            interval = tf.keras.layers.multiply((inputs_mod, interval))

            full_experts = tf.keras.layers.multiply((f, interval))
            full_experts = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-1))(full_experts)
            full_experts = tf.keras.layers.add(full_experts)
            full_experts = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name='full_experts')(
                full_experts)

            ineq1_float = tf.keras.layers.Lambda(lambda z: 1. - tf.cast(z, tf.float32))(ineq2)
            bias = tf.keras.layers.multiply((ineq1_float, h))
            bias = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-1))(bias)
            bias = tf.keras.layers.add(bias)
            bias = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name='bias')(bias)

            functions = tf.keras.layers.add((bias, g, full_experts), name='functions')

            functions = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-2))(functions)
            functions = tf.keras.layers.add(functions)
            functions = tf.keras.layers.Lambda(lambda z: tf.exp(z), name='foe')(functions)

            differential_operator = tf.keras.Model([inputs, experts], functions, name='differential_operator')
            return differential_operator

        elif self.typee == 'monomials':
            experts_input = tf.keras.Input(shape=(self.crop, self.crop, self.num_filters, 1))
            f, g = self.model_functions(inputs)
            experts = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-2))(experts_input)
            f = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.unstack(z, axis=-1), axis=-1))(f)
            g = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.unstack(z, axis=-1), axis=-1))(g)

            functions = []
            for i in range(self.num_filters):
                ff = tf.keras.layers.Lambda(lambda z: z[0] - z[1])([experts[i], g[i]])
                ff = tf.keras.layers.Lambda(lambda z: tf.pow(z, i))(ff)
                ff = tf.keras.layers.multiply((f[i], ff))
                functions.append(ff)

            functions = tf.keras.layers.Concatenate()(functions)
            functions = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name='functions')(functions)
            functions = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-2))(functions)
            functions = tf.keras.layers.add(functions)
            functions = tf.keras.layers.Lambda(lambda z: tf.exp(z))(functions)
            differential_operator = tf.keras.Model([inputs, experts_input], functions, name='differential_operator')
            return differential_operator

        elif self.typee == 'RothBlack':
            experts_input = tf.keras.Input(shape=(self.crop, self.crop, self.num_filters, 1))
            experts = tf.keras.layers.Lambda(lambda z: 1. + 0.5 * tf.pow(z, 2.))(experts_input)
            f = self.model_functions(inputs)
            functions = tf.keras.layers.Lambda(lambda z: tf.pow(z[0], -z[1]))([experts, f])
            functions = tf.keras.layers.Lambda(lambda z: tf.unstack(z, axis=-2))(functions)
            functions = tf.keras.layers.multiply(functions)
            differential_operator = tf.keras.Model([inputs, experts_input], functions, name='differential_operator')
            return differential_operator


class UNet(tf.keras.Model):
    def __init__(self, crop, depth, degree, it_lim=10, gamma=0.005):
        super().__init__()
        self.it_lim = it_lim
        self.gamma = gamma
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.crop = crop
        self.depth = depth
        self.degree = degree

        input_shape = (crop, crop, 1)
        self.differential_operator = self.get_differential_operator(input_shape, depth, degree)

        outputs = tf.keras.Input(shape=input_shape)
        g = self.differential_operator(outputs)

        deltaS, deltaE = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(outputs)
        E = tf.keras.layers.multiply((g, deltaE))
        S = tf.keras.layers.multiply((g, deltaS))

        NS = S
        EW = E
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:, 1], axis=-3)
        NS = tf.keras.layers.Concatenate(axis=1)([zeros_x, NS])
        EW = tf.keras.layers.Concatenate(axis=2)([zeros_y, EW])
        NS = tf.keras.layers.Lambda(lambda z: z[:, 1:] - z[:, :-1])(NS)
        EW = tf.keras.layers.Lambda(lambda z: z[:, :, 1:] - z[:, :, :-1])(EW)

        mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(gamma, dtype=tf.float32), z))(tf.ones_like(NS))

        adding = tf.keras.layers.add([NS, EW])
        adding = tf.keras.layers.multiply((mult, adding))

        self.diffusor = tf.keras.Model(outputs, adding, name='diffusing')

        inputs = tf.keras.Input(shape=input_shape)
        outputs = inputs

        for num_it in range(self.it_lim):
            adding = self.diffusor(outputs)
            outputs = tf.keras.layers.add((outputs, adding))

        self.denoiser = tf.keras.Model(inputs, outputs)

    def call(self, inputs, **kwargs):

        return self.denoiser(inputs, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self.denoiser(x)
            loss = tf.keras.losses.mean_squared_error(outputs, y)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        outputs = self.denoiser(x)
        loss = tf.keras.losses.mean_squared_error(outputs, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def conv_block(self, x, n_filt, size_conv=(5, 5), n_conv=3):
        for c in range(n_conv):
            x = tf.keras.layers.Conv2D(n_filt, size_conv, padding="same", activation=None)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        return x

    def u_encoder(self, x, n_filt, degree=5):
        x = self.conv_block(x, n_filt, size_conv=(degree, degree))
        return tf.keras.layers.MaxPool2D()(x), x

    def u_decoder(self, pooled, skipped, n_filt, degree=5):
        upsampled = tf.keras.layers.Convolution2DTranspose(n_filt, (degree, degree), strides=(2, 2),
                                                           padding='same')(pooled)
        return self.conv_block(tf.keras.layers.concatenate([upsampled, skipped]), n_filt)

    def get_differential_operator(self, input_shape, depth=5, degree=5, output_channels=1, nfilt=2):
        skipped = []
        inp = tf.keras.Input(input_shape, name='input')
        p = inp
        for _ in range(depth):
            p, s = self.u_encoder(p, 2 ** (nfilt + _), degree=degree)
            skipped.append(s)
        p = self.conv_block(p, 2 ** (2 + depth))
        for _ in reversed(range(depth)):
            p = self.u_decoder(p, skipped[_], 2 ** (nfilt + _), degree=degree)
        p = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='relu')(p)
        return tf.keras.Model(inp, p, name='differential_operator')