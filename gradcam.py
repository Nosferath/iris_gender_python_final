import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from cnn_test import make_simple_cnn_model, make_cnn_model, prepare_x_array
from load_partitions import load_partitions


class GradCAM:
    def __init__(self, model, class_idx, layer_name=None):
        self.model = model
        self.class_idx = int(class_idx)
        self.layer_name = layer_name

        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        grad_model = Model(
            # inputs=[self.model.inputs[0]],
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                     self.model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, self.class_idx]

        grads = tape.gradient(loss, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        # print(grads)
        # print(type(grads))
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        w, h = image.shape[2], image.shape[1]
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    @staticmethod
    def overlay_heatmap(heatmap, image, alpha=0.5):
        colormap = cv2.COLORMAP_VIRIDIS

        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return heatmap, output


def visualize_gradcam(dataset, partition, dropout, simpler):
    # Load model and weights
    model_folder = "cnn_models"
    if dropout:
        model_folder += "_dropout"
    if simpler:
        model_folder += "_simpler"
    model_name = dataset + '_' + str(partition) + '.h5'
    orig_shape = (dataset.split('_')[-1]).split('x')
    orig_shape = (int(orig_shape[1]), int(orig_shape[0]))
    if simpler:
        model = make_simple_cnn_model(orig_shape, dropout)
    else:
        model = make_cnn_model(orig_shape, dropout)
    model.load_weights(model_folder + '/' + model_name)
    # Load dataset
    train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
        dataset, partition, mask_value=0, scale_dataset=True)
    test_x = prepare_x_array(test_x, orig_shape)
    # Select an image
    img_idx = 0
    img = test_x[img_idx]
    img_class = test_y[img_idx]
    img = np.expand_dims(img, 0)
    # Create GradCAM
    grad = GradCAM(model, img_class)
    # Visualize
    heatmap = grad.compute_heatmap(img)
    heatmap = cv2.resize(heatmap, (orig_shape[1], orig_shape[0]))
    orig_gray = (test_x[img_idx, :, :, 0]*255).astype('uint8')
    orig = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
    heatmap, output = grad.overlay_heatmap(heatmap, orig)
    output = np.vstack([orig, heatmap, output])
    return model, grad, test_x, test_y, heatmap, output


# grouped = df.groupby(['dataset', 'method'])
# with pd.option_context('display.max_columns', 40):
#     print(grouped.accuracy.describe().round(2))
