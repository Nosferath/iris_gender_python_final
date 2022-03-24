# import numpy as np
# from PIL import Image

# from mask_pairs import absolute_growth
# from load_data import load_dataset_both_eyes
# from utils import find_dataset_shape


# def test_absolute_growth():
#     def retrieve_reshape_rgb(array, index, to_rgb=False):
#         """Retrieves the iris or mask, and reshapes to its original
#         shape. If to_rgb, the array is turned to RGB (3 channels).
#         """
#         out_array = array[index, :].reshape(orig_shape)
#         if to_rgb:
#             return np.tile(out_array[..., np.newaxis], (1, 1, 3))
#         return out_array

#     mask_ones = np.ones(10, dtype=bool)
#     mask_zeros = np.zeros(10, dtype=bool)
#     assert absolute_growth(mask_ones, mask_zeros) == 1, "Test 1 failed"
#     mask_half = np.array([True, False] * 5)
#     assert absolute_growth(mask_ones, mask_half) == 0.5, "Test 2 failed"
#     assert absolute_growth(mask_zeros, mask_half) == 0.5, "Test 3 failed"

#     dataset_name = '240x40_fixed'
#     orig_shape = tuple(int(i)
#                        for i in dataset_name.split('_')[0].split('x')[::-1])
#     all_data, _, _ = load_dataset_both_eyes(dataset_name)
#     left_data = all_data['left']
#     data_x = left_data[0]
#     data_m = left_data[2]
#     n_random_pairs = 10
#     random_idxs = np.random.choice(left_data[0].shape[0],
#                                    (2, n_random_pairs),
#                                    replace=False)
#     for i in range(n_random_pairs):
#         cur_pair = random_idxs[:, i]
#         # Reshape current iris into RGB rectangular images
#         iris_a = retrieve_reshape_rgb(data_x, cur_pair[0], to_rgb=True)
#         iris_b = retrieve_reshape_rgb(data_x, cur_pair[1], to_rgb=True)

#         # Generate visualization of original masks
#         mask_a = retrieve_reshape_rgb(data_m, cur_pair[0])
#         mask_b = retrieve_reshape_rgb(data_m, cur_pair[1])
#         iris_a_pre = iris_a.copy()
#         iris_a_pre[mask_a == 1] = [255, 0, 255]  # Magenta
#         iris_b_pre = iris_b.copy()
#         iris_b_pre[mask_b == 1] = [255, 0, 255]  # Magenta

#         # Generate visualization of paired masks
#         # - mask_ab_x is the union of the masks minus mask x
#         mask_ab_a = mask_b.copy()
#         mask_ab_a[mask_a == 1] = 0
#         iris_a_post = iris_a_pre.copy()
#         iris_a_post[mask_ab_a == 1] = [0, 255, 0]  # Green
#         mask_ab_b = mask_a.copy()
#         mask_ab_b[mask_b == 1] = 0
#         iris_b_post = iris_b_pre.copy()
#         iris_b_post[mask_ab_b == 1] = [0, 255, 0]  # Green

#         # Stack all images
#         print(iris_a_pre.shape)
#         stacked_img = np.vstack([iris_a_pre,
#                                  iris_a_post,
#                                  iris_b_pre,
#                                  iris_b_post])

#         # Visualize
#         stacked_img = Image.fromarray(stacked_img.astype('uint8'))
#         print(absolute_growth(mask_a, mask_b))
#         stacked_img.show()
#         input()

    # TODO APLICAR OOP AL PROBLEMA: CLASE IRIS QUE CONSERVE
    # MÁSCARA E IRIS. MATRIZ SPP QUE CONSERVE REFERENCIAS A
    # IRIS ORIGINALES. WRAPPER DE linear_sum_assignment QUE
    # RECIBA LOS OBJETOS IRIS Y RETORNE LAS PAREJAS YA HECHAS.


from constants import datasets_botheyes
from load_data_utils import partition_both_eyes
from load_data import load_dataset_both_eyes

for d in datasets_botheyes:
    all_data, males_set, females_set = load_dataset_both_eyes(d)
    _ = partition_both_eyes(all_data, males_set, females_set, 0.2, 1, True, d)
    