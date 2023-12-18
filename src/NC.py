from typing import Optional

import numpy as np


def image_transpose(image: np.ndarray):

    return np.stack((image[..., 0].T, image[..., 1].T, image[..., 2].T), axis=-1)


def transofrmed_domain_box_filter_hotizontal(
    image: np.ndarray, xform_domain_position: np.ndarray, box_radius: float
):

    H, W, C = image.shape

    l_pos = xform_domain_position - box_radius
    u_pos = xform_domain_position + box_radius

    l_idx = np.zeros_like(l_pos, dtype=np.int32)
    u_idx = np.zeros_like(u_pos, dtype=np.int32)

    for row in range(H):
        xform_domain_pos_row = np.concatenate([xform_domain_position[row, :], [np.inf]])

        l_pos_row = l_pos[row, :]
        u_pos_row = u_pos[row, :]

        local_l_idx = np.zeros((W,), dtype=np.int32)
        local_u_idx = np.zeros((W,), dtype=np.int32)

        local_l_idx[0] = np.argmax(xform_domain_pos_row > l_pos_row[0], axis=0)
        local_u_idx[0] = np.argmax(xform_domain_pos_row > u_pos_row[0], axis=0)

        for col in range(1, W):
            local_l_idx[col] = local_l_idx[col - 1] + np.argmax(
                xform_domain_pos_row[local_l_idx[col - 1] :] > l_pos_row[col]
            )
            local_u_idx[col] = local_u_idx[col - 1] + np.argmax(
                xform_domain_pos_row[local_u_idx[col - 1] :] > u_pos_row[col]
            )

        l_idx[row, :] = local_l_idx
        u_idx[row, :] = local_u_idx

    SAT = np.zeros([H, W + 1, C])
    SAT[:, 1:, :] = np.cumsum(image, axis=1)
    row_indices = np.tile(np.arange(H)[:, None], (1, W))

    F = np.zeros_like(image)
    D = np.zeros_like(image)

    for c in range(C):
        a = np.ravel_multi_index(
            (row_indices, l_idx, c * np.ones_like(row_indices)),
            SAT.shape,
        )
        b = np.ravel_multi_index(
            (row_indices, u_idx, c * np.ones_like(row_indices)),
            SAT.shape,
        )
        D[..., c] = u_idx - l_idx
        F[..., c] = (SAT.flat[b] - SAT.flat[a]) / (u_idx - l_idx)

    return F, D


def normalized_convolution(
    image: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    num_iterations: int = 3,
    joint_image: Optional[np.ndarray] = None,
):

    image = image.astype(np.float64)
    if joint_image is None:
        joint_image = image
    else:
        joint_image = joint_image.astype(np.float64)

    H, W, C = image.shape
    dIcdx = np.diff(joint_image, 1, 1)
    dIcdy = np.diff(joint_image, 1, 0)

    dIdx = np.zeros((H, W))
    dIdy = np.zeros((H, W))

    dIdx[:, 1:] = np.abs(dIcdx).sum(axis=-1)
    dIdy[1:, :] = np.abs(dIcdy).sum(axis=-1)

    dHdx = 1 + sigma_s / sigma_r * dIdx
    dVdy = 1 + sigma_s / sigma_r * dIdy

    ct_H = np.cumsum(dHdx, axis=1)
    ct_V = np.cumsum(dVdy, axis=0).T

    N = num_iterations
    F = image.copy()

    sigma_H = sigma_s

    for i in range(num_iterations):

        sigma_H_i = (
            sigma_H * np.sqrt(3) * np.power(2, N - i - 1) / np.sqrt(np.power(4, N) - 1)
        )

        box_radius = np.sqrt(3) * sigma_H_i

        F, D = transofrmed_domain_box_filter_hotizontal(F, ct_H, box_radius)
        F = image_transpose(F)
        F, D = transofrmed_domain_box_filter_hotizontal(F, ct_V, box_radius)
        F = image_transpose(F)

    return F, D
