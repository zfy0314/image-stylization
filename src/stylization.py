import cv2
import numpy as np

from NC import normalized_convolution


def vanilla(filename: str, output: str, sigma_s: float = 40, sigma_r: float = 0.77):

    image = cv2.imread(filename).astype(np.float64) / 255
    image_filtered, image_diff = normalized_convolution(
        image, sigma_s if sigma_s > 1 else max(image.shape) * sigma_s, sigma_r, 3
    )
    cv2.imwrite(output, (image_filtered * 255).astype(np.uint8))


def border(
    filename: str,
    output: str,
    scale: float = 3,
    dilation: int = 0,
    sigma: float = 0,
):

    image = cv2.imread(filename)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    dx, dy = np.gradient(image_gray)
    gradients = scale * np.abs(np.sqrt(np.square(dx) + np.square(dy)))
    if dilation > 0:
        gradients = cv2.dilate(gradients, np.ones((dilation, dilation)), iterations=1)
    if sigma > 1e-8:
        gradients = cv2.GaussianBlur(gradients, (0, 0), sigma)
    gradients = 1 - gradients.clip(0, 1)
    cv2.imwrite(output, (gradients * 255).astype(np.uint8))


def merge_border(
    filename: str, bordername: str, output: str, transparency: float = 1.0
):

    image_base = cv2.imread(filename)
    image_border = cv2.imread(bordername)
    image_base = (
        (image_base.astype(np.float64) * transparency).clip(0, 255).astype(np.uint8)
    )
    image_merged = np.minimum(image_base, image_border)
    cv2.imwrite(output, image_merged)


def style_transfer(
    filename: str,
    stylename: str,
    output: str,
    sigma_s: float = 40,
    sigma_r: float = 0.7,
):

    image = cv2.imread(filename).astype(np.float64) / 255
    h_img, w_img, _ = image.shape
    image_style = cv2.imread(stylename).astype(np.float64) / 255
    h_sty, w_sty, _ = image_style.shape

    image_style = image_style[:h_img, :w_img]

    image_filtered, image_diff = normalized_convolution(
        image,
        sigma_s if sigma_s > 1 else max(image.shape) * sigma_s,
        sigma_r,
        3,
        image_style,
    )
    cv2.imwrite(output, (image_filtered * 255).astype(np.uint8))


def run_statue():
    vanilla("data/statue.png", "data/statue_nc.png", sigma_s=40, sigma_r=0.77)
    border(
        "data/statue_nc.png", "data/statue_border.png", scale=7, dilation=2, sigma=0.7
    )
    merge_border(
        "data/statue_nc.png",
        "data/statue_border.png",
        "data/status_stylized.png",
        transparency=1.3,
    )


def run_burger():
    vanilla("data/burger.jpg", "data/burger_nc.png", sigma_s=0.03, sigma_r=0.8)
    vanilla("data/burger.jpg", "data/burger_nc_high.png", sigma_s=0.03, sigma_r=0.98)
    border(
        "data/burger_nc_high.png",
        "data/burger_border.png",
        scale=7,
        dilation=2,
        sigma=0.7,
    )
    merge_border(
        "data/burger_nc.png",
        "data/burger_border.png",
        "data/burger_stylized.png",
        transparency=1.5,
    )


def run_transfer():
    style_transfer(
        "data/burger.jpg",
        "data/stroke.jpg",
        "data/burger_stroke.png".format(r),
        sigma_s=0.03,
        sigma_r=1.5,
    )
